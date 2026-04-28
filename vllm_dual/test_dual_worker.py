import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

# This test file lives next to a source-only ``vllm/`` directory.  Python puts
# the script directory at the front of ``sys.path``, so a plain ``import vllm``
# would accidentally load that source tree instead of the conda-installed vLLM
# package that contains the compiled CUDA extension ``vllm._C``.  Removing only
# this directory keeps the smoke test aligned with the real sync target.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from vllm import LLM, SamplingParams
from vllm.config import DualModelConfig
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

DEFAULT_REASONER_MODEL = "/home/disk1/public_checkpoint/Qwen3-1.7B/"
# The non-Instruct Qwen2.5-0.5B directory on this server is not readable by
# the current user, so the smoke test uses the readable 0.5B-Instruct copy.
DEFAULT_CONTROLLER_MODEL = "/home/disk1/public_checkpoint/Qwen2.5-0.5B-Instruct/"

MODE_FIELD_CANDIDATES = (
    "adversarial_mode",
    "adv_mode",
    "decoding_mode",
    "mode",
)
SOFT_ALPHA_FIELD_CANDIDATES = (
    "soft_student_weight",
    "adversarial_alpha",
    "adv_alpha",
    "soft_alpha",
    "alpha",
)
HARD_TOP_K_FIELD_CANDIDATES = (
    "hard_candidate_top_k",
    "hard_top_k",
    "teacher_top_k",
    "candidate_top_k",
)
HARD_TOP_P_FIELD_CANDIDATES = (
    "hard_candidate_top_p",
    "hard_top_p",
    "teacher_top_p",
    "candidate_top_p",
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run a short vLLM-dual smoke test.")
    parser.add_argument("--adv-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--reasoner-model", default=DEFAULT_REASONER_MODEL)
    parser.add_argument("--controller-model", default=DEFAULT_CONTROLLER_MODEL)
    parser.add_argument("--special-token-id", type=int, default=91799)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--small-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--small-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    parser.add_argument("--hard-top-k", type=int, default=16)
    parser.add_argument("--hard-top-p", type=float, default=0.95)
    parser.add_argument(
        "--strict-adversarial-config",
        action="store_true",
        help="Fail if the synced vLLM DualModelConfig cannot accept adv-mode.",
    )
    return parser.parse_args()


def dual_config_fields() -> set[str]:
    return set(getattr(DualModelConfig, "__dataclass_fields__", {}))


def set_first_supported(
    config: dict[str, object],
    supported_fields: set[str],
    candidates: tuple[str, ...],
    value: object,
) -> str | None:
    for field_name in candidates:
        if field_name in supported_fields:
            config[field_name] = value
            return field_name
    return None


def build_dual_model_config(
    args: Namespace,
) -> tuple[dict[str, object], dict[str, str | None]]:
    supported_fields = dual_config_fields()
    dual_model_config: dict[str, object] = {
        "small_model_engine_args": EngineArgs(
            model=args.controller_model,
            dtype="bfloat16",
            tensor_parallel_size=args.small_tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.small_gpu_memory_utilization,
            enforce_eager=True,
        ),
        "special_token_id": args.special_token_id,
    }

    forwarded = {
        "mode_field": set_first_supported(
            dual_model_config,
            supported_fields,
            MODE_FIELD_CANDIDATES,
            args.adv_mode,
        ),
        "soft_alpha_field": None,
        "hard_top_k_field": None,
        "hard_top_p_field": None,
    }
    if args.adv_mode == "soft":
        forwarded["soft_alpha_field"] = set_first_supported(
            dual_model_config,
            supported_fields,
            SOFT_ALPHA_FIELD_CANDIDATES,
            args.soft_alpha,
        )
    else:
        forwarded["hard_top_k_field"] = set_first_supported(
            dual_model_config,
            supported_fields,
            HARD_TOP_K_FIELD_CANDIDATES,
            args.hard_top_k,
        )
        forwarded["hard_top_p_field"] = set_first_supported(
            dual_model_config,
            supported_fields,
            HARD_TOP_P_FIELD_CANDIDATES,
            args.hard_top_p,
        )

    if args.strict_adversarial_config and forwarded["mode_field"] is None:
        raise RuntimeError(
            "Synced vLLM DualModelConfig does not expose an adversarial mode "
            f"field. Supported fields: {sorted(supported_fields)}"
        )
    return dual_model_config, forwarded


def main() -> None:
    """Run a short dual-worker smoke test.

    vLLM can use Python multiprocessing. When the start method is ``spawn``,
    child processes import this file again. Keeping the heavy LLM construction
    inside ``main()`` plus the ``if __name__ == "__main__"`` guard is the
    standard Python pattern that prevents every child process from recursively
    creating another engine.
    """
    args = parse_args()
    dual_model_config, forwarded = build_dual_model_config(args)
    supported_fields = ",".join(sorted(dual_config_fields()))
    forwarded_items = " ".join(
        f"{name}={value or 'none'}" for name, value in forwarded.items()
    )

    print(
        "SMOKE_DUAL_REQUEST "
        f"adv_mode={args.adv_mode} reasoner_model={args.reasoner_model} "
        f"controller_model={args.controller_model} max_tokens={args.max_tokens}"
    )
    print(f"SMOKE_DUAL_CONFIG_SUPPORTED fields={supported_fields}")
    print(f"SMOKE_DUAL_CONFIG_FORWARDED {forwarded_items}")

    llm = LLM(
        model=args.reasoner_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        dual_model_config=dual_model_config,
    )
    vllm_config = llm.llm_engine.vllm_config
    effective_dual_config = vllm_config.dual_model_config
    forwarded_mode_field = forwarded["mode_field"]
    effective_adv_mode = (
        getattr(effective_dual_config, forwarded_mode_field, "legacy")
        if forwarded_mode_field else "legacy"
    )
    print(
        "SMOKE_DUAL_EFFECTIVE "
        f"worker_cls={vllm_config.parallel_config.worker_cls} "
        f"dual_config_cls={type(effective_dual_config).__name__} "
        f"adv_mode={effective_adv_mode} "
        f"special_token_id={getattr(effective_dual_config, 'special_token_id', 'missing')}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.reasoner_model,
        trust_remote_code=True,
    )
    raw_prompts = [
        "What is $\\sum_{i=1}^\\infty C_{2i}^{i+1}x^i$? Please think step by step and put your final answer within \\boxed{}.",
        "What is $\\sum_{i=2}^\\infty C_{2i}^{i+2}x^i$? Please think step by step and put your final answer within \\boxed{}.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in raw_prompts
    ]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=1,
    )
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    print(f"Time taken: {time.perf_counter() - start_time:.2f} seconds")

    for idx, output in enumerate(outputs, start=1):
        print(f"\n=== Prompt {idx} ===")
        print(output.outputs[0].text.strip())


if __name__ == "__main__":
    main()

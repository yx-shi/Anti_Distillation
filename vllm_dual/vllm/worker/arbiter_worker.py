import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from vllm.worker.worker_base import WorkerBase, DelegateWorkerBase
from vllm.config import VllmConfig, ArbiterModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelRequest, CompletionSequenceGroupOutput
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import SequenceOutput, Logprob
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class SmallerTpWorkerWrapper(WorkerBase):
    """
    与 /vllm/worker/dual_worker.py 保持一致的 Smaller-TP wrapper。
    """

    def __init__(
        self,
        worker: DelegateWorkerBase,
        draft_ranks: List[int],
    ):
        self._worker = worker
        self._draft_ranks = draft_ranks
        self._tp_group = None
        self._is_dummy = False

    def _patch_tp(self):
        return patch_tensor_parallel_group(self._tp_group)

    def init_device(self) -> None:
        global_tp = get_tp_group()
        self._is_dummy = global_tp.rank not in self._draft_ranks
        if self._is_dummy:
            logger.info("Rank %d is dummy for smaller-tp worker", global_tp.rank)
            return

        local_rank = global_tp.local_rank
        backend = torch.distributed.get_backend(global_tp.device_group)
        self._tp_group = init_model_parallel_group([self._draft_ranks], local_rank, backend)
        with self._patch_tp():
            self._worker.init_device()

    def _get_rank_in_group(self):
        global_tp = get_tp_group()
        return global_tp.rank_in_group

    def load_model(self) -> None:
        if self._is_dummy:
            return
        with self._patch_tp():
            self._worker.load_model()

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks) -> None:
        if self._is_dummy:
            return
        with self._patch_tp():
            self._worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def determine_num_available_blocks(self):
        if self._is_dummy:
            return -1, -1
        with self._patch_tp():
            return self._worker.determine_num_available_blocks()

    def get_cache_block_size_bytes(self) -> int:
        if self._is_dummy:
            return 0
        with self._patch_tp():
            return self._worker.get_cache_block_size_bytes()

    @torch.inference_mode()
    def execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if self._is_dummy:
            return []
        with self._patch_tp():
            return self._worker.execute_model(execute_model_req)

    @property
    def rank(self):
        return self._worker.rank

    @property
    def device(self):
        return self._worker.device

    @property
    def vocab_size(self):
        return self._worker.vocab_size


class HiddenStateArbiter(nn.Module):
    """
    简化版 gate 网络：concat([r_h, c_h, p_h]) -> logit -> sigmoid -> Bernoulli。
    实际结构请按你 arbiter_model.ControlledReasonerHiddenState 的 gate 网络替换。
    """

    def __init__(self, in_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 1),
        )
        self.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (N,)

    def load_weight(self, state_dict: Dict[str, torch.Tensor], prefix: str = "") -> None:
        # 直接匹配 net.*
        if any(k.startswith("net.") for k in state_dict.keys()):
            self.load_state_dict(state_dict, strict=False)
            return

        # 尝试匹配 {prefix}arbiter. 或 arbiter.
        candidates = [f"{prefix}arbiter.", "arbiter.", prefix]
        for cand in candidates:
            sub = {k[len(cand):]: v for k, v in state_dict.items() if k.startswith(cand)}
            if any(k.startswith("net.") for k in sub.keys()):
                self.load_state_dict(sub, strict=False)
                return

        raise KeyError("Cannot find arbiter weights in state_dict (expect net.* or *arbiter.net.*).")


class HiddenStateArbiterWorker(WorkerBase):
    """
    三路 worker：
      - reasoner: 使用当前 vllm_config 指定的 model（即 LLM(model=reasoner_model)）
      - controller: cfg.controller_engine_args
      - prefix: cfg.prefix_engine_args

    仲裁：
      - 要求三路 SamplerOutput.hidden_states 非空（由 vLLM 模型执行器回填）
      - token 用 reasoner/controller 各自的 sample output_token（保持与 dual_worker 一致）
      - final token = choice ? controller : reasoner

    备注：
      - 仍然保留 “不同 worker 不同 cache layout/CacheEngine” 的路径（各自独立 Worker 实例）。
      - block 数分配这里先做保守版：取三者中最小的可用 blocks。
    """

    def __init__(self, *args, **kwargs):
        vllm_config: VllmConfig = kwargs["vllm_config"]
        cfg: ArbiterModelConfig = vllm_config.arbiter_model_config
        assert cfg is not None, "vllm_config.arbiter_model_config is required"

        self.cfg = cfg

        # reasoner 使用外部传入的 vllm_config（即当前大 worker / 主 worker）
        # controller/prefix 用各自 EngineArgs.create_engine_config()
        self.controller_cfg = cfg.controller_engine_args.create_engine_config()
        self.prefix_cfg = cfg.prefix_engine_args.create_engine_config()

        self.reasoner_tp = vllm_config.parallel_config.tensor_parallel_size
        self.controller_tp = self.controller_cfg.parallel_config.tensor_parallel_size
        self.prefix_tp = self.prefix_cfg.parallel_config.tensor_parallel_size

        self.max_tp_size = max(self.reasoner_tp, self.controller_tp, self.prefix_tp)
        self.min_tp_size = min(self.reasoner_tp, self.controller_tp, self.prefix_tp)

        from vllm.worker.worker import Worker

        reasoner_worker = Worker(*args, **kwargs)
        if self.reasoner_tp != self.max_tp_size:
            reasoner_worker = SmallerTpWorkerWrapper(
                reasoner_worker,
                draft_ranks=list(range(self.reasoner_tp)),
            )

        controller_worker = Worker(
            *args,
            vllm_config=self.controller_cfg,
            **{k: v for k, v in kwargs.items() if k != "vllm_config"},
        )
        if self.controller_tp != self.max_tp_size:
            controller_worker = SmallerTpWorkerWrapper(
                controller_worker,
                draft_ranks=list(range(self.controller_tp)),
            )
        prefix_worker = Worker(
            *args,
            vllm_config=self.prefix_cfg,
            **{k: v for k, v in kwargs.items() if k != "vllm_config"},
        )
        if self.prefix_tp != self.max_tp_size:
            prefix_worker = SmallerTpWorkerWrapper(
                prefix_worker,
                draft_ranks=list(range(self.prefix_tp)),
            )

        self.reasoner_worker = reasoner_worker
        self.controller_worker = controller_worker
        self.prefix_worker = prefix_worker

        self.arbiter: Optional[HiddenStateArbiter] = None
        self._arbiter_in_dim: Optional[int] = None

        from vllm.platforms import current_platform
        self.current_platform = current_platform

    # ---------------- lifecycle ----------------
    def init_device(self) -> None:
        # init & load（对齐 dual_worker 的顺序：先 init_device，再 load_model）
        self.reasoner_worker.init_device()
        self.controller_worker.init_device()
        self.prefix_worker.init_device()

        self.reasoner_worker.load_model()
        self.controller_worker.load_model()
        self.prefix_worker.load_model()

        # 推断 hidden sizes（尽量走 vLLM worker 的 model_runner.model.config）
        def _get_hidden_size(w) -> int:
            base = getattr(w, "_worker", w)  # wrapper / raw
            mr = getattr(base, "model_runner", None)
            print(f'{mr=}')
            m = getattr(mr, "model", None) if mr is not None else None
            print(f'{m=}')
            mc = getattr(m, "config", None) if m is not None else None
            print(f'{mc=}')
            hs = getattr(mc, "hidden_size", None) if mc is not None else None
            if hs is None and mc is not None:
                hs = getattr(mc, "n_embd", None)
            if hs is None:
                raise RuntimeError("Cannot infer hidden_size from worker model config for worker {}".format(type(w)))
            return int(hs)

        # Hr = _get_hidden_size(self.reasoner_worker)
        # Hc = _get_hidden_size(self.controller_worker)
        # Hp = _get_hidden_size(self.prefix_worker)
        # self._arbiter_in_dim = Hr + Hc + Hp
        self._arbiter_in_dim = self.cfg.arbiter_input_dim

        self.arbiter = HiddenStateArbiter(self._arbiter_in_dim).to("cuda").eval()

        # load arbiter weights（可选）
        if self.cfg.arbiter_state_dict is not None:
            self.arbiter.load_weight(self.cfg.arbiter_state_dict, prefix=self.cfg.arbiter_state_dict_prefix)
        elif self.cfg.arbiter_ckpt_path is not None:
            sd = torch.load(self.cfg.arbiter_ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            if not isinstance(sd, dict):
                raise TypeError(f"arbiter_ckpt_path loaded object is not a state_dict: {type(sd)}")
            self.arbiter.load_weight(sd, prefix=self.cfg.arbiter_state_dict_prefix)
        else:
            logger.warning("HiddenStateArbiterWorker: arbiter weights not provided, using random init.")

        logger.info("HiddenStateArbiterWorker init done. in_dim=%d", self._arbiter_in_dim)

    def load_model(self, *args, **kwargs):
        # vLLM 可能会再次调用，保持与 dual_worker 一样 no-op
        pass

    def load_weights(self, weights):
        for name, weight in weights:
            if name.startswith("arbiter."):
                assert self.arbiter is not None
                sub_name = name[len("arbiter.") :]
                self.arbiter.load_weight({sub_name: weight}, prefix="")
                logger.info("Loaded arbiter weight: %s", sub_name)
            elif name.startswith("controller."):
                sub_name = name[len("controller.") :]
                self.controller_worker.load_weights([(sub_name, weight)])
                logger.info("Loaded controller weight: %s", sub_name)
            elif name.startswith("prefix."):
                sub_name = name[len("prefix.") :]
                self.prefix_worker.load_weights([(sub_name, weight)])
                logger.info("Loaded prefix weight: %s", sub_name)

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
        # 简单策略：三路都用同样 blocks（更精确的 bytes 平分需要扩展 vLLM 接口）
        self.reasoner_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self.controller_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self.prefix_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def determine_num_available_blocks(self):
        # 根据他们分别占满整个gpu可以获得的block数，加权平均
        r_gpu, r_cpu = self.reasoner_worker.determine_num_available_blocks()
        c_gpu, c_cpu = self.controller_worker.determine_num_available_blocks()
        p_gpu, p_cpu = self.prefix_worker.determine_num_available_blocks()
        print('original num available blocks:')
        print(f'  reasoner: gpu={r_gpu} cpu={r_cpu}')
        print(f'  controller: gpu={c_gpu} cpu={c_cpu}')
        print(f'  prefix: gpu={p_gpu} cpu={p_cpu}')
        # 加权平均
        r_block_bytes = self.reasoner_worker.get_cache_block_size_bytes()
        c_block_bytes = self.controller_worker.get_cache_block_size_bytes()
        p_block_bytes = self.prefix_worker.get_cache_block_size_bytes()
        total_block_bytes = r_block_bytes + c_block_bytes + p_block_bytes
        return (
            int((r_gpu * r_block_bytes) / total_block_bytes),
            r_cpu
        )

    # ---------------- execution ----------------
    @torch.inference_mode()
    def execute_model(self, execute_model_req):
        out_r = self.reasoner_worker.execute_model(execute_model_req)
        out_c = self.controller_worker.execute_model(execute_model_req)
        out_p = self.prefix_worker.execute_model(execute_model_req)

        # dummy ranks：wrapper 会返回 []
        if out_r == [] and out_c == [] and out_p == []:
            return []

        # 只要 reasoner 是 dummy，就按 dual_worker 行为直接返回另一侧
        if hasattr(self.reasoner_worker, "_is_dummy") and self.reasoner_worker._is_dummy:
            return out_c

        if hasattr(self.controller_worker, "_is_dummy") and self.controller_worker._is_dummy:
            return out_r

        if out_r is None and out_c is None:
            return None

        if out_r == [] and out_c == []:
            return []

        # 取 SamplerOutput
        sr = out_r[0]
        sc = out_c[0]
        sp = out_p[0]

        if sr.hidden_states is None or sc.hidden_states is None or sp.hidden_states is None:
            raise RuntimeError(
                "SamplerOutput.hidden_states is None. "
                "需要在 vLLM 模型执行链路中启用并回填 hidden_states 到 SamplerOutput。"
            )

        r_h = sr.hidden_states
        c_h = sc.hidden_states
        p_h = sp.hidden_states

        # out_r/out_c 的 tokens：沿用 dual_worker 的方式，从 outputs 里遍历 sample.output_token
        out_r_groups = sr.outputs
        out_c_groups = sc.outputs
        assert len(out_r_groups) == len(out_c_groups)

        tokens_r: List[int] = []
        tokens_c: List[int] = []
        for gr, gc in zip(out_r_groups, out_c_groups):
            assert len(gr.samples) == len(gc.samples)
            for s_r, s_c in zip(gr.samples, gc.samples):
                tokens_r.append(s_r.output_token)
                tokens_c.append(s_c.output_token)

        tokens_r_t = torch.tensor(tokens_r, device=r_h.device, dtype=torch.long)
        tokens_c_t = torch.tensor(tokens_c, device=r_h.device, dtype=torch.long)

        # gate + choice
        assert self.arbiter is not None and self._arbiter_in_dim is not None
        gate_in = torch.cat([r_h, c_h, p_h], dim=-1)
        if gate_in.dim() != 2:
            gate_in = gate_in.view(gate_in.size(0), -1)
        if gate_in.size(-1) != self._arbiter_in_dim:
            raise RuntimeError(f"arbiter input dim mismatch: got {gate_in.size(-1)} expect {self._arbiter_in_dim}")

        gate_logits = self.arbiter(gate_in) / max(float(self.cfg.gate_temperature), 1e-6)
        # import pdb; pdb.set_trace()
        # print(f'{gate_logits=}')
        gate = torch.sigmoid(gate_logits)  # (N,)
        choice = torch.rand_like(gate) < gate  # True => choose controller
        final_tokens_t = torch.where(choice, tokens_c_t, tokens_r_t)

        final_tokens = final_tokens_t.tolist()
        mask = choice.tolist()  # is_intervened

        # 组装 merged outputs（参考 dual_worker）
        merged_outputs = []
        i = 0
        for group_r in out_r_groups:
            samples = []
            for sample_r in group_r.samples:
                merged_sample = SequenceOutput(
                    parent_seq_id=sample_r.parent_seq_id,
                    output_token=final_tokens[i],
                    is_intervened=mask[i],
                    logprobs={
                        final_tokens[i]: Logprob(
                            logprob=0.0,
                            rank=None,
                            decoded_token=None,
                        )
                    }
                )
                samples.append(merged_sample)
                i += 1
            merged_group = CompletionSequenceGroupOutput(samples=samples, prompt_logprobs=None)
            merged_outputs.append(merged_group)

        merged_sampler_output = SamplerOutput(
            outputs=merged_outputs,
            hidden_states=sr.hidden_states,  # 保留 reasoner hidden_state（可选）
        )

        return [merged_sampler_output]

    # ---------------- proxy ----------------
    @property
    def rank(self):
        return self.reasoner_worker.rank

    @property
    def device(self):
        return self.reasoner_worker.device
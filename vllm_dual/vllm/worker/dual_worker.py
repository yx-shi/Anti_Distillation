import torch
import copy
from typing import Dict, List, Optional, Tuple
from transformers import AutoConfig, PretrainedConfig

from vllm.worker.worker_base import WorkerBase, DelegateWorkerBase
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelRequest, CompletionSequenceGroupOutput
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import VLLM_INVALID_TOKEN_ID, SequenceOutput, Logprob
from vllm.spec_decode.util import create_sequence_group_output
# SequenceOutput
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
from vllm.utils import resolve_obj_by_qualname
from vllm.logger import init_logger

logger = init_logger(__name__)


class SpecialTokenArbiter:
    def __init__(self, special_token_id: int):
        self.special_token_id = special_token_id

    def choose(self, small_token_id: torch.LongTensor, large_token_id: torch.LongTensor) -> Tuple[torch.BoolTensor, torch.LongTensor]:
        """
        Choose between small and large model token ids based on the special token.

        Args:
            small_token_id: Tensor of token ids from the small model
            large_token_id: Tensor of token ids from the large model
        Returns:
            A tuple of (mask, final_token_ids) where mask indicates which tokens were taken from the small model.
        """
        # Implement vectorized version
        mask = (small_token_id == self.special_token_id)
        return ~mask, torch.where(mask, large_token_id, small_token_id)


class SmallerTpWorkerWrapper(WorkerBase):
    """
    Worker wrapper that allows an inner worker to run with a smaller
    tensor parallel size than the global TP group.

    This is done by:
      - selecting a subset of TP ranks
      - creating a private TP group
      - patching the global TP group during forward / init / cache ops
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patch_tp(self):
        return patch_tensor_parallel_group(self._tp_group)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        global_tp = get_tp_group()
        self._is_dummy = global_tp.rank not in self._draft_ranks

        if self._is_dummy:
            logger.info(
                "Rank %d is dummy for smaller-tp worker", global_tp.rank)
            return

        local_rank = global_tp.local_rank
        backend = torch.distributed.get_backend(global_tp.device_group)

        self._tp_group = init_model_parallel_group(
            [self._draft_ranks],
            local_rank,
            backend,
        )

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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:

        if self._is_dummy:
            return []

        with self._patch_tp():
            return self._worker.execute_model(execute_model_req)

    # ------------------------------------------------------------------
    # Proxy attributes
    # ------------------------------------------------------------------

    @property
    def rank(self):
        return self._worker.rank

    @property
    def device(self):
        return self._worker.device

    @property
    def vocab_size(self):
        return self._worker.vocab_size
    

# def create_dual_worker(*args, **kwargs):


class DualModelWorker(WorkerBase):
    def __init__(self, *args, **kwargs):
        vllm_config: VllmConfig = kwargs["vllm_config"]
        dual_cfg = vllm_config.dual_model_config
        assert dual_cfg is not None
        vllm_config.parallel_config.tensor_parallel_size = dual_cfg.large_worker_tensor_parallel_size
        small_engine_args: EngineArgs = dual_cfg.small_model_engine_args
        small_cfg = small_engine_args.create_engine_config()
        print('small_config: {}'.format(small_cfg))

        self.large_actually_small = vllm_config.parallel_config.tensor_parallel_size < small_cfg.parallel_config.tensor_parallel_size

        draft_ranks = list(range(min(vllm_config.parallel_config.tensor_parallel_size, small_cfg.parallel_config.tensor_parallel_size)))

        # ---------- Large model worker ----------
        from vllm.worker.worker import Worker
        large_worker = Worker(
            *args, 
            **kwargs
        )

        if self.large_actually_small:
            large_worker = SmallerTpWorkerWrapper(
                worker=large_worker,
                draft_ranks=draft_ranks,
            )

        # ---------- Small model config ----------

        small_worker = Worker(
            *args,
            vllm_config=small_cfg,
            **{k: v for k, v in kwargs.items() if k != "vllm_config"},
        )

        if not self.large_actually_small:
            small_worker = SmallerTpWorkerWrapper(
                worker=small_worker,
                draft_ranks=draft_ranks,
            )

        # ---------- Arbiter ----------
        special_token_id = dual_cfg.special_token_id

        arbiter = SpecialTokenArbiter(special_token_id)

        self.small_worker = small_worker
        self.large_worker = large_worker
        self.arbiter = arbiter
        self.dual_cfg = dual_cfg
        self.adversarial_mode = dual_cfg.adversarial_mode.lower()
        if self.adversarial_mode not in ("special_token", "hard", "soft"):
            raise ValueError(
                "dual_model_config.adversarial_mode must be one of "
                "'special_token', 'hard', or 'soft', got "
                f"{dual_cfg.adversarial_mode!r}")
        self._adversarial_step = 0
        if self.adversarial_mode in ("hard", "soft"):
            self._set_include_gpu_probs_tensor(self.large_worker)
            self._set_include_gpu_probs_tensor(self.small_worker)
            print(
                "ADISTILL_DUAL_ADVERSARIAL enabled "
                f"mode={self.adversarial_mode} "
                f"hard_candidate_top_k={dual_cfg.hard_candidate_top_k} "
                f"hard_candidate_top_p={dual_cfg.hard_candidate_top_p} "
                f"soft_student_weight={dual_cfg.soft_student_weight} "
                f"soft_temperature={dual_cfg.soft_temperature}"
            )
        from vllm.platforms import current_platform
        self.current_platform = current_platform

    # ------------------------------------------------
    # 生命周期
    # ------------------------------------------------

    def init_device(self):
        if self.large_actually_small:
            self.small_worker.init_device()
            self.large_worker.init_device()
            self.small_worker.load_model()
            self.large_worker.load_model()
        else:
            self.large_worker.init_device()
            self.small_worker.init_device()
            self.large_worker.load_model()
            self.small_worker.load_model()

    def _get_rank_in_group(self):
        global_tp = get_tp_group()
        return global_tp.rank_in_group

    def _log(self, msg, *args, **kwargs):
        logger.info(f"[Rank {self._get_rank_in_group()}] {msg}", *args, **kwargs)

    def load_model(self, *args, **kwargs):
        pass
    
    def load_small_model(self, weights):
        print('dual loading small model: {}'.format(weights[0][0]))
        if self.large_actually_small:
            self.small_worker.model_runner.model.load_weights(weights)
        else:
            self.small_worker._worker.model_runner.model.load_weights(weights)

    def _unwrap_worker(self, worker):
        return getattr(worker, "_worker", worker)

    def _set_include_gpu_probs_tensor(self, worker) -> None:
        worker = self._unwrap_worker(worker)
        if hasattr(worker, "model_runner") and hasattr(worker.model_runner,
                                                       "sampler"):
            worker.model_runner.sampler.include_gpu_probs_tensor = True

    def _build_sample_row_map(self, execute_model_req, groups):
        """Map Pythonized SequenceOutput samples back to sampler tensor rows.

        vLLM 的 sampler tensor 按 sequence group 展平；prompt 阶段 n>1 时
        多个 sample 共享同一个 parent row，decode 阶段每个 live sequence
        对应一个 row。这里只依赖 ExecuteModelRequest 中的 seq_data 顺序，
        避免改 scheduler/model runner。
        """
        seq_group_metadata_list = []
        if execute_model_req is not None:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        row_cursor = 0
        row_map = []
        for group_idx, group in enumerate(groups):
            metadata = (seq_group_metadata_list[group_idx]
                        if group_idx < len(seq_group_metadata_list) else None)
            seq_ids = []
            if metadata is not None:
                seq_ids = list(metadata.seq_data.keys())
            row_count = max(len(seq_ids), 1) if group.samples else 0

            seq_id_to_row = {
                seq_id: row_cursor + idx
                for idx, seq_id in enumerate(seq_ids)
            }
            for sample_idx, sample in enumerate(group.samples):
                row_idx = seq_id_to_row.get(
                    sample.parent_seq_id,
                    row_cursor + min(sample_idx, max(row_count - 1, 0)),
                )
                row_map.append((group_idx, sample, row_idx))
            row_cursor += row_count

        return row_map

    def _teacher_candidate_ids(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        top_k = int(self.dual_cfg.hard_candidate_top_k)
        vocab_size = teacher_probs.shape[-1]
        if top_k <= 0 or top_k > vocab_size:
            top_k = vocab_size

        candidate_probs, candidate_ids = torch.topk(teacher_probs,
                                                    k=top_k,
                                                    dim=-1)
        top_p = float(self.dual_cfg.hard_candidate_top_p)
        if 0.0 < top_p < 1.0:
            cumulative_probs = torch.cumsum(candidate_probs, dim=-1)
            keep = cumulative_probs <= top_p
            keep[1:] = keep[:-1].clone()
            keep[0] = True
            candidate_ids = candidate_ids[keep]

        return candidate_ids

    def _choose_hard_token(
        self,
        teacher_probs: torch.Tensor,
        student_logprobs: torch.Tensor,
    ) -> int:
        candidate_ids = self._teacher_candidate_ids(teacher_probs)
        candidate_student_logprobs = student_logprobs[candidate_ids]
        selected_idx = torch.argmin(candidate_student_logprobs)
        return int(candidate_ids[selected_idx].item())

    def _choose_soft_token(
        self,
        teacher_probs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        student_logprobs: torch.Tensor,
    ) -> int:
        candidate_ids = self._teacher_candidate_ids(teacher_probs)
        weight = float(self.dual_cfg.soft_student_weight)
        temperature = max(float(self.dual_cfg.soft_temperature), 1e-6)
        scores = teacher_logprobs[candidate_ids] - (
            weight * student_logprobs[candidate_ids])
        scores = torch.nan_to_num(scores,
                                  nan=-1e20,
                                  neginf=-1e20,
                                  posinf=1e20)
        scores = scores / temperature
        probs = torch.softmax(scores - scores.max(), dim=-1)
        selected_idx = torch.multinomial(probs, num_samples=1)
        return int(candidate_ids[selected_idx].item())

    def _make_logprob(
        self,
        logprobs: torch.Tensor,
        row_idx: int,
        token_id: int,
    ) -> Logprob:
        token_logprob = float(logprobs[row_idx, token_id].item())
        token_rank = int((logprobs[row_idx] > logprobs[row_idx, token_id]).
                         sum().item() + 1)
        return Logprob(
            logprob=token_logprob,
            rank=token_rank,
            decoded_token=None,
        )

    def _merge_adversarial_outputs(self, execute_model_req, out_l, out_s):
        teacher_sampler = out_l[0]
        student_sampler = out_s[0]
        teacher_logprobs = teacher_sampler.logprobs
        student_logprobs = student_sampler.logprobs
        teacher_probs = teacher_sampler.sampled_token_probs
        if teacher_probs is None and teacher_logprobs is not None:
            teacher_probs = torch.exp(teacher_logprobs)

        if teacher_logprobs is None or student_logprobs is None \
                or teacher_probs is None:
            raise RuntimeError(
                "hard/soft adversarial decoding requires sampler GPU "
                "probs/logprobs. Check include_gpu_probs_tensor setup.")
        if teacher_logprobs.shape != student_logprobs.shape:
            raise RuntimeError(
                "Teacher/Student sampler tensor shape mismatch: "
                f"{teacher_logprobs.shape} vs {student_logprobs.shape}. "
                "Token-level adversarial decoding currently requires aligned "
                "tokenizers and sampling rows.")

        out_l_groups = teacher_sampler.outputs
        row_map = self._build_sample_row_map(execute_model_req, out_l_groups)
        merged_outputs: List[CompletionSequenceGroupOutput] = []
        rows_by_group: Dict[int, List[Tuple[SequenceOutput, int]]] = {}
        for group_idx, sample, row_idx in row_map:
            rows_by_group.setdefault(group_idx, []).append((sample, row_idx))

        intervention_count = 0
        total_count = 0
        for group_idx, group_l in enumerate(out_l_groups):
            samples = []
            for sample_l, row_idx in rows_by_group.get(group_idx, []):
                if row_idx >= teacher_logprobs.shape[0]:
                    raise RuntimeError(
                        f"Sampler row index {row_idx} is out of range for "
                        f"logprobs shape {teacher_logprobs.shape}.")

                if self.adversarial_mode == "hard":
                    final_token = self._choose_hard_token(
                        teacher_probs[row_idx],
                        student_logprobs[row_idx],
                    )
                elif self.adversarial_mode == "soft":
                    final_token = self._choose_soft_token(
                        teacher_probs[row_idx],
                        teacher_logprobs[row_idx],
                        student_logprobs[row_idx],
                    )
                else:
                    raise RuntimeError(
                        f"Unknown adversarial mode: {self.adversarial_mode}")

                is_intervened = final_token != sample_l.output_token
                intervention_count += int(is_intervened)
                total_count += 1
                samples.append(
                    SequenceOutput(
                        parent_seq_id=sample_l.parent_seq_id,
                        output_token=final_token,
                        is_intervened=is_intervened,
                        logprobs={
                            final_token: self._make_logprob(
                                teacher_logprobs, row_idx, final_token)
                        },
                    ))

            merged_outputs.append(
                CompletionSequenceGroupOutput(samples=samples,
                                              prompt_logprobs=None))

        self._adversarial_step += total_count
        interval = max(int(self.dual_cfg.debug_log_interval), 1)
        if total_count and (self._adversarial_step == total_count
                            or self._adversarial_step % interval == 0):
            print(
                "ADISTILL_DUAL_ADVERSARIAL step "
                f"mode={self.adversarial_mode} "
                f"interventions={intervention_count}/{total_count}"
            )

        return [SamplerOutput(outputs=merged_outputs)]

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
        print('in dual\'s initialize_cache: num_gpu_blocks={}, num_cpu_blocks={}'.format(
            num_gpu_blocks, num_cpu_blocks))
        if self.large_actually_small:
            self.small_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
            self.large_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        else:
            self.large_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
            self.small_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def determine_num_available_blocks(self):
        actually_large_worker, actually_small_worker = \
            (self.small_worker, self.large_worker) if self.large_actually_small else (self.large_worker, self.small_worker)

        num_gpu_blocks, num_cpu_blocks = \
            actually_large_worker.determine_num_available_blocks()

        print("decided num blocks: gpu {}, cpu {}".format(
            num_gpu_blocks, num_cpu_blocks))

        large_block_bytes = actually_large_worker.get_cache_block_size_bytes()
        small_block_bytes = actually_small_worker.get_cache_block_size_bytes()
        print("large block bytes: {}, small block bytes: {}".format(
            large_block_bytes, small_block_bytes))

        # 平分显存，而不是平分 block 数
        new_num_gpu_blocks = int(
            num_gpu_blocks * large_block_bytes /
            (large_block_bytes + small_block_bytes)
        )
        print("revised num gpu blocks: {}, num cpu blocks: {}".format(
            new_num_gpu_blocks, num_cpu_blocks))

        return new_num_gpu_blocks, num_cpu_blocks

    # ------------------------------------------------
    # 核心执行逻辑
    # ------------------------------------------------

    @torch.inference_mode()
    def execute_model(self, execute_model_req):
        # if execute_model_req:
        #     for seq_group_metadata in execute_model_req.seq_group_metadata_list:
        #         self._log(seq_group_metadata.request_id)
        #         for seq_id, seq_data in seq_group_metadata.seq_data.items():
        #             self._log('  seq_id={}'.format(
        #                 seq_id,
        #             ))
        # self._log('DualModelWorker: execute_model called. Execute_model_req={}'.format(execute_model_req))

        # 1️⃣ 两个模型对同一 prefix forward
        out_l = self.large_worker.execute_model(execute_model_req)
        out_s = self.small_worker.execute_model(execute_model_req)
        # return out_l

        # self._log('out_l={}, out_s={}'.format(out_l, out_s))

        if self.large_actually_small and self.large_worker._is_dummy:
            return out_s

        if not self.large_actually_small and self.small_worker._is_dummy:
            return out_l

        if out_l is None and out_s is None:
            return None

        if out_l == [] and out_s == []:
            return []

        if self.adversarial_mode in ("hard", "soft"):
            return self._merge_adversarial_outputs(execute_model_req, out_l,
                                                   out_s)
        
        out_l = out_l[0].outputs
        out_s = out_s[0].outputs

        # print('DualModelWorker: out_l=', out_l)
        # print('DualModelWorker: out_s=', out_s)

        merged_outputs = []

        # 2️⃣ 对每个 sequence group 进行仲裁

        tokens_l = []
        tokens_s = []
        assert len(out_l) == len(out_s)
        for group_l, group_s in zip(out_l, out_s):
            # assert len(group_l.samples) == 1
            # assert len(group_s.samples) == 1
            # sample_l = group_l.samples[0]
            # sample_s = group_s.samples[0]
            assert len(group_l.samples) == len(group_s.samples)
            for sample_l, sample_s in zip(group_l.samples, group_s.samples):
                token_l = sample_l.output_token
                token_s = sample_s.output_token

                tokens_l.append(token_l)
                tokens_s.append(token_s)

        tokens_l = torch.tensor(tokens_l).cuda()
        tokens_s = torch.tensor(tokens_s).cuda()

        # 3️⃣ 仲裁得到最终 token
        mask, final_tokens = self.arbiter.choose(tokens_s, tokens_l)
        final_tokens = final_tokens.tolist()
        mask = mask.tolist()

        i = 0
        for group_l in out_l:
            # sample_l = group_l.samples[0]
            samples = []
            for sample_l in group_l.samples:
                # 4️⃣ 构造新的 SequenceOutput
                # if mask[i]:
                #     self._log('group {}, {}'.format(i, final_tokens[i]))
                merged_sample = SequenceOutput(
                    parent_seq_id=sample_l.parent_seq_id,
                    output_token=final_tokens[i],
                    is_intervened=mask[i],
                    logprobs={
                        final_tokens[i]: Logprob(
                            logprob=0.0,   # ✅ 可以是 dummy，vLLM 不强依赖
                            rank=None,
                            decoded_token=None,
                        )
                    }
                )
                samples.append(merged_sample)
                i += 1

            merged_group = CompletionSequenceGroupOutput(
                samples=samples,
                prompt_logprobs=None
            )

            merged_outputs.append(merged_group)

        # 5️⃣ 构造最终 SamplerOutput
        merged_sampler_output = SamplerOutput(
            outputs=merged_outputs,
        )

        return [merged_sampler_output]

    # ------------------------------------------------
    # 属性代理
    # ------------------------------------------------

    @property
    def rank(self):
        return self.large_worker.rank

    @property
    def device(self):
        return self.large_worker.device

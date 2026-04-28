from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any, Tuple

import torch
import torch.nn as nn

from vllm.worker.worker import Worker
from vllm.worker.worker_base import WorkerBase, DelegateWorkerBase
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import SequenceOutput, Logprob
from vllm.distributed.parallel_state import get_tp_group, init_model_parallel_group, patch_tensor_parallel_group
from vllm.logger import init_logger

logger = init_logger(__name__)


# ------------------------
# 1) Smaller-TP wrapper (Copied from arbiter_model.py for standalone usage)
# ------------------------
class SmallerTpWorkerWrapper(WorkerBase):
    def __init__(self, worker: DelegateWorkerBase, draft_ranks: List[int]):
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
            return
        local_rank = global_tp.local_rank
        backend = torch.distributed.get_backend(global_tp.device_group)
        self._tp_group = init_model_parallel_group([self._draft_ranks], local_rank, backend)
        with self._patch_tp():
            self._worker.init_device()

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
    def execute_model(self, execute_model_req: Optional[ExecuteModelRequest] = None) -> List[SamplerOutput]:
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


# ------------------------
# 2) Config & Network
# ------------------------
@dataclass
class KeywordArbiterWorkerConfig:
    """Configuration for KeywordArbiterWorker."""
    reasoner_engine_args: Any
    prefix_engine_args: Any

    reasoner_tp: int
    prefix_tp: int

    # Trigger & Candidates
    # Set of token IDs that trigger the arbiter check
    trigger_token_ids: Set[int] 
    # List of candidate interventions (each is a list of token ids)
    # Index 0 is reserved for "No Intervention" if arbiter decides so? 
    # Or arbiter outputs logits of size len(candidates) + 1? 
    # Let's assume prediction 0 means "keep reasoner token", 1..N means "use candidate i-1"
    intervention_candidates: List[List[int]]

    # Arbiter parameters
    arbiter_state_dict: Optional[Dict[str, torch.Tensor]] = None
    
    # Embedding dim for the reasoner token input to Arbiter
    token_embed_dim: int = 64
    num_hidden_layers: int = 1
    reasoner_hidden_dim: int = 1536


class KeywordArbiter(nn.Module):
    """
    Arbiter network that takes:
    1. Reasoner's base model hidden state
    2. Prefix encoder's hidden state
    
    Outputs: Logits over [Keep, Candidate_1, Candidate_2, ...]
    """
    def __init__(self, prefix_dim: int, reasoner_dim: int, num_candidates: int, num_hidden_layers: int = 1, dtype=torch.bfloat16):
        super().__init__()
        
        in_dim = prefix_dim + reasoner_dim
        
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(in_dim, num_candidates + 1)) # +1 for "Keep Reasoner Token"
        self.net = nn.Sequential(*layers)

        self.to(dtype=dtype)

    def forward(self, reasoner_hidden: torch.Tensor, prefix_hidden: torch.Tensor) -> torch.Tensor:
        # reasoner_hidden: (batch_size, reasoner_dim)
        # prefix_hidden: (batch_size, prefix_dim)
        
        x = torch.cat([reasoner_hidden, prefix_hidden], dim=-1)
        logits = self.net(x) # (B, num_candidates + 1)
        return logits

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
    

# ------------------------
# 3) Keyword Arbiter Worker
# ------------------------
class KeywordArbiterWorker(WorkerBase):
    def __init__(self, *args, **kwargs):
        vllm_config = kwargs["vllm_config"]
        # Use ArbiterModelConfig from vllm_config
        cfg = vllm_config.arbiter_model_config
        if cfg is None:
            raise ValueError("Missing vllm_config.arbiter_model_config")
        self.cfg = cfg
        
        if self.cfg.prefix_engine_args is None:
             raise ValueError("Missing prefix_engine_args in arbiter_model_config")

        prefix_vllm_config = self.cfg.prefix_engine_args.create_engine_config()
        
        # Store TPs
        self.reasoner_tp = vllm_config.parallel_config.tensor_parallel_size
        self.prefix_tp = prefix_vllm_config.parallel_config.tensor_parallel_size
        self.max_tp_size = max(self.reasoner_tp, self.prefix_tp)

        # Reasoner Worker
        # We assume *args, **kwargs are correct for reasoner
        reasoner_worker = Worker(*args, **kwargs)
        if self.reasoner_tp != self.max_tp_size:
            reasoner_worker = SmallerTpWorkerWrapper(
                reasoner_worker,
                draft_ranks=list(range(self.reasoner_tp))
            )
        self.reasoner_worker = reasoner_worker
        
        # Prefix Worker
        # We need to filter kwargs to remove "vllm_config" and pass the prefix_vllm_config
        prefix_kwargs = {k: v for k, v in kwargs.items() if k != "vllm_config"}
        prefix_worker = Worker(*args, vllm_config=prefix_vllm_config, **prefix_kwargs)
        if self.prefix_tp != self.max_tp_size:
            prefix_worker = SmallerTpWorkerWrapper(
                prefix_worker,
                draft_ranks=list(range(self.prefix_tp))
            )
        self.prefix_worker = prefix_worker

        self._draft_ranks: Optional[List[int]] = None
        self.arbiter: Optional[KeywordArbiter] = None
        self._prefix_dim: Optional[int] = None
        self._reasoner_dim: Optional[int] = None
        
        if self.cfg.trigger_token_ids is None:
             raise ValueError("trigger_token_ids must be set in arbiter_model_config")
             
        # Maps trigger_token_id -> index for embedding
        self.trigger_id_to_index: Dict[int, int] = {
            tid: i for i, tid in enumerate(sorted(list(self.cfg.trigger_token_ids)))
        }
        
        # State Machine: seq_id -> List[int] (remaining tokens to force output)
        self.intervention_states: Dict[str, List[int]] = {}

    def init_device(self) -> None:
        self.reasoner_worker.init_device()
        self.prefix_worker.init_device()

        self.reasoner_worker.load_model()
        self.prefix_worker.load_model()

        # Infer hidden size for prefix worker
        def _get_hidden_size(w) -> int:
            model = getattr(getattr(w, "_worker", w), "model_runner", None)
            if model is None:
                mr = getattr(getattr(w, "_worker", w), "model_runner", None)
                model = mr
            m = getattr(getattr(getattr(w, "_worker", w), "model_runner", None), "model", None)
            cfg = getattr(m, "config", None)
            hs = getattr(cfg, "hidden_size", None)
            if hs is None:
                hs = getattr(cfg, "n_embd", None)
            if hs is None:
                raise RuntimeError("Cannot infer hidden_size from worker model.")
            return int(hs)

        Hp = _get_hidden_size(self.prefix_worker)
        self._prefix_dim = Hp

        Hr = _get_hidden_size(self.reasoner_worker)
        self._reasoner_dim = Hr
        
        # Initialize Arbiter
        num_triggers = len(self.trigger_id_to_index)
        num_candidates = len(self.cfg.intervention_candidates)
        
        self.arbiter = KeywordArbiter(
            prefix_dim=Hp, 
            reasoner_dim=Hr,
            num_candidates=num_candidates,
            num_hidden_layers=getattr(self.cfg, "num_hidden_layers", 1),
        ).to("cuda").eval()
        
        if self.cfg.arbiter_state_dict is not None:
            self.arbiter.load_state_dict(self.cfg.arbiter_state_dict, strict=False)

        logger.info("KeywordArbiterWorker initialized. Hp=%d Hr=%d num_triggers=%d num_candidates=%d", Hp, Hr, num_triggers, num_candidates)

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
            elif name.startswith("prefix."):
                sub_name = name[len("prefix.") :]
                self.prefix_worker.load_weights([(sub_name, weight)])
                logger.info("Loaded prefix weight: %s", sub_name)
    
    def determine_num_available_blocks(self):
        nr, cr = self.reasoner_worker.determine_num_available_blocks()
        # np, cp = self.prefix_worker.determine_num_available_blocks()
        print("decided num blocks: gpu {}, cpu {}".format(
            nr, cr))
        reasoner_block_bytes = self.reasoner_worker.get_cache_block_size_bytes()
        prefix_encoder_block_bytes = self.prefix_worker.get_cache_block_size_bytes()

        new_nr = int(nr * reasoner_block_bytes / (reasoner_block_bytes + prefix_encoder_block_bytes))
        print("revised num gpu blocks: {}, num cpu blocks: {}".format(
            new_nr, cr))
        return new_nr, cr

    def get_cache_block_size_bytes(self) -> int:
        br = self.reasoner_worker.get_cache_block_size_bytes()
        bp = self.prefix_worker.get_cache_block_size_bytes()
        return max(br, bp)

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks) -> None:
        self.reasoner_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self.prefix_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    @torch.inference_mode()
    def execute_model(self, execute_model_req: Optional[ExecuteModelRequest] = None) -> List[SamplerOutput]:
        if execute_model_req is None:
            return []

        # 1. Run both workers (synchronous execution to maintain cache state)
        out_r_list = self.reasoner_worker.execute_model(execute_model_req)
        # Assuming Prefix Encoder needs to run every step:
        out_p_list = self.prefix_worker.execute_model(execute_model_req)

        # Handle dummy ranks
        if not out_r_list or not out_p_list:
            return []

        out_r = out_r_list[0]
        out_p = out_p_list[0]

        # Check required fields
        # if out_r.sampled_token_ids is None:
        #     raise RuntimeError("SamplerOutput.sampled_token_ids is None.")

        # 2. Prepare for Logic
        # flatten r_tokens just for reference if needed, but we mostly work with sample objects
        # p_h: (Total_Active_Seqs, Hidden_Dim)
        p_h = out_p.hidden_states 
        r_h = out_r.hidden_states

        # We need to align flat sequence index with p_h
        current_flat_idx = 0
        
        # For batch arbitration
        arb_indices = []        # Index in p_h
        arb_samples = []        # Tuples of (SampleObject, SeqID)
        
        # 3. Iterate over outputs to identify interventions vs arbitrations
        # We assume out_r.outputs aligns with execute_model_req.seq_group_metadata_list
        if len(out_r.outputs) != len(execute_model_req.seq_group_metadata_list):
            logger.warning("Output groups length mismatch with request groups. Skipping arbitration.")
            return [out_r]

        for g_out, g_meta in zip(out_r.outputs, execute_model_req.seq_group_metadata_list):
            known_seq_ids = list(g_meta.seq_data.keys())
            
            for s_idx, sample in enumerate(g_out.samples):
                # Initialize arbiter prediction to -1 explicitly
                if not hasattr(sample, 'arbiter_prediction'):
                    sample.arbiter_prediction = -1
                    sample.is_intervened = False

                # Resolve Seq ID: vLLM SequenceOutput has parent_seq_id
                if hasattr(sample, 'parent_seq_id') and sample.parent_seq_id is not None:
                    seq_id = sample.parent_seq_id
                else:
                    # Fallback for safe compatibility: assume 1-to-1 or ordered mapping
                    if s_idx < len(known_seq_ids):
                        seq_id = known_seq_ids[s_idx]
                    else:
                        seq_id = known_seq_ids[0]
                
                # Check 1: State Machine (Priority)
                if seq_id in self.intervention_states:
                    remaining = self.intervention_states[seq_id]
                    next_token = remaining.pop(0)
                    
                    # Override Output
                    sample.output_token = next_token
                    # Logprob -inf indicates it was forced/intervened
                    sample.logprobs = {next_token: Logprob(float("-inf"), rank=1)}
                    
                    if not remaining:
                        del self.intervention_states[seq_id]
                
                # Check 2: Trigger Condition
                elif sample.output_token in self.trigger_id_to_index:
                    # Prepare for Arbitration
                    # Only if we have hidden states
                    if p_h is not None and current_flat_idx < p_h.size(0) and r_h is not None and current_flat_idx < r_h.size(0):
                        arb_indices.append(current_flat_idx)
                        arb_samples.append((sample, seq_id))
                
                # Increment flat index for alignment with p_h
                current_flat_idx += 1

        # 4. Batch Arbitration
        if arb_indices:
            device = self.device
            # Gather hidden states
            prefix_input = p_h[arb_indices]
            reasoner_input = r_h[arb_indices]
            
            # Forward Arbiter
            logits = self.arbiter(reasoner_input, prefix_input)
            
            # Simple greedy choice for arbiter
            # logits: (Batch, Candidates + 1)
            # 0 -> Keep, >0 -> Candidate
            decisions = torch.argmax(logits, dim=-1).cpu().tolist()
            
            for k, decision_idx in enumerate(decisions):
                if decision_idx > 0:
                    cand_idx = decision_idx - 1
                    sample, seq_id = arb_samples[k]
                    
                    # Start Intervention
                    cand_tokens = list(self.cfg.intervention_candidates[cand_idx])
                    
                    if cand_tokens:
                        # Apply first token immediately
                        first_token = cand_tokens.pop(0)
                        sample.output_token = first_token
                        sample.logprobs = {first_token: Logprob(float("-inf"), rank=1)}
                        
                        # Save remaining to state machine
                        if cand_tokens:
                            self.intervention_states[seq_id] = cand_tokens
                
                # Finally, record the arbiter output
                sample = arb_samples[k][0]
                sample.arbiter_prediction = decision_idx
                sample.is_intervened = decision_idx > 0

        return [out_r]

    def _get_prefix_embeddings(self, token_ids: List[int]) -> Optional[torch.Tensor]:
        """
        Try to extract embeddings for the given token_ids from the prefix model.
        Returns tensor of shape (len(token_ids), hidden_size) or None if not possible.
        """
        # Unwrap worker safely
        w = self.prefix_worker
        if isinstance(w, SmallerTpWorkerWrapper):
             w = w._worker
        
        # Check if attribute exists
        if not hasattr(w, "model_runner"):
             return None
             
        model_runner = w.model_runner
        vllm_model = getattr(model_runner, "model", None)
        
        # Try to find embedding
        embed = None
        # Common locations for vLLM models
        locations = [
            lambda m: m.embed_tokens,
            lambda m: m.model.embed_tokens,
            lambda m: m.transformer.wte,
        ]
        
        for loc in locations:
            try:
                embed = loc(vllm_model)
                if embed is not None:
                    break
            except AttributeError:
                pass
        
        if embed is None:
             logger.warning("Could not find embedding layer in prefix model.")
             return None
             
        # Check if parallel (VocabParallelEmbedding)
        is_parallel = "VocabParallel" in embed.__class__.__name__ or hasattr(embed, "vocab_start_index")
        
        # If parallel, we only support retrieval if self.prefix_tp == 1 AND embed is full vocab?
        # Or if embed holds the required tokens?
        # For simplicity, if prefix_tp > 1, we skip.
        if self.prefix_tp > 1:
            logger.warning(f"Prefix model uses TP={self.prefix_tp}. Skipping embedding init.")
            return None
        
        # Handle regular Embedding or VocabParallelEmbedding (which mimics Embedding but has sharded weight)
        # If prefix_tp == 1, VocabParallelEmbedding holds the full vocab.
        
        if not hasattr(embed, "weight"):
             return None

        weight = embed.weight # (V, D)
        device = weight.device
        
        # Prepare indices
        indices = torch.tensor(token_ids, device=device, dtype=torch.long)
        
        # Check bounds
        if indices.max() >= weight.shape[0]:
             # It means vocab size mismatch or token id issue
             logger.warning(f"Token ID {indices.max().item()} out of bounds for prefix model embedding size {weight.shape[0]}.")
             return None
             
        try:
            return torch.nn.functional.embedding(indices, weight).detach().to(dtype=weight.dtype)
        except Exception as e:
            logger.warning(f"Failed to extract embeddings: {e}")
            return None

    @property
    def rank(self):
        return self.reasoner_worker.rank

    @property
    def device(self):
        return self.reasoner_worker.device

    @property
    def vocab_size(self):
        return self.reasoner_worker.vocab_size

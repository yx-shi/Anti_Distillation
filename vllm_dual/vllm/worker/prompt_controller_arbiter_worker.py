import torch
from typing import List, Tuple, Dict, Any, Optional
import math

from vllm.worker.worker_base import WorkerBase
from vllm.worker.arbiter_worker import HiddenStateArbiterWorker
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.sequence import SequenceData
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.config import VllmConfig, PromptControllerArbiterModelConfig

import copy

logger = init_logger(__name__)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/disk1/public_checkpoint/DeepSeek-R1-Distill-Qwen-1.5B_lxj")

class PromptControllerArbiterWorker(HiddenStateArbiterWorker):
    """
    Inherits from HiddenStateArbiterWorker but manages a private KV cache reserve
    for the Controller model to support prompt wrapping.
    
    The Controller sees: [A] [R'] [C]
    The Reasoner sees:   [R]
    
    where R = Problem + Trace.
    R' is structurally identical to R (same length), but with special tokens masked/replaced
    to be compatible with the Controller's embedding space if needed (or identical).
    
    CRITICAL: 
    1. A is padded to block_size boundary.
    2. R' reuses the EXACT same block IDs as R from the Scheduler.
    3. C is appended using private blocks.
    """
    
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        vllm_config: VllmConfig = kwargs["vllm_config"]
        cfg: PromptControllerArbiterModelConfig = vllm_config.arbiter_model_config
        
        # self.user_token_id = cfg.user_token_id
        # self.assistant_token_id = cfg.assistant_token_id
        self.special_token_ids = getattr(cfg, 'special_token_ids', None) or []
        
        # --- Alignment Strategy ---
        # We ensure A is padded to block_size so that R' starts at a fresh block boundary
        # in the Controller's view, perfectly matching the Scheduler's allocation for R.
        
        block_size = 16 
        if hasattr(self.controller_cfg, 'cache_config') and self.controller_cfg.cache_config:
            block_size = self.controller_cfg.cache_config.block_size
            
        self.user_token_id = cfg.user_token_id
        self.assistant_token_id = cfg.assistant_token_id
        self.pad_token_id = cfg.pad_token_id
        self.warmup_steps = getattr(cfg, 'warmup_steps', 0)
        self.c_lookback_n = getattr(cfg, 'c_lookback_n', 0)
        self.controller_banned_token_ids = cfg.controller_banned_token_ids or []
        self.controller_banned_token_tensor = torch.tensor(self.controller_banned_token_ids, dtype=torch.long) if self.controller_banned_token_ids else None

        # Replacement spans for special token masking in R -> R'
        self.user_replacement_ids = list(getattr(cfg, 'user_replacement_ids', []))
        self.assistant_replacement_ids = list(getattr(cfg, 'assistant_replacement_ids', []))
             
        def _align_to_block(token_list):
            remainder = len(token_list) % block_size
            if remainder != 0:
                padding_needed = block_size - remainder
                return [self.pad_token_id] * padding_needed + token_list
            return token_list

        self.prompt_prefix_ids = _align_to_block(list(cfg.prompt_prefix_ids))
        self.prompt_suffix_ids = cfg.prompt_suffix_ids # C does not need alignment
        
        # We no longer insert B (middle_ids). 
        # Ideally, cfg.prompt_middle_ids should be empty or handled inside R' transformation if needed.
        if cfg.prompt_middle_ids:
            logger.warning("PromptControllerArbiterWorker: prompt_middle_ids (B) is ignored in A+R'+C mode.")

        self.max_num_seqs = cfg.max_num_seqs

        
        # Will be calculated during determine_num_available_blocks
        self.num_reserved_blocks = 0
        self.private_block_start_index = -1
        self.private_block_end_index = -1
        
        # Scratchpad management
        self.available_private_blocks: List[int] = []
        # Helper to protect internal private block allocation
        # seq_id -> {'blocks': [persistent_blocks], 'len_a': int, 'len_b': int}
        self.seq_private_allocs = {}
        self.blocks_to_free_after_step = []
        self.shared_a_blocks = []


    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Overridden to under-report GPU blocks to the scheduler.
        Allocates blocks based on block size (bytes) weighted average across workers,
        while reserving private blocks for the Controller.
        """
        # 1. Get raw available blocks from all workers
        r_gpu, r_cpu = self.reasoner_worker.determine_num_available_blocks()
        c_gpu, c_cpu = self.controller_worker.determine_num_available_blocks()
        p_gpu, p_cpu = self.prefix_worker.determine_num_available_blocks()
        
        # 2. Calculate Reservation Size (Controller Only)
        # Block size for controller
        if hasattr(self.controller_cfg, 'cache_config') and self.controller_cfg.cache_config:
            block_size = self.controller_cfg.cache_config.block_size
        else:
            # Fallback attempts
            try:
                block_size = self.controller_worker.cache_config.block_size
            except:
                block_size = 16 # Default vLLM block size
        
        # Tokens needed per sequence for wrappers
        # In A+R'+C mode, we need A and C to be covered by private blocks (or partially covered by R's unused slots)
        # But wait, A is padded to full blocks, so A needs full private blocks.
        # R' reuses Scheduler blocks.
        # C needs private blocks.
        
        # New Strategy: A is SHARED. C is private per seq.
        a_len = len(self.prompt_prefix_ids)
        c_len = len(self.prompt_suffix_ids) + self.c_lookback_n
        
        blocks_a = a_len // block_size
        # Conservative estimate for C blocks needed per seq (including potential misalignment with R)
        blocks_c_per_seq = (c_len + block_size - 1) // block_size + 1
        
        # Total reserved blocks
        self.num_reserved_blocks = blocks_a + self.max_num_seqs * blocks_c_per_seq
        
        logger.info(
            f"PromptControllerArbiterWorker: Reserving {self.num_reserved_blocks} blocks "
            f"for controller (MacSeq={self.max_num_seqs}, SharedA={blocks_a} blks, PerSeqC={blocks_c_per_seq} blks)."
        )
        
        # Check Controller OOM
        if c_gpu < self.num_reserved_blocks:
            raise RuntimeError(
                f"OOM: Controller model has {c_gpu} blocks, but needs {self.num_reserved_blocks} "
                "reserved for prompt wrapping. Reduce max_num_seqs or prompt length."
            )

        print('PromptControllerArbiterWorker available blocks:')
        print(f'  reasoner: gpu={r_gpu} cpu={r_cpu}')
        print(f'  controller: gpu={c_gpu} cpu={c_cpu} (reserved={self.num_reserved_blocks})')
        print(f'  prefix: gpu={p_gpu} cpu={p_cpu}')

        # 3. Weighted Allocation (Arbiter Style)
        # Calculate bytes per block for each worker
        r_block_bytes = self.reasoner_worker.get_cache_block_size_bytes()
        c_block_bytes = self.controller_worker.get_cache_block_size_bytes()
        p_block_bytes = self.prefix_worker.get_cache_block_size_bytes()
        
        total_block_bytes = r_block_bytes + c_block_bytes + p_block_bytes
        
        # Calculate Total Available Bytes (using Reasoner as anchor, consistent with ArbiterWorker)
        # r_gpu * r_block_bytes approximates the total shared GPU memory available for KV cache
        total_memory_bytes = r_gpu * r_block_bytes
        
        # Subtract Reserved Bytes for Controller Private Use
        reserved_bytes = self.num_reserved_blocks * c_block_bytes
        
        available_memory_bytes = total_memory_bytes - reserved_bytes
        
        if available_memory_bytes < 0:
            raise RuntimeError(f"OOM: Reserved bytes {reserved_bytes} > Total Memory {total_memory_bytes}")
        
        # Number of "Common" blocks (Public blocks) that fit in the remaining memory
        # Each common block consists of 1 Reasoner block + 1 Controller block + 1 Prefix block
        num_common_blocks = int(available_memory_bytes / total_block_bytes)
        
        return (num_common_blocks, r_cpu)


    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
        """
        Interprets num_gpu_blocks as the "Public" blocks allocated by Scheduler.
        We force Controller to allocate Public + Private blocks.
        """
        logger.info(f"PromptControllerArbiterWorker.initialize_cache called with {num_gpu_blocks=}")
        
        # 1. Initialize Reasoner & Prefix normally
        self.reasoner_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self.prefix_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        
        # 2. Adjust Controller Initialization to include reserved private blocks
        real_c_gpu_blocks = num_gpu_blocks + self.num_reserved_blocks
        logger.info(f"  -> Controller cache: request {real_c_gpu_blocks} blocks (Reserve {self.num_reserved_blocks})")
        
        self.controller_worker.initialize_cache(real_c_gpu_blocks, num_cpu_blocks)
        
        # 3. Record Private Block Range
        self.private_block_start_index = num_gpu_blocks
        self.private_block_end_index = real_c_gpu_blocks
        
        # 4. Initialize Scratchpad / Free List management for private blocks
        # We have range [num_gpu_blocks, real_c_gpu_blocks)
        self.available_private_blocks = list(range(self.private_block_start_index, self.private_block_end_index))
        logger.info(f"  -> Private blocks available: {len(self.available_private_blocks)}")

        # Allocate shared A blocks
        block_size = self.controller_worker.cache_config.block_size
        num_a_blocks = len(self.prompt_prefix_ids) // block_size
        if num_a_blocks > 0:
            self.shared_a_blocks = self._allocate_private_blocks(num_a_blocks)
            logger.info(f"  -> Allocated {num_a_blocks} shared blocks for A: {self.shared_a_blocks}")
        else:
            self.shared_a_blocks = []

    def _allocate_private_blocks(self, num_blocks: int) -> List[int]:
        if len(self.available_private_blocks) < num_blocks:
            # Try to reclaim from finished sequences? 
            # For now simpler to raise error
            raise RuntimeError(f"OOM: Run out of private blocks! Need {num_blocks}, have {len(self.available_private_blocks)}")
        allocated = self.available_private_blocks[:num_blocks]
        self.available_private_blocks = self.available_private_blocks[num_blocks:]
        return allocated

    def _free_private_blocks(self, blocks: List[int]):
        self.available_private_blocks.extend(blocks)
    
    def _extract_problem_and_trace(self, input_ids: List[int]) -> Tuple[List[int], List[int], int]:
        # Find indices of user and assistant tokens
        try:
            # Check exactly once
            if input_ids.count(self.user_token_id) != 1:
                raise ValueError(f"User token {self.user_token_id} must appear exactly once.")
            if input_ids.count(self.assistant_token_id) != 1:
                raise ValueError(f"Assistant token {self.assistant_token_id} must appear exactly once.")
                
            user_idx = input_ids.index(self.user_token_id)
            assistant_idx = input_ids.index(self.assistant_token_id)
            
            if user_idx >= assistant_idx:
                raise ValueError(f"User token appearing at {user_idx} must appear before Assistant token at {assistant_idx}.")
            
            # Extract content between them
            problem = input_ids[user_idx + 1 : assistant_idx]
            trace = [] 
            return problem, trace, user_idx + 1
            
        except ValueError as e:
            # Log and re-raise or handle? 
            # Re-raising ensures we catch malformed inputs early.
            raise ValueError(f"Failed to extract problem/trace: {e}") from e

    def _execute_controller(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        runner = self.controller_worker.model_runner
        block_size = self.controller_worker.cache_config.block_size
        
        # NOTE: We do NOT rely on runner.prepare_model_input(execute_model_req) here.
        # Calling prepare_model_input on the original request might fail or be truncated 
        # if the Controller's max_num_batched_tokens is smaller than the Reasoner's.
        # Instead, we derive necessary info (block IDs) directly from block_tables.
        
        new_meta_list = []
        
        from vllm.sequence import SequenceData
        import copy

        current_step_seq_ids = set()

        for seq_meta in execute_model_req.seq_group_metadata_list:
            new_seq_data_dict = {}
            new_block_tables_dict = {}
            
            # Register alive sequences to prevent premature freeing
            for seq_id, seq_data in seq_meta.seq_data.items():
                current_step_seq_ids.add(seq_id)

            # NOTE: We MUST execute the controller even if do_sample=False (Chunked Prefill).
            # We need to update the KV cache for the processed tokens.
            # However, we should align do_sample status so we don't produce extra outputs.

            for seq_id, seq_data in seq_meta.seq_data.items():
                current_tokens = seq_data.get_token_ids()
                num_computed = seq_data.get_num_computed_tokens()
                
                # In A + R' + C mode:
                # - R is the Reasoner sequence (Problem + Trace) represented by current_tokens
                # - R' is R with special tokens replaced by pad, same length
                # - A (prompt_prefix_ids) is block-aligned so R' starts at a new block boundary
                # Construct R'
                r_prime = self._generate_masked_trace(current_tokens)

                # Full Controller tokens: A + R' (+ C transient)
                c_dynamic_tokens = []
                if self.c_lookback_n > 0 and (decode_len := seq_data.get_output_len()) > 0:
                    c_dynamic_tokens = r_prime[-min(self.c_lookback_n, decode_len):]

                full_persistent_tokens = self.prompt_prefix_ids + r_prime
                full_tokens = full_persistent_tokens + self.prompt_suffix_ids + c_dynamic_tokens

                block_size = self.controller_worker.cache_config.block_size
                total_persistent_len = len(full_persistent_tokens)
                persistent_blocks_needed = (total_persistent_len + block_size - 1) // block_size

                sched_blocks = seq_meta.block_tables[seq_id]

                # Use shared A blocks
                if seq_id not in self.seq_private_allocs:
                    self.seq_private_allocs[seq_id] = {}
                
                a_blocks = self.shared_a_blocks

                # Pool = A_blocks (private) + Scheduler blocks (for R) // block_sizes are consistent
                persistent_pool = list(a_blocks) + list(sched_blocks)

                # If Scheduler hasn't provided enough blocks for R, allocate extra private blocks
                needed = persistent_blocks_needed - len(persistent_pool)
                if needed > 0:
                    raise ValueError(f"Scheduler under-allocated blocks for R. This should not happen if A is properly block-aligned. Needed extra private blocks: {needed}")
                    padding_blocks = self._allocate_private_blocks(needed)
                    persistent_pool.extend(padding_blocks)

                # Save pool state
                self.seq_private_allocs[seq_id]['pool'] = persistent_pool

                # 2. Allocation for Transient (C)
                total_len_with_c = len(full_tokens)
                total_blocks_needed = (total_len_with_c + block_size - 1) // block_size
                needed_c = total_blocks_needed - len(persistent_pool)
                if needed_c > 0:
                    transient_blocks = self._allocate_private_blocks(needed_c)
                    full_pool = persistent_pool + transient_blocks
                    self.blocks_to_free_after_step.append(transient_blocks)
                else:
                    full_pool = persistent_pool

                # 3. Compute `num_computed_tokens` for Chunked Prefill
                if seq_meta.is_prompt:
                    controller_computed = 0
                else:
                    # Already computed = len(A) + num_computed (num_computed refers to R tokens computed by Reasoner)
                    controller_computed = len(self.prompt_prefix_ids) + num_computed
                    if controller_computed < 0:
                        controller_computed = 0

                # Construct SequenceData
                new_sd = SequenceData.from_seqs(full_tokens)
                new_sd.update_num_computed_tokens(controller_computed)

                new_seq_data_dict[seq_id] = new_sd
                new_block_tables_dict[seq_id] = full_pool

            # Construct new metadata
            new_meta = copy.copy(seq_meta)
            # FORCE Prompt Mode (Chunked Prefill) because we are appending [C] (length > 1 usually)
            new_meta.is_prompt = True 
            new_meta.seq_data = new_seq_data_dict
            new_meta.block_tables = new_block_tables_dict
            
            # Ensure token_chunk_size covers the full processing length for the Controller
            if hasattr(new_meta, "token_chunk_size") and new_meta.token_chunk_size is not None:
                max_len = max(sd.get_len() for sd in new_seq_data_dict.values())
                new_meta.token_chunk_size = max_len

            # Fixes for Controller Incompatibility
            new_meta.computed_block_nums = []
            new_meta.do_sample = seq_meta.do_sample
            new_meta.lora_request = None
            new_meta.multi_modal_data = None
            new_meta.multi_modal_placeholders = None

            new_meta_list.append(new_meta)

        # Execute
        if not new_meta_list: 
            # If no work to do, we still MUST check for dead sequences
            pass
        else:
            new_model_input = runner.prepare_model_input(new_meta_list)
            with torch.no_grad():
                output = runner.execute_model(new_model_input, kv_caches=self.controller_worker.gpu_cache)
        
        # Cleanup Transient Blocks
        for blocks in self.blocks_to_free_after_step:
            self._free_private_blocks(blocks)
        self.blocks_to_free_after_step = []

        # Cleanup Dead Sequences
        # Any sequence allocated but not present in current request is considered finished/preempted
        all_allocated_ids = list(self.seq_private_allocs.keys())
        for seq_id in all_allocated_ids:
            if seq_id not in current_step_seq_ids:
                self._free_seq_resources(seq_id)

        if not new_meta_list: 
            return []

        # Slice output Hidden States to match Reasoner shape (Batch, Hidden)
        # Since Controller runs in Chunked Prefill mode, it returns hidden states for all 
        # computed tokens. We extract only the last token for each sequence.
        # if output and output[0].hidden_states is not None:
        #     hs = output[0].hidden_states
        #     last_indices = []
        #     current_idx = 0
        #     for new_meta in new_meta_list:
        #         for seq_id, seq_data in new_meta.seq_data.items():
        #             # Calculate how many tokens were actually processed in this forward pass
        #             total_len = seq_data.get_len()
        #             computed_len = seq_data.get_num_computed_tokens()
        #             processed_len = total_len - computed_len
                    
        #             # The token of interest is the last one in this processed chunk
        #             if processed_len > 0:
        #                 last_indices.append(current_idx + processed_len - 1)
        #                 current_idx += processed_len
            
        #     if len(last_indices) > 0 and hs.size(0) == current_idx:
        #         output[0].hidden_states = hs[last_indices]
        #     else:
        #         # Fallback/Error path if shapes don't match expectation
        #         pass

        return output

    def execute_model(self, execute_model_req) -> List[SamplerOutput]:
        # 1. Execute Reasoner & Prefix normally
        out_r = self.reasoner_worker.execute_model(execute_model_req)

        out_p = self.prefix_worker.execute_model(execute_model_req)

        # 2. Handle Controller
        # Use unified mixed-batch handler
        out_c = self._execute_controller(execute_model_req)


        # 3. Merge Logic (Copied/Adapted from HiddenStateArbiterWorker.execute_model)
        # Only if we have valid outputs
        if not out_c or not out_r:
            return self._merge_outputs_dummy(out_r, out_c) # Fallback

        sr = out_r[0]
        sc = out_c[0]
        sp = out_p[0]

        r_h = sr.hidden_states
        c_h = sc.hidden_states
        p_h = sp.hidden_states

        tokens_r = [s.output_token for g in sr.outputs for s in g.samples]
        tokens_c = [s.output_token for g in sc.outputs for s in g.samples]

        # for seq_meta in execute_model_req.seq_group_metadata_list:
        #     if 0 in seq_meta.seq_data:
        #         print('input tokens: {}'.format(tokenizer.decode(seq_meta.seq_data[0].output_token_ids[-20:])))
        #         token_idx = 0
        #         for i, sample in enumerate(sr.outputs):
        #             samples = sample.samples
        #             if samples and samples[0].parent_seq_id == 0:
        #                 token_idx = i
        #                 break
        #         print('reasoner output tokens: {}'.format(tokenizer.decode(tokens_r[token_idx]).replace('\n', '\\n')))
        #         print('controller output tokens: {}'.format(tokenizer.decode(tokens_c[token_idx]).replace('\n', '\\n')))
        #         break

        device = r_h.device
        tokens_r_t = torch.tensor(tokens_r, device=device, dtype=torch.long)
        tokens_c_t = torch.tensor(tokens_c, device=device, dtype=torch.long)

        gate_in = torch.cat([r_h, c_h, p_h], dim=-1)
        if gate_in.dim() != 2: gate_in = gate_in.view(gate_in.size(0), -1)

        gate_logits = self.arbiter(gate_in) / max(float(self.cfg.gate_temperature), 1e-6)
        gate = torch.sigmoid(gate_logits)
        choice = torch.rand_like(gate) < gate

        # Apply warmup steps logic: Force Reasoner (False) if generated tokens < warmup_steps
        if self.warmup_steps > 0:
            warmup_mask = []
            for i, group_r in enumerate(sr.outputs):
                seq_meta = execute_model_req.seq_group_metadata_list[i]
                for sample in group_r.samples:
                    seq_id = sample.parent_seq_id
                    current_gen_len = 0
                    if seq_id in seq_meta.seq_data:
                        sd = seq_meta.seq_data[seq_id]
                        prompt_len = sd.get_prompt_len()
                        total_len = sd.get_len()
                        # If is_prompt=True, this is prefill. Generated = 0.
                        # However, for continuous batching, prefill could be partial.
                        # The key is total tokens - user provided prompt.
                        # vLLM SequenceData generally tracks original prompt length.
                        current_gen_len = max(0, total_len - prompt_len)
                    
                    warmup_mask.append(current_gen_len < self.warmup_steps)
            
            # Convert mask to tensor and apply
            if len(warmup_mask) == choice.size(0):
                w_t = torch.tensor(warmup_mask, device=choice.device, dtype=torch.bool).view_as(choice)
                # If warmup is True, set choice to False (use Reasoner tokens)
                choice.masked_fill_(w_t, False)

        if self.controller_banned_token_tensor is not None:
            if not self.controller_banned_token_tensor.device == choice.device:
                self.controller_banned_token_tensor = self.controller_banned_token_tensor.to(choice.device)
            banned_mask = torch.isin(tokens_c_t, self.controller_banned_token_tensor)
            if banned_mask.any():
                choice.masked_fill_(banned_mask, False)

        final_tokens_t = torch.where(choice, tokens_c_t, tokens_r_t)

        final_tokens = final_tokens_t.tolist()
        mask = choice.tolist()

        merged_outputs = []
        idx = 0
        from vllm.sequence import SequenceOutput, Logprob, CompletionSequenceGroupOutput
        from vllm.model_executor.layers.sampler import SamplerOutput

        # Correctly iterate through reasoner outputs to structure the merged output
        for group_r in sr.outputs:
            samples = []
            for sample_r in group_r.samples:
                if idx >= len(final_tokens):
                    # Should not happen if shapes match
                    raise RuntimeError(f"Index {idx} out of bounds for final tokens of length {len(final_tokens)}. Check alignment logic.")
                    break
                    
                merged_sample = SequenceOutput(
                    parent_seq_id=sample_r.parent_seq_id,
                    output_token=final_tokens[idx],
                    is_intervened=mask[idx], # mask is boolean/0-1. is_intervened expects bool? Logprobs expects dict.
                    logprobs={final_tokens[idx]: Logprob(0.0)},
                    reasoner_token=sample_r.output_token,
                    controller_token=tokens_c[idx]
                )
                samples.append(merged_sample)
                idx += 1
            merged_outputs.append(CompletionSequenceGroupOutput(samples, None))

        return [SamplerOutput(merged_outputs, sr.hidden_states)]

    def _merge_outputs_dummy(self, out_r, out_c):
        if out_r: return out_r
        if out_c: return out_c
        return []
    
    def _free_seq_resources(self, seq_id):
        """Free private blocks associated with a sequence."""
        if seq_id not in self.seq_private_allocs:
            return
            
        pool = self.seq_private_allocs[seq_id]['pool']
        # Identify private blocks (IDs in our reserved range)
        private_to_free = [
            b for b in pool 
            if self.private_block_start_index <= b < self.private_block_end_index
        ]
        
        if private_to_free:
            self._free_private_blocks(private_to_free)
        
        del self.seq_private_allocs[seq_id]
    
    def _generate_masked_trace(self, current_tokens: List[int]) -> List[int]:
        """
        Generate R' from R (current_tokens) by replacing spans of consecutive special tokens.
        - Spans containing `user_token_id` are replaced by `self.user_replacement_ids` (padded).
        - Spans containing `assistant_token_id` are replaced by `self.assistant_replacement_ids` (padded).
        """
        r_prime = list(current_tokens)
        special_set = set(self.special_token_ids)
        if not special_set:
            special_set = {self.user_token_id, self.assistant_token_id}

        i = 0
        n = len(r_prime)
        while i < n:
            if r_prime[i] in special_set:
                # Found start of a special span
                j = i
                has_user = False
                has_asst = False
                while j < n and r_prime[j] in special_set:
                    if r_prime[j] == self.user_token_id: has_user = True
                    if r_prime[j] == self.assistant_token_id: has_asst = True
                    j += 1

                if has_user or has_asst:
                    span_len = j - i
                    
                    # Determine replacement target
                    target = []
                    if has_user:
                        target = self.user_replacement_ids
                    else:
                        target = self.assistant_replacement_ids
                    
                    # Assert length constraint
                    if len(target) > span_len:
                        # This assertion is critical for alignment safety
                        raise ValueError(f"Masked replacement length ({len(target)}) > Original special span length ({span_len}). Alignment broken.")

                    # Apply replacement + padding
                    for k in range(len(target)):
                        r_prime[i+k] = target[k]
                    for k in range(len(target), span_len):
                        r_prime[i+k] = self.pad_token_id
                    
                i = j
            else:
                i += 1
        return r_prime

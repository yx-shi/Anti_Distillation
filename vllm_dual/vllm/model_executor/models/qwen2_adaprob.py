# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                    maybe_prefix)
import inspect
import copy
import time

logger = init_logger(__name__)

    
class AdaprobLogitsProcessor(LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_as_input = True
        
    def forward(
        self,
        logits: torch.Tensor,
        agent_logits: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> Optional[torch.Tensor]:
        if logits is not None:
            # Apply logits processors (if any).
            if sampling_metadata is not None and \
                sampling_metadata.seq_groups is not None:
                _apply_logits_processors_fn = _apply_logits_processors_multi_seq
                logits = _apply_logits_processors_fn(logits, agent_logits, sampling_metadata)

        return logits

main_timer = {
    'timer1': {
        'sum': 0,
        'cnt': 0,
    },
    'timer2': {
        'sum': 0,
        'cnt': 0,
    },
    'timer3': {
        'sum': 0,
        'cnt': 0,
    },
}

def split_hf_config(config: VllmConfig) -> Tuple[VllmConfig, VllmConfig]:
    hf_config = config.model_config.hf_config
    base_hf_config = PretrainedConfig(**hf_config.base_hf_config)
    controller_hf_config = PretrainedConfig(**hf_config.controller_hf_config)
    base_config = copy.deepcopy(config)
    base_config.model_config.hf_config = base_hf_config
    controller_config = copy.deepcopy(config)
    controller_config.model_config.hf_config = controller_hf_config
    return base_config, controller_config


class Qwen2ForAdaprob(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        base_vllm_config, controller_vllm_config = split_hf_config(vllm_config)
        self.config = base_vllm_config.model_config.hf_config
        model_map = {
            'Qwen2ForCausalLM': Qwen2ForCausalLM,
            'Qwen3ForCausalLM': Qwen3ForCausalLM,
        }
        self.model = model_map[base_vllm_config.model_config.hf_config.architectures[0]](vllm_config=base_vllm_config,
                                       prefix=maybe_prefix(prefix, "model"))
        self.controller = model_map[controller_vllm_config.model_config.hf_config.architectures[0]](vllm_config=controller_vllm_config,
                                       prefix=maybe_prefix(prefix, "controller"))
        self.model_hidden_size = base_vllm_config.model_config.hf_config.hidden_size
        self.controller_hidden_size = controller_vllm_config.model_config.hf_config.hidden_size
        self.logits_processor = AdaprobLogitsProcessor(self.config.vocab_size)
        self.sampler = get_sampler()


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Compute both and branch during `compute_logits`
        start_time = time.perf_counter()
        model_hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        controller_hidden_states = self.controller(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        # controller_hidden_states = model_hidden_states.clone()
        hidden_states = torch.cat((model_hidden_states, controller_hidden_states), dim=-1)
        end_time = time.perf_counter()
        main_timer['timer3']['sum'] += end_time - start_time
        main_timer['timer3']['cnt'] += 1
        if main_timer['timer3']['cnt'] % 1000 == 0:
            logger.info(f"main timer3: {main_timer['timer3']['sum'] / main_timer['timer3']['cnt']}")
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        start_time = time.perf_counter()
        # model_hidden_states, controller_hidden_states = hidden_states.chunk(2, dim=-1)
        # split according to hidden size
        assert hidden_states.size(-1) == self.model_hidden_size + self.controller_hidden_size
        model_hidden_states = hidden_states[..., :self.model_hidden_size]
        controller_hidden_states = hidden_states[..., self.model_hidden_size:]
        logits_processors = []
        if sampling_metadata and sampling_metadata.seq_groups is not None:
            for seq_group in sampling_metadata.seq_groups:
                logits_processors.append(seq_group.sampling_params.logits_processors)
                seq_group.sampling_params.logits_processors = None
        model_logits = self.model.compute_logits(model_hidden_states, sampling_metadata)
        controller_logits = self.controller.compute_logits(controller_hidden_states, sampling_metadata)
        # controller_logits = model_logits.clone()
        end_time = time.perf_counter()
        main_timer['timer1']['sum'] += end_time - start_time
        main_timer['timer1']['cnt'] += 1
        start_time = time.perf_counter()
        if sampling_metadata and sampling_metadata.seq_groups is not None:
            for seq_group, logits_processor in zip(sampling_metadata.seq_groups, logits_processors):
                seq_group.sampling_params.logits_processors = logits_processor
        logits = self.logits_processor(model_logits, controller_logits, sampling_metadata)
        end_time = time.perf_counter()
        main_timer['timer2']['sum'] += end_time - start_time
        main_timer['timer2']['cnt'] += 1
        if main_timer['timer1']['cnt'] % 1000 == 0:
            logger.info(f"main timer1: {main_timer['timer1']['sum'] / main_timer['timer1']['cnt']}")
            logger.info(f"main timer2: {main_timer['timer2']['sum'] / main_timer['timer2']['cnt']}")
        return logits
        
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


timer = {
    'timer1': {
        'sum': 0,
        'cnt': 0
    },
    'timer2': {
        'sum': 0,
        'cnt': 0
    },
    'timer3': {
        'sum': 0,
        'cnt': 0
    }
}

def _apply_logits_processors_multi_seq(
    logits: torch.Tensor,
    agent_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    start_time = time.perf_counter()
    found_logits_processors = False
    logits_processed = 0
    # logits_row_ids_and_logits_row_futures = []
    logits_processors = sampling_metadata.seq_groups[0].sampling_params.logits_processors
    assert all(type(seq_group.sampling_params.logits_processors) == type(logits_processors) for seq_group in sampling_metadata.seq_groups)
    if logits_processors is None:
        return logits
    
    batch_logits = []
    batch_agent_logits = []
    batch_past_tokens_ids = []
    batch_prompt_tokens_ids = []
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        # if logits_processors:
        #     found_logits_processors = True
        for seq_id, logits_row_idx in zip(seq_ids,
                                        seq_group.sample_indices):
            logits_row = logits[logits_row_idx]
            agent_logits_row = agent_logits[logits_row_idx]
            past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
            prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

            batch_logits.append(logits_row)
            batch_agent_logits.append(agent_logits_row)
            batch_past_tokens_ids.append(past_tokens_ids)
            batch_prompt_tokens_ids.append(prompt_tokens_ids)

            # if _logits_processor_threadpool is not None:
            #     logits_row_ids_and_logits_row_futures.append(
            #         (logits_row_idx,
            #          _logits_processor_threadpool.submit(
            #              _apply_logits_processors_single_seq, logits_row,
            #              logits_processors, past_tokens_ids,
            #              prompt_tokens_ids)))
            # else:
            #     logits[logits_row_idx] = \
            #         _apply_logits_processors_single_seq(
            #             logits_row, logits_processors, past_tokens_ids,
            #             prompt_tokens_ids)

    end_time = time.perf_counter()
    # timer['timer1']['sum'] += end_time - start_time
    # timer['timer1']['cnt'] += 1


    if batch_logits:
        start_time = time.perf_counter()
        batch_logits = torch.stack(batch_logits)
        batch_agent_logits = torch.stack(batch_agent_logits)
        for logits_processor in logits_processors:
            parameters = inspect.signature(logits_processor).parameters
            if len(parameters) == 4:
                batch_logits = logits_processor(batch_prompt_tokens_ids, batch_past_tokens_ids,
                                            batch_logits, batch_agent_logits)
            elif len(parameters) == 3:
                batch_logits = logits_processor(batch_prompt_tokens_ids, batch_past_tokens_ids,
                                            batch_logits)
            else:
                batch_logits = logits_processor(batch_past_tokens_ids, batch_logits)

        end_time = time.perf_counter()
        # timer['timer2']['sum'] += end_time - start_time
        # timer['timer2']['cnt'] += 1

        start_time = time.perf_counter()
        i = 0
        for seq_group in sampling_metadata.seq_groups:
            for logits_row_idx in seq_group.sample_indices:
                logits[logits_row_idx] = batch_logits[i]
                i += 1

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)
        end_time = time.perf_counter()
        # timer['timer3']['sum'] += end_time - start_time
        # timer['timer3']['cnt'] += 1
        
    # for logits_row_idx, future in logits_row_ids_and_logits_row_futures:
    #     logits[logits_row_idx] = future.result()

    # if timer['timer1']['cnt'] % 100 == 0:
    #     print('logits processor timer1', timer['timer1']['sum'] / timer['timer1']['cnt'])
    #     print('logits processor timer2', timer['timer2']['sum'] / timer['timer2']['cnt'])
    #     print('logits processor timer3', timer['timer3']['sum'] / timer['timer3']['cnt'])

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]
    return logits

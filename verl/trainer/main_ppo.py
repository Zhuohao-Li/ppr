# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import asyncio

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., process_reward_enable=False, process_reward_config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.process_reward_enable = process_reward_enable
        self.process_reward_config = process_reward_config

    def _process_reward_per_turn_batch(self, rollout_results, reward_tensor):
        """Batch process reward model for all rollout_results (per turn).
        Args:
            rollout_results: list of dicts, each with 'index', 'turn_pos', 'prompt_str', 'response_str'
            reward_tensor: in-place operation
        Returns:
            (total_count, valid_count)
        """
        from verl.utils.reward_score.process_reward import compute_process_reward_in_batch, filter_process_score, write_tag_rewards
        
        total_count = len(rollout_results)
        valid_count = 0
        if total_count == 0:
            return 0, 0

        print(f"[INFO] Processing PPR with {total_count} conversations")

        try:
            batch_score_results = asyncio.run(compute_process_reward_in_batch(rollout_results))
            if batch_score_results is None:
                print(f"[WARN] Process reward batch returned None scores")
                return total_count, 0
        except Exception as e:
            print(f"[WARN] Process reward batch computation failed: {e}")
            return total_count, 0

        # Write all scores to reward tensor
        for rollout_result, score_result in zip(rollout_results, batch_score_results):
            try:
                # Validate score result first
                if not filter_process_score(score_result):
                    print(f"[WARN] Invalid process reward score for index {rollout_result['index']}, skipping")
                    continue
                
                score = score_result['score']
                max_score_extracted = score_result['max_score']
                
                write_tag_rewards(
                    reward_tensor=reward_tensor,
                    batch_idx=rollout_result['index'],
                    turn_pos=rollout_result['turn_pos'],
                    score=score,
                    max_score_extracted=max_score_extracted  # Use the max_score from process reward's response
                )
                valid_count += 1
            except Exception as e:
                print(f"[WARN] Failed to write process reward for index {rollout_result['index']} turn {rollout_result['turn_pos']}: {e}")
        
        return total_count, valid_count

    def _apply_process_reward_optimizations(self, data: DataProto, reward_tensor, beta):
        """Apply process reward optimizations"""
        print(f"[INFO] Applying process reward optimizations: {self.process_reward_enable}")
        
        if 'turn_mask' not in data.batch:
            return
        turn_mask = data.batch['turn_mask']
        
        for i in range(len(data)):
            boundary_indices = torch.nonzero(turn_mask[i]).flatten().tolist()
            if len(boundary_indices) <= 1:
                continue  # Need at least 2 positions (PPR + ORM)
            ppr_positions = boundary_indices[:-1]
            orm_position = boundary_indices[-1]
            ppr_scores = [reward_tensor[i, pos].item() for pos in ppr_positions]
            orm_score = reward_tensor[i, orm_position].item()

            enhanced_scores = [score + orm_score - beta for score in ppr_scores]
            
            # Update reward tensor with enhanced scores
            for pos, enhanced_score in zip(ppr_positions, enhanced_scores):
                reward_tensor[i, pos] = enhanced_score

    def __call__(self, data: DataProto):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        
        # Process reward tracking
        total_ppr_count = 0
        valid_ppr_count = 0

        # Process all rollouts first
        rollout_results = []
        for i in range(len(data)):
            data_item = data[i]
            
            # Process single rollout (reward computation, etc.)
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)

            if self.process_reward_enable:   # generate multi-turn qa pair
                response_positions = data_item.batch['turn_mask'].nonzero(as_tuple=False).flatten().tolist()
                last_turn_pos = 0

                ori_q = re.search(r'<\|im_start\|>user(.*?)<\|im_end\|>', prompt_str, re.DOTALL)
                ori_q = ori_q.group(1).strip()

                prompt_str_per_turn = ori_q
                response_str = ''

                for turn_idx in range(len(response_positions)):
                    response_str_per_turn = self.tokenizer.decode(valid_response_ids[last_turn_pos:response_positions[turn_idx]+1])
                    last_turn_pos = response_positions[turn_idx]+1

                    rollout_result = {
                        'index': i,
                        'turn_index': turn_idx,
                        'turn_pos': response_positions[turn_idx],
                        'prompt_str': prompt_str_per_turn,
                        'response_str': response_str_per_turn
                    }
                    if turn_idx<len(response_positions)-1:  # dont judge last <answer> turn, leave it to orm em
                        rollout_results.append(rollout_result)
                    prompt_str_per_turn = prompt_str_per_turn + response_str_per_turn
                    response_str += response_str_per_turn

            sequences_str = prompt_str + response_str
            
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            # Update reward tensor
            reward_tensor[i, valid_response_length - 1] = score

            # Handle printing logic
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # Process process reward if enabled
        if self.process_reward_enable:
            print(f"Trigger process reward model in {len(rollout_results)} conversations")
            total_ppr_count, valid_ppr_count = self._process_single_process_reward_per_turn_batch(rollout_results, reward_tensor)

            # Apply process reward optimizations if enabled
            if self.process_reward_config.renorm:
                beta = self.process_reward_config.renorm_beta
                self._apply_process_reward_optimizations(data, reward_tensor, beta)
            
            # Save process reward metrics
            if total_ppr_count > 0:
                valid_rate = valid_ppr_count / total_ppr_count
                print(f"[Process Reward Valid Rate] {valid_ppr_count}/{total_ppr_count} = {valid_rate:.3f}")
                
                # Save to data for metrics logging
                if not hasattr(data, 'process_reward_metrics'):
                    data.process_reward_metrics = {}
                data.process_reward_metrics['process_reward_valid_rate'] = valid_rate
                data.process_reward_metrics['process_reward_valid_count'] = valid_ppr_count
                data.process_reward_metrics['process_reward_total_count'] = total_ppr_count

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                            process_reward_enable=config.process_reward_enable,
                            process_reward_config=config.process_reward_config)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, include_llm_judge_score=config.llm_as_judge.include_score_val, 
                            process_reward_enable=False)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

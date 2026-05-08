# Search-R1 训练过程数据流

本文梳理本仓库当前 PPO/GRPO 训练链路中的数据流，重点说明入口文件、函数调用顺序、主要输入输出张量维度、奖励函数设置和数据集构造方式。

## 1. 训练入口

常用启动脚本：

- `train_ppo.sh`：PPO，`algorithm.adv_estimator=gae`，`actor_rollout_ref.rollout.n_agent=1`。
- `train_grpo.sh`：GRPO，`algorithm.adv_estimator=grpo`，`actor_rollout_ref.rollout.n_agent=5`。

两者最终都进入：

```text
verl/trainer/main_ppo.py:113 main(config)
  -> verl/trainer/main_ppo.py:123 main_task(config)
  -> verl/trainer/ppo/ray_trainer.py:313 RayPPOTrainer(...)
  -> RayPPOTrainer.init_workers()
  -> RayPPOTrainer.fit()
```

典型脚本覆盖的关键维度参数：

| 配置项 | 典型值 | 含义 |
| --- | ---: | --- |
| `data.train_batch_size` | 512 | DataLoader 每步读入的原始 prompt 数，记作 `B` |
| `data.val_batch_size` | 256 | 验证 batch size |
| `data.max_prompt_length` | 4096 | 训练样本 prompt 张量长度，记作 `P` |
| `data.max_response_length` | 500 | 每次生成的 response 最大长度，记作 `R` |
| `data.max_start_length` | 2048 | 多轮搜索初始 prompt 截断长度，记作 `S` |
| `data.max_obs_length` | 500 | 一次检索观察文本最大 token 数，记作 `O` |
| `max_turns` | 2 | 最多搜索/回答轮数 |
| `actor_rollout_ref.rollout.n_agent` | PPO: 1, GRPO: 5 | 每个问题采样几条 agent 轨迹，记作 `A` |
| `actor_rollout_ref.rollout.n` | 默认 1 | 单次 vLLM rollout 的采样条数 |

后文中：

- `B`：DataLoader 原始 batch size。
- `A`：`n_agent`，每条 prompt 重复的轨迹数。
- `B' = B * A`：进入 rollout/reward/update 的实际 batch size。
- `P`：`max_prompt_length`。
- `R`：`max_response_length`。
- `S`：`max_start_length`。

## 2. 数据集构造

### 2.1 NQ 搜索数据预处理

单数据集 NQ 的构造入口：

```text
scripts/data_process/nq_search.py
  -> make_prefix(dp, template_type)                       # line 54
  -> make_map_fn(split).process_fn(example, idx)          # line 89
  -> train_dataset.to_parquet(.../train.parquet)          # line 125
  -> test_dataset.to_parquet(.../test.parquet)            # line 126
```

`process_fn` 每条样本写入 parquet 的主要字段：

```python
{
    "data_source": "nq",
    "prompt": [{
        "role": "user",
        "content": question_with_search_instruction,
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": example["golden_answers"],
        },
    },
    "extra_info": {
        "split": split,
        "index": idx,
    },
}
```

prompt 模板要求模型：

- 先在 `<think>...</think>` 中推理。
- 缺知识时用 `<search> query </search>` 调用搜索。
- 搜索返回内容会放在 `<information>...</information>` 中。
- 确定答案后用 `<answer> answer </answer>` 输出最终答案。

多数据集合并脚本在 `scripts/nq_hotpotqa/data_process.sh` 中调用：

- `scripts/data_process/qa_search_train_merge.py`
- `scripts/data_process/qa_search_test_merge.py`

它们生成的样本结构与上面一致，只是 `data_source` 可覆盖 `nq/hotpotqa/triviaqa/popqa/2wikimultihopqa/musique/bamboogle` 等。

### 2.2 训练时读取 parquet

训练时数据读取链路：

```text
verl/trainer/ppo/ray_trainer.py:372 RayPPOTrainer._create_dataloader()
  -> verl/utils/dataset/rl_dataset.py:58 RLHFDataset(...)
  -> RLHFDataset._read_files_and_tokenize()               # line 96
  -> torch.utils.data.DataLoader(..., collate_fn=collate_fn)
```

`RLHFDataset.__getitem__` 在 `verl/utils/dataset/rl_dataset.py:120`：

1. 从 parquet 行里取出 `prompt`。
2. 如果 tokenizer 有 chat template，调用 `tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)`；否则使用 `chat[0]["content"]`。
3. 调用 `verl_F.tokenize_and_postprocess_data(...)` 做 tokenization、左 padding 和长度约束。
4. 调用 `compute_position_id_with_mask(attention_mask)` 生成 position ids。
5. 把 `extra_info.index` 提升为顶层字段 `index`，供 GRPO 分组。

单条样本输出：

| 字段 | 类型/维度 | 说明 |
| --- | --- | --- |
| `input_ids` | `LongTensor[P]` | 左 padding 后的 prompt token |
| `attention_mask` | `LongTensor[P]` | prompt 有效 token 为 1，pad 为 0 |
| `position_ids` | `LongTensor[P]` | 由 attention mask 累加得到 |
| `data_source` | object/string | 奖励函数路由用 |
| `reward_model` | object/dict | 包含 `ground_truth.target` |
| `ability` | object/string | 任务类型 |
| `extra_info` | object/dict | split/index 等 |
| `index` | object/int | 样本 id，GRPO 中用作同题分组 id |

`collate_fn` 在 `verl/utils/dataset/rl_dataset.py:31`：

- tensor 字段用 `torch.stack(..., dim=0)`。
- 非 tensor 字段转为 `np.array(..., dtype=object)`。

DataLoader 输出 batch：

| 字段 | 维度 |
| --- | --- |
| `input_ids` | `[B, P]` |
| `attention_mask` | `[B, P]` |
| `position_ids` | `[B, P]` |
| `data_source/reward_model/extra_info/index/...` | `[B]` object array |

## 3. Ray Worker 初始化

入口：

```text
verl/trainer/main_ppo.py:123 main_task(config)
  -> 根据 config.actor_rollout_ref.actor.strategy 选择 FSDP 或 Megatron worker
  -> role_worker_mapping:
       Role.ActorRollout -> ActorRolloutRefWorker
       Role.Critic       -> CriticWorker
       Role.RefPolicy    -> ActorRolloutRefWorker
       Role.RewardModel  -> RewardModelWorker, 仅 reward_model.enable=True 时
```

当前默认配置 `reward_model.enable=False`，所以训练主要使用规则奖励 `RewardManager`，不是神经 reward model。

FSDP worker 主要函数：

```text
verl/workers/fsdp_workers.py:285 ActorRolloutRefWorker.init_model()
  -> 构建 actor FSDP 模型
  -> 构建 vLLMRollout, 若 rollout.name=vllm
  -> 构建 ref policy, 若 role=ref

verl/workers/fsdp_workers.py:432 ActorRolloutRefWorker.generate_sequences()
verl/workers/fsdp_workers.py:401 ActorRolloutRefWorker.compute_log_prob()
verl/workers/fsdp_workers.py:356 ActorRolloutRefWorker.update_actor()
verl/workers/fsdp_workers.py:705 CriticWorker.compute_values()
```

## 4. 训练主循环

核心入口：

```text
verl/trainer/ppo/ray_trainer.py:654 RayPPOTrainer.fit()
```

每一步训练的数据流：

```text
batch_dict = next(train_dataloader)
  -> DataProto.from_single_dict(batch_dict)
  -> batch.repeat(repeat_times=n_agent, interleave=True)
  -> gen_batch = batch.pop(["input_ids", "attention_mask", "position_ids"])
  -> 搜索模式: LLMGenerationManager.run_llm_loop(...)
     非搜索模式: actor_rollout_wg.generate_sequences(...)
  -> actor_rollout_wg.compute_log_prob(...)
  -> batch.union(generation_output)
  -> ref_policy_wg.compute_ref_log_prob(...)
  -> critic_wg.compute_values(...), 仅 GAE/PPO 使用
  -> reward_fn(batch)
  -> apply_kl_penalty(...) 或直接使用 token_level_scores
  -> compute_advantage(...)
  -> critic_wg.update_critic(...), 仅 GAE/PPO 使用
  -> actor_rollout_wg.update_actor(...)
```

### 4.1 样本重复

在 `RayPPOTrainer.fit()` 中：

```python
batch = batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n_agent, interleave=True)
```

因此：

- 原始 DataLoader batch：`B`。
- 重复后实际 rollout batch：`B' = B * A`。
- PPO 默认 `A=1`。
- GRPO 默认 `A=5`，同一个问题会产生 5 条采样轨迹，后续用相同 `index/uid` 做组内归一化 advantage。

## 5. 搜索模式 Rollout 数据流

默认配置 `do_search=true`，因此进入：

```text
search_r1/llm_agent/generation.py:220 LLMGenerationManager.run_llm_loop()
```

输入：

| 输入 | 维度 | 说明 |
| --- | --- | --- |
| `gen_batch.batch["input_ids"]` | `[B', P]` | 原始 prompt |
| `gen_batch.batch["attention_mask"]` | `[B', P]` | 原始 prompt mask |
| `gen_batch.batch["position_ids"]` | `[B', P]` | 原始 prompt position |
| `initial_input_ids` | `[B', S]` | `input_ids[:, -max_start_length:]` |

### 5.1 多轮 agent 循环

`run_llm_loop` 内维护两侧状态：

- `original_left_side["input_ids"]`: `[B', S]`，最终作为 `prompts`。
- `original_right_side["responses"]`: `[B', variable_len]`，累积所有生成 action 和 observation。
- `rollings`: 当前送进模型的上下文，长度被裁剪到不超过 `P`。

每一轮：

```text
rollings.batch = TensorHelper.cut_to_effective_len(...)
  -> active 样本过滤
  -> _generate_with_gpu_padding(...)
  -> actor_rollout_wg.generate_sequences(...)
  -> vLLMRollout.generate_sequences(...)
  -> _postprocess_responses(...)
  -> execute_predictions(...)
  -> _update_rolling_state(...)
  -> _update_right_side(...)
```

关键函数：

- `generation.py:54 _postprocess_responses`：将 vLLM 输出 decode 成文本，并截断到第一个 `</search>` 或 `</answer>`。
- `generation.py:353 execute_predictions`：解析 `<search>...</search>` 或 `<answer>...</answer>`。
- `generation.py:438 batch_search`：向 `retriever.url` 发送 `{"queries": queries, "topk": topk, "return_scores": True}`。
- `generation.py:93 _update_rolling_state`：把当前 action 和搜索 observation 拼回下一轮模型输入。
- `generation.py:145 _update_right_side`：把 action 和 observation 累积到最终 response 侧。

### 5.2 单次 vLLM 生成

worker 调用：

```text
verl/workers/fsdp_workers.py:432 ActorRolloutRefWorker.generate_sequences()
  -> verl/workers/rollout/vllm_rollout/vllm_rollout.py:142 vLLMRollout.generate_sequences()
```

`vLLMRollout.generate_sequences()` 输入：

| 输入 | 维度 | 说明 |
| --- | --- | --- |
| `prompts.batch["input_ids"]` | `[b_active, L]` | 当前活跃样本上下文，左 padding |
| `attention_mask` | `[b_active, L]` | 当前上下文 mask |
| `position_ids` | `[b_active, L]` | 当前上下文 position |

输出：

| 输出字段 | 维度 | 说明 |
| --- | --- | --- |
| `prompts` | `[b_active, L]` | 本次输入 prompt |
| `responses` | `[b_active, R]` | 本次生成 response，右 padding 到 `R` |
| `input_ids` | `[b_active, L + R]` | prompt + response |
| `attention_mask` | `[b_active, L + R]` | prompt mask + response eos mask |
| `position_ids` | `[b_active, L + R]` | prompt position + response position |
| `old_log_probs` | `[b_active, R]` | 若 `recompute_log_prob=True`，worker 会额外重算 |

搜索 agent 中 `_postprocess_responses` 会把 `[b_active, R]` decode 后裁剪为 action 文本，再重新 tokenize 成 `[b_active, r_step]`，其中 `r_step <= R`，但具体长度由本轮 action 文本决定。

### 5.3 最终 rollout 输出

`generation.py:321 _compose_final_output()` 组合最终训练样本。

输出 `final_gen_batch_output.batch`：

| 字段 | 维度 | 说明 |
| --- | --- | --- |
| `prompts` | `[B', S]` | 初始 prompt 截断到 `S` |
| `responses` | `[B', T]` | 多轮 action + observation + final answer 的右侧序列，`T <= P` |
| `responses_with_info_mask` | `[B', T]` | 与 responses 同形；`<information>` 内容位置被 pad id 替换，用于 loss mask |
| `input_ids` | `[B', S + T]` | `prompts + responses` |
| `attention_mask` | `[B', S + T]` | prompts 和 responses 的有效 token mask |
| `info_mask` | `[B', S + T]` | information 区域置 0 的 mask |
| `position_ids` | `[B', S + T]` | 由 attention_mask 生成 |

`meta_info` 额外记录：

- `turns_stats`: 每条轨迹 action 轮数。
- `active_mask`: 结束时是否仍未完成。
- `valid_action_stats`: 有效 action 数。
- `valid_search_stats`: 有效 search 数。

随后训练循环调用：

```text
actor_rollout_wg.compute_log_prob(final_gen_batch_output)
```

返回：

| 字段 | 维度 |
| --- | --- |
| `old_log_probs` | `[B', T]` |

## 6. 奖励函数设置

默认规则奖励链路：

```text
verl/trainer/main_ppo.py:32 RewardManager
  -> RewardManager.__call__(data)                         # line 41
  -> _select_rm_score_fn(data_source)                     # line 25
  -> verl/utils/reward_score/qa_em.py:85 compute_score_em(...)
```

`_select_rm_score_fn` 支持的数据源：

```python
["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
```

### 6.1 RewardManager 输入输出

输入 `data: DataProto` 中至少包含：

| 字段 | 维度/类型 | 说明 |
| --- | --- | --- |
| `batch["prompts"]` | `[B', S]` 或非搜索时 `[B', P]` | prompt token |
| `batch["responses"]` | `[B', T]` 或 `[B', R]` | response token |
| `batch["attention_mask"]` | `[B', S+T]` | prompt+response mask |
| `non_tensor_batch["data_source"]` | `[B']` | 数据源 |
| `non_tensor_batch["reward_model"]` | `[B']` | 内含 `ground_truth.target` |

输出：

| 输出 | 维度 | 说明 |
| --- | --- | --- |
| `reward_tensor` | `[B', response_length]` | token-level score，只有最后一个有效 response token 放标量分数，其余为 0 |

代码逻辑：

1. 如果 `data.batch` 已有 `rm_scores`，直接返回它。
2. 否则创建 `torch.zeros_like(data.batch["responses"], dtype=torch.float32)`。
3. 对每条样本：
   - 通过 attention mask 找有效 prompt 和有效 response。
   - 拼接并 decode 成完整 `sequences_str`。
   - 取 `reward_model["ground_truth"]`。
   - 根据 `data_source` 选择 EM 奖励函数。
   - 把标量分数写入 `reward_tensor[i, valid_response_length - 1]`。

### 6.2 EM 奖励

`verl/utils/reward_score/qa_em.py`：

```text
extract_solution(solution_str)       # line 62
compute_score_em(...)                # line 85
```

规则：

- 用正则提取 `<answer>...</answer>`。
- 当前 `extract_solution` 要求匹配数量 `len(matches) > 1`，然后取最后一个 `<answer>` 内容；如果匹配数量为 0 或 1，返回 `None`。
- `answer is None`：得分 `0`。
- `answer` 与任一 `ground_truth["target"]` 做 normalize 后 exact match：
  - 匹配：返回 `score`，默认 `1.0`。
  - 不匹配但格式可提取：返回 `format_score`，默认 `0.0`。

normalize 包括：

- lowercase
- 去标点
- 去英文冠词 `a/an/the`
- 合并空白

注意：`train_grpo.sh` 和 `train_ppo.sh` 顶层脚本没有覆盖 `reward_model.structure_format_score/final_format_score/retrieval_score`；当前 `main_ppo.py` 构造 `RewardManager(tokenizer, num_examine=0)` 时也没有传 `format_score`，所以默认格式分是 `0.0`。`scripts/nq_hotpotqa/v0.3/*_format.sh` 会覆盖这些格式奖励配置，但当前 `RewardManager` 只接收 `format_score`，是否生效取决于对应入口是否把配置传进去。

## 7. KL、Advantage 和 Return

训练循环在 `RayPPOTrainer.fit()` 的 `adv` 阶段：

```text
reward_tensor = reward_fn(batch)
batch.batch["token_level_scores"] = reward_tensor

if not actor.use_kl_loss:
    apply_kl_penalty(...)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

compute_advantage(...)
```

### 7.1 PPO/GAE

配置：

```text
algorithm.adv_estimator=gae
actor_rollout_ref.actor.use_kl_loss=false
```

数据流：

```text
critic_wg.compute_values(batch)
  -> batch["values"]: [B', response_length]

apply_kl_penalty(data, kl_ctrl, kl_penalty)                # ray_trainer.py:91
  old_log_probs: [B', response_length]
  ref_log_prob:  [B', response_length]
  token_level_scores: [B', response_length]
  response_mask: [B', response_length]
  token_level_rewards = token_level_scores - beta * KL

compute_advantage(..., adv_estimator="gae")                # ray_trainer.py:123
  -> core_algos.compute_gae_advantage_return(...)
```

输出：

| 字段 | 维度 |
| --- | --- |
| `values` | `[B', response_length]` |
| `token_level_rewards` | `[B', response_length]` |
| `advantages` | `[B', response_length]` |
| `returns` | `[B', response_length]` |

### 7.2 GRPO

配置：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.rollout.n_agent=5
```

数据流：

```text
batch.non_tensor_batch["uid"] = batch.non_tensor_batch["index"].copy()
token_level_rewards = token_level_scores
compute_advantage(..., adv_estimator="grpo")
  -> core_algos.compute_grpo_outcome_advantage(...)
```

GRPO 不创建 critic worker，`self.use_critic=False`。

`compute_grpo_outcome_advantage` 输入：

| 输入 | 维度 | 说明 |
| --- | --- | --- |
| `token_level_rewards` | `[B', response_length]` | 最后有效 token 上有 outcome reward |
| `eos_mask` | `[B', response_length]` | response 有效 token mask |
| `index/uid` | `[B']` | 同一个原始问题的多条轨迹共享 id |

逻辑：

1. 将每条 response 的 token-level reward 求和为标量 `score`。
2. 按 `uid/index` 分组。
3. 组内做 `(score - mean) / (std + epsilon)`。
4. 将归一化后的标量铺满到 response 有效 token 上。

输出：

| 字段 | 维度 |
| --- | --- |
| `advantages` | `[B', response_length]` |
| `returns` | `[B', response_length]` |

## 8. Actor/Critic 更新

### 8.1 Actor log prob 和 policy update

Actor 实现在：

```text
verl/workers/actor/dp_actor.py:39 DataParallelPPOActor
  -> compute_log_prob(data)       # line 153
  -> update_policy(data)          # line 203
  -> _forward_micro_batch(...)    # line 58
```

`compute_log_prob` 输入：

| 字段 | 维度 |
| --- | --- |
| `responses` | `[B', response_length]` |
| `input_ids` | `[B', sequence_length]` |
| `attention_mask` | `[B', sequence_length]` |
| `position_ids` | `[B', sequence_length]` |

输出：

| 字段 | 维度 |
| --- | --- |
| `old_log_probs` 或 `ref_log_prob` | `[B', response_length]` |

`update_policy` 输入还包括：

| 字段 | 维度 | 说明 |
| --- | --- | --- |
| `old_log_probs` | `[B', response_length]` | rollout policy log prob |
| `advantages` | `[B', response_length]` | PPO/GRPO advantage |
| `ref_log_prob` | `[B', response_length]` | GRPO `use_kl_loss=true` 时用于 actor KL loss |
| `loss_mask` | `[B', response_length]` | `state_masking=true` 时使用，屏蔽 information token |

核心 loss：

- PPO clipped policy loss：`core_algos.compute_policy_loss(...)`
- entropy bonus：`policy_loss = pg_loss - entropy_loss * entropy_coeff`
- 若 `use_kl_loss=true`：额外加 `kl_loss * kl_loss_coef`

### 8.2 State masking

搜索训练中常用：

```text
actor_rollout_ref.actor.state_masking=true
```

训练循环在 actor update 前调用：

```text
verl/trainer/ppo/ray_trainer.py:854 _create_loss_mask(batch, metrics)
```

逻辑：

- `response_mask = attention_mask[:, -response_length:]`
- `loss_mask = info_mask[:, -response_length:]`
- 写入 `batch.batch["loss_mask"]`

`info_mask` 来自 `generation.py:321 _compose_final_output()`，其中 `<information>...</information>` 检索返回内容被 mask 掉。因此 actor loss 不会鼓励模型去学习复述检索器返回的 observation token，但仍会学习自己生成的 `<search>`、`<answer>`、推理 token。

### 8.3 Critic update

PPO/GAE 才启用 critic：

```text
verl/workers/fsdp_workers.py:705 CriticWorker.compute_values()
verl/workers/fsdp_workers.py update_critic(...)
```

`compute_values` 返回：

| 字段 | 维度 |
| --- | --- |
| `values` | `[B', response_length]` |

critic loss 使用：

```text
core_algos.compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value)
```

输入 `returns/values/eos_mask` 均为 `[B', response_length]`。

## 9. 验证流程

验证入口：

```text
verl/trainer/ppo/ray_trainer.py:436 RayPPOTrainer._validate()
```

流程与训练 rollout 类似，但：

- `test_gen_batch.meta_info["do_sample"] = False`，即 greedy 生成。
- `test_gen_batch.meta_info["validate"] = True`。
- 不更新 actor/critic。
- 使用 `val_reward_fn = RewardManager(tokenizer, num_examine=1)` 计算规则奖励。
- 最终按 `data_source` 汇总 `val/test_score/{data_source}`。

验证 reward 汇总时会对 `reward_tensor.sum(-1)`，得到每条样本的标量分数 `[B_val]`。

## 10. 总览图

```text
parquet row
  data_source, prompt, reward_model.ground_truth, extra_info.index
        |
        v
RLHFDataset.__getitem__
  input_ids/attention_mask/position_ids: [P]
        |
        v
DataLoader + collate_fn
  tensor batch: [B, P]
  non_tensor batch: [B]
        |
        v
DataProto.from_single_dict
        |
        v
repeat by n_agent
  tensor batch: [B', P], B'=B*A
        |
        v
LLMGenerationManager.run_llm_loop
  multi-turn vLLM generation + search API observation
        |
        v
final_gen_batch_output
  prompts: [B', S]
  responses: [B', T]
  input_ids/attention_mask/info_mask/position_ids: [B', S+T]
        |
        v
compute_log_prob / ref_log_prob / values
  old_log_probs: [B', T]
  ref_log_prob:  [B', T]
  values:        [B', T] only PPO/GAE
        |
        v
RewardManager
  token_level_scores: [B', T]
  only last valid response token has EM reward
        |
        v
KL + Advantage
  PPO/GAE: token_level_rewards, advantages, returns: [B', T]
  GRPO: group-normalized advantages/returns: [B', T]
        |
        v
update_critic, update_actor
```


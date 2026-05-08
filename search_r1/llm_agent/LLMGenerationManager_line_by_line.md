# LLMGenerationManager 逐行解析

本文逐行解析 `search_r1/llm_agent/generation.py` 中的 `GenerationConfig` 和 `LLMGenerationManager`。重点说明每个函数的作用、调用到的库函数/项目函数怎么用，以及训练中关键张量的数据流和维度。

## 0. 维度约定

后文统一使用以下符号：

| 符号 | 含义 |
| --- | --- |
| `B` | 当前 rollout batch size，训练中通常是 `train_batch_size * n_agent` |
| `b` | 当前仍 active 的样本数，`b <= B` |
| `P` | `config.max_prompt_length` |
| `S` | `config.max_start_length` |
| `R` | `config.max_response_length`，单次 vLLM 生成的最大长度 |
| `O` | `config.max_obs_length`，一次检索 observation 最大 token 数 |
| `L` | 当前 rolling prompt 的有效长度，`L <= P` |
| `T` | 最终 response 侧累积长度，`T <= P` |
| `r` | 当前轮模型输出经截断和重新 tokenize 后的长度，`r <= R` |
| `o` | 当前轮 observation token 长度，`o <= O` |

`DataProto` 是 veRL 的数据容器，主要包含：

- `batch`：tensor 字段，例如 `input_ids/attention_mask/position_ids/responses`。
- `non_tensor_batch`：非 tensor 字段。
- `meta_info`：元信息字典。

## 1. import 部分，line 1-11

```python
1  import torch
```

引入 PyTorch。本文涉及的主要用法：

- `torch.Tensor`：张量类型。
- `torch.cat(tensors, dim=1)`：按序列维拼接多个 `[B, length]` 张量。
- `torch.full(size, fill_value, dtype, device)`：创建指定形状、值、类型和设备的张量。
- `torch.ones/zeros(shape, dtype=...)`：创建全 1/全 0 张量。
- `torch.tensor(list, dtype=...)`：从 Python list 创建张量。

```python
2  import re
```

Python 正则库。这里用于从模型输出文本里提取 `<search>...</search>` 或 `<answer>...</answer>`。

```python
3  from collections import defaultdict
4  import os
```

当前文件中没有实际使用 `defaultdict` 和 `os`。

```python
5  from typing import List, Dict, Any, Tuple
```

类型标注：

- `List[str]`：字符串列表。
- `Dict`：字典。
- `Any`：任意类型。
- `Tuple`：元组返回值。

```python
6  from dataclasses import dataclass
```

`@dataclass` 会自动生成配置类的 `__init__`，使 `GenerationConfig(...)` 可以直接用字段初始化。

```python
7  from .tensor_helper import TensorHelper, TensorConfig
```

引入本目录的辅助类，定义在 `search_r1/llm_agent/tensor_helper.py`。

关键方法：

- `TensorHelper.concatenate_with_padding(...)`：拼接后重新整理 padding。
- `TensorHelper.create_attention_mask(input_ids)`：`input_ids != pad_token_id` 的位置为 1。
- `TensorHelper.create_position_ids(attention_mask)`：用 `cumsum` 生成 position id。
- `TensorHelper.cut_to_effective_len(...)`：按当前 batch 最大有效长度裁剪。
- `TensorHelper._example_level_pad(...)`：active 样本生成后，补回完整 batch 位置。

```python
8  from verl import DataProto
```

veRL 数据容器。本文中常用：

- `DataProto.from_dict({"input_ids": tensor, ...})`：从 tensor 字典构造 `DataProto`。
- `data.batch["input_ids"]`：读取 tensor 字段。
- `data.meta_info.update(...)`：更新元信息。

```python
9  from verl.utils.tracking import Tracking
10 import shutil
```

当前文件中没有实际使用 `Tracking` 和 `shutil`。

```python
11 import requests
```

HTTP 请求库。这里用 `requests.post(url, json=payload).json()` 调用检索服务。

## 2. GenerationConfig，line 13-23

```python
13 @dataclass
14 class GenerationConfig:
15     max_turns: int
16     max_start_length: int
17     max_prompt_length: int 
18     max_response_length: int
19     max_obs_length: int
20     num_gpus: int
21     no_think_rl: bool=False
22     search_url: str = None
23     topk: int = 3
```

这是多轮搜索生成的配置对象。

字段含义：

- `max_turns`：主循环里最多执行多少轮“生成 action -> 执行 action -> 拼 observation”。
- `max_start_length`：从原始 prompt 末尾截取多少 token 作为最终 `prompts` 的左侧输入，维度记作 `S`。
- `max_prompt_length`：rolling state 和最终右侧 response 的最大保留长度，记作 `P`。
- `max_response_length`：单次 vLLM 输出最大 token 数，记作 `R`。本文件只保存配置，真实使用在 rollout worker 里。
- `max_obs_length`：检索返回 observation tokenize 后最大长度，记作 `O`。
- `num_gpus`：生成时需要保证 batch size 能被 GPU 数整除。
- `no_think_rl`：当前实现中如果为 True 会直接 `raise ValueError('stop')`，等于未启用。
- `search_url`：检索服务地址，例如 `http://127.0.0.1:8000/retrieve`。
- `topk`：每个 query 取多少篇检索文档。

## 3. LLMGenerationManager 初始化，line 25-43

```python
25 class LLMGenerationManager:
26     def __init__(
27         self,
28         tokenizer,
29         actor_rollout_wg,
30         config: GenerationConfig,
31         is_validation: bool = False,
32     ):
```

构造函数。输入：

- `tokenizer`：HuggingFace tokenizer，提供 `pad_token_id/pad_token/batch_decode/__call__` 等方法。
- `actor_rollout_wg`：Ray worker group，主要调用 `generate_sequences(active_batch)`。
- `config`：上面的 `GenerationConfig`。
- `is_validation`：是否验证模式，当前类内部只保存，没有分支使用。

```python
33         self.tokenizer = tokenizer
34         self.actor_rollout_wg = actor_rollout_wg
35         self.config = config
36         self.is_validation = is_validation
```

保存依赖对象。

```python
38         self.tensor_fn = TensorHelper(TensorConfig(
39             pad_token_id=tokenizer.pad_token_id,
40             max_prompt_length=config.max_prompt_length,
41             max_obs_length=config.max_obs_length,
42             max_start_length=config.max_start_length
43         ))
```

创建 tensor 辅助对象。

`TensorConfig` 的字段来自 tokenizer 和 generation config：

- `pad_token_id`：padding token id。
- `max_prompt_length/max_obs_length/max_start_length`：用于裁剪和 padding 处理。

## 4. `_batch_tokenize`，line 45-52

```python
45     def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
46         """Tokenize a batch of responses."""
47         return self.tokenizer(
48             responses, 
49             add_special_tokens=False, 
50             return_tensors='pt', 
51             padding="longest"
52         )['input_ids']
```

函数作用：把一批字符串 tokenize 成 padded token id。

输入：

- `responses`: `List[str]`，长度为 `b` 或 `B`。

调用的 tokenizer 方法：

- `tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding="longest")`
- `add_special_tokens=False`：不额外添加 BOS/EOS 等特殊 token。
- `return_tensors="pt"`：返回 PyTorch tensor。
- `padding="longest"`：按当前 batch 内最长字符串 padding。

输出：

- `input_ids`: `LongTensor[N, l]`
- `N = len(responses)`
- `l = 当前 responses tokenize 后的最大长度`

注意：这里只有 `input_ids`，没有返回 `attention_mask`。

## 5. `_postprocess_responses`，line 54-75

```python
54     def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
55         """Process responses to stop at search operation or answer operation."""
```

函数作用：把 vLLM 的 token 输出转为文本，截断到第一个完整 search/answer action，然后重新 tokenize。

输入：

- `responses`: `LongTensor[b, R]`，来自 vLLM rollout 的本轮输出。

```python
56         responses_str = self.tokenizer.batch_decode(
57             responses, 
58             skip_special_tokens=True
59         )
```

调用 HuggingFace tokenizer 的 `batch_decode`：

- 输入 `[b, R]` token ids。
- 输出 `List[str]`，长度为 `b`。
- `skip_special_tokens=True` 会跳过 tokenizer 定义的特殊 token，例如 pad/eos 等。

```python
61         responses_str = [resp.split('</search>')[0] + '</search>'
62                  if '</search>' in resp 
63                  else resp.split('</answer>')[0] + '</answer>'
64                  if '</answer>' in resp 
65                  else resp
66                  for resp in responses_str]
```

逐条文本裁剪：

- 如果出现 `</search>`，保留第一个 `</search>` 之前的内容，并补回 `</search>`。
- 否则如果出现 `</answer>`，保留第一个 `</answer>` 之前的内容，并补回 `</answer>`。
- 否则保留原文本。

这一步让当前轮 action 停在一个 search 或 answer 操作处。注意它优先处理 `</search>`，如果文本中同时出现 search 和 answer，search 优先。

```python
68         if self.config.no_think_rl:
69             raise ValueError('stop')
70             # if no_think_rl is enabled, only keep action in the str
71             actions, _ = self.env.postprocess_predictions(responses_str)
72             responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
73             print("RESPONSES:", responses_str)
```

`no_think_rl=True` 分支目前不可用，因为直接抛出异常。后面的 `self.env` 和 `envs` 也没有在类里定义。

```python
74         responses = self._batch_tokenize(responses_str)
75         return responses, responses_str
```

把裁剪后的文本重新 tokenize。

输出：

- `responses`: `LongTensor[b, r]`，`r` 是裁剪后文本的 batch 最大 token 长度。
- `responses_str`: `List[str]`，长度 `b`。

这里输出长度 `r` 不一定等于 vLLM 的 `R`，通常更短。

## 6. `_process_next_obs`，line 77-91

```python
77     def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
78         """Process next observations from environment."""
```

函数作用：将搜索环境返回的 observation 文本 tokenize，并截断到 `max_obs_length`。

输入：

- `next_obs`: `List[str]`，长度为 `B`。非 active 或 answer 完成的样本通常是空字符串。

```python
80         next_obs_ids = self.tokenizer(
81             next_obs, 
82             padding='longest',
83             return_tensors='pt',
84             add_special_tokens=False,
85         )['input_ids']
```

调用 tokenizer：

- `padding="longest"`：按最长 observation padding。
- `return_tensors="pt"`：返回 PyTorch tensor。
- `add_special_tokens=False`：不加额外特殊 token。

输出临时维度：

- `next_obs_ids`: `LongTensor[B, o_raw]`

```python
87         if next_obs_ids.shape[1] > self.config.max_obs_length:
88             print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
89             next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
```

如果 observation 太长，从右侧不保留，直接取前 `O` 个 token。

输出：

- `next_obs_ids`: `LongTensor[B, o]`
- `o = min(o_raw, O)`

## 7. `_update_rolling_state`，line 93-118

```python
93     def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
94                             next_obs_ids: torch.Tensor) -> Dict:
95         """Update rolling state with new responses and observations."""
```

函数作用：更新下一轮送入 LLM 的上下文。它把“旧 rolling 输入 + 当前模型 action + 当前 observation”拼起来，重新生成 mask/position，并裁剪到不超过 `P`。

输入：

| 参数 | 维度 | 含义 |
| --- | --- | --- |
| `rollings.batch["input_ids"]` | `[B, L_old]` | 当前轮送入 LLM 的上下文 |
| `rollings.batch["attention_mask"]` | `[B, L_old]` | 当前上下文 mask |
| `rollings.batch["position_ids"]` | `[B, L_old]` | 当前上下文 position |
| `cur_responses` | `[B, r]` | 当前轮 action，已经补回完整 batch |
| `next_obs_ids` | `[B, o]` | 当前轮 observation |

```python
97         new_input_ids = self.tensor_fn.concatenate_with_padding([
98             rollings.batch['input_ids'],
99             cur_responses,
100            next_obs_ids
101        ])
```

调用 `TensorHelper.concatenate_with_padding`：

1. 内部先执行 `torch.cat(tensors, dim=1)`，把 `[B, L_old] + [B, r] + [B, o]` 拼成 `[B, L_old+r+o]`。
2. 再调用 `convert_pad_structure(..., pad_to_left=True)`，把非 pad token 排到右边，即左 padding。

输出：

- `new_input_ids`: `LongTensor[B, L_old+r+o]`，左 padding 布局。

```python
104        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
105        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
```

`create_attention_mask` 使用：

```python
torch.where(input_ids != pad_token_id, 1, 0)
```

输出 `[B, L_new]`，非 pad 为 1。

`create_position_ids` 使用：

```python
(torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
```

输出 `[B, L_new]`，pad 位置为 0，有效 token 从 0 递增。

```python
108        effective_len = new_attention_mask.sum(dim=1).max()
109        max_len = min(self.config.max_prompt_length, effective_len)
```

`new_attention_mask.sum(dim=1)` 得到每条样本有效 token 数 `[B]`。

`.max()` 取当前 batch 最大有效长度，避免保留全 pad 前缀。

`max_len <= P`。

```python
111        new_rollings = DataProto.from_dict({
112            'input_ids': new_input_ids[:, -max_len:],
113            'position_ids': new_position_ids[:, -max_len:],
114            'attention_mask': new_attention_mask[:, -max_len:]
115        })
```

构造新的 `DataProto`。因为 rolling state 是左 padding，所以用 `[:, -max_len:]` 保留最右侧有效上下文。

输出字段维度：

| 字段 | 维度 |
| --- | --- |
| `input_ids` | `[B, max_len]` |
| `position_ids` | `[B, max_len]` |
| `attention_mask` | `[B, max_len]` |

```python
116        new_rollings.meta_info.update(rollings.meta_info)
118        return new_rollings
```

继承原来的 meta 信息并返回。这个返回值用于下一轮 LLM 生成。

## 8. `_info_masked_concatenate_with_padding`，line 120-143

```python
120    def _info_masked_concatenate_with_padding(self, 
121                prompt: torch.Tensor, 
122                prompt_with_mask: torch.Tensor, 
123                response: torch.Tensor, 
124                info: torch.Tensor = None,
125                pad_to_left: bool = True
126            ) -> torch.Tensor:
127        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
```

函数作用：维护最终 response 侧序列，同时维护一个用于 loss mask 的版本。它会把 `info` 这段替换成 pad token，但保留模型自己生成的 `response`，包括 `<search> query </search>`。

这里的参数名 `prompt` 容易误解。在实际调用中：

- `prompt = right_side["responses"]`：历史累积的右侧内容，不是最开始的问题。
- `prompt_with_mask = right_side["responses_with_info_mask"]`：历史右侧内容的 masked 版本。
- `response = cur_responses`：当前轮模型生成的 action。
- `info = next_obs_ids`：当前轮搜索返回的 observation。

输入维度：

| 参数 | 维度 | 说明 |
| --- | --- | --- |
| `prompt` | `[B, T_old]` | 历史 response 侧内容 |
| `prompt_with_mask` | `[B, T_old]` | 历史 response 侧 masked 内容 |
| `response` | `[B, r]` | 当前轮模型 action |
| `info` | `None` 或 `[B, o]` | 当前轮 observation |

```python
128        pad_id = self.tokenizer.pad_token_id
129        tensors = [prompt, response]
130        tensors_with_mask = [prompt_with_mask, response]
```

准备两条并行序列：

- `tensors`：真实 token 序列。
- `tensors_with_mask`：用于生成 `info_mask` 的序列。

当前模型输出的 `response` 在两个序列里都保留，所以 search query 不会被 mask。

```python
131        if info is not None:
132            tensors.append(info)
133            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
134            tensors_with_mask.append(info_mask)
```

如果有 observation：

- 真实序列追加 `info`。
- masked 序列追加同形状的 pad token。

`torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)` 用法：

- `info.size()` 返回 `[B, o]`。
- 创建 `[B, o]` 的张量，每个位置都是 `pad_id`。
- dtype/device 与 `info` 保持一致。

因此只有搜索返回的 `<information>...</information>` 内容会在 masked 版本中被置为 pad。`<search> query </search>` 作为模型 response 的一部分不会被置 pad。

```python
136        concatenated = torch.cat(tensors, dim=1)
137        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
```

沿序列维拼接：

- 如果 `info is not None`：
  - `concatenated`: `[B, T_old+r+o]`
  - `concatenated_with_info`: `[B, T_old+r+o]`
- 如果 `info is None`：
  - 二者都是 `[B, T_old+r]`

```python
138        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
139        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
```

这里通过排序来重新整理 padding。

`mask.to(torch.int64)`：

- bool mask 转成 0/1。

`argsort(dim=1, stable=True)`：

- 返回排序后的索引。
- `stable=True` 保持相同值元素的相对顺序。

当 `pad_to_left=False`：

- `mask = concatenated == pad_id`
- pad 位置是 1，非 pad 是 0
- argsort 后非 pad 在前、pad 在后，也就是右 padding

当 `pad_to_left=True`：

- `mask = concatenated != pad_id`
- pad 是 0，非 pad 是 1
- argsort 后 pad 在前、非 pad 在后，也就是左 padding

实际 `_update_right_side` 传的是 `pad_to_left=False`，所以最终 response 侧是右 padding。

```python
140        padded_tensor = concatenated.gather(1, sorted_indices)
141        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
```

`Tensor.gather(dim, index)` 按 `index` 从指定维度取值。

这里 `sorted_indices` 形状 `[B, T_new]`，所以：

- `padded_tensor`: 真实序列重新排列后 `[B, T_new]`
- `padded_tensor_with_info`: masked 序列按同一索引重排后 `[B, T_new]`

重要点：两个序列使用同一 `sorted_indices`，保证真实序列和 masked 序列 token 对齐。

```python
143        return padded_tensor, padded_tensor_with_info
```

返回：

- `padded_tensor`：真实 response 侧。
- `padded_tensor_with_info`：把 observation 区域替换为 pad 的 response 侧。

## 9. `_update_right_side`，line 145-167

```python
145    def _update_right_side(self, right_side: Dict, 
146                          cur_responses: torch.Tensor,
147                          next_obs_ids: torch.Tensor = None) -> Dict:
148        """Update right side state."""
```

函数作用：更新最终输出里的右侧 response 序列。它和 `_update_rolling_state` 的区别是：

- `_update_rolling_state` 更新下一轮 LLM 输入上下文。
- `_update_right_side` 更新最终用于训练 loss/reward 的 response 侧内容。

输入：

| 参数 | 维度 | 说明 |
| --- | --- | --- |
| `right_side["responses"]` | `[B, T_old]` | 历史真实 response 侧 |
| `right_side["responses_with_info_mask"]` | `[B, T_old]` | 历史 masked response 侧 |
| `cur_responses` | `[B, r]` | 当前轮 action |
| `next_obs_ids` | `None` 或 `[B, o]` | 当前轮 observation |

```python
149        if next_obs_ids != None:
150            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
151                    right_side['responses'],
152                    right_side['responses_with_info_mask'],
153                    cur_responses,
154                    next_obs_ids, 
155                    pad_to_left=False
156                )
```

有 observation 时，拼接历史 response、当前 action、当前 observation。`pad_to_left=False` 表示输出右 padding。

```python
157        else:
158            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
159                    right_side['responses'],
160                    right_side['responses_with_info_mask'],
161                    cur_responses,
162                    pad_to_left=False
163                )
```

没有 observation 时，只拼接历史 response 和当前 action。最终一轮 answer 通常走这个分支。

```python
164        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
165        max_len = min(self.config.max_prompt_length, effective_len)
```

计算当前 response 侧最大有效长度，并限制不超过 `P`。

因为 right side 是右 padding，后续要保留左侧 `:max_len`。

```python
167        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
```

输出：

| 字段 | 维度 |
| --- | --- |
| `responses` | `[B, T_new]` |
| `responses_with_info_mask` | `[B, T_new]` |

其中 `T_new <= P`。

## 10. `_generate_with_gpu_padding`，line 169-218

```python
169    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
```

函数作用：封装 `actor_rollout_wg.generate_sequences`。当 active batch size 不能被 GPU 数整除时，复制第一条样本补齐，生成后再裁掉补齐样本。

输入：

| 字段 | 维度 |
| --- | --- |
| `active_batch.batch["input_ids"]` | `[b, L]` |
| `active_batch.batch["attention_mask"]` | `[b, L]` |
| `active_batch.batch["position_ids"]` | `[b, L]` |

```python
176        num_gpus = self.config.num_gpus
177        if num_gpus <= 1:
178            return self.actor_rollout_wg.generate_sequences(active_batch)
```

如果只有 1 个 GPU，直接生成。

`actor_rollout_wg.generate_sequences(active_batch)` 是 Ray worker group 方法，底层进入 `ActorRolloutRefWorker.generate_sequences`，再进入 rollout 实现，例如 vLLM rollout。

```python
180        batch_size = active_batch.batch['input_ids'].shape[0]
181        remainder = batch_size % num_gpus
```

计算 active batch size 是否能整除 GPU 数。

```python
183        for key in active_batch.batch.keys():
184            active_batch.batch[key] = active_batch.batch[key].long()
```

把所有 tensor 转成 `torch.int64`。`Tensor.long()` 是 PyTorch 类型转换方法，等价于 `.to(torch.int64)`。

```python
185        if remainder == 0:
186            return self.actor_rollout_wg.generate_sequences(active_batch)
```

如果可整除，直接生成。

```python
189        padding_size = num_gpus - remainder
190        padded_batch = {}
```

需要补齐的样本数。

```python
192        for k, v in active_batch.batch.items():
193            # Use first sequence as padding template
194            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
195            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
```

逐个 tensor 字段补 batch 维。

`v[0:1]` 保留第一条样本，形状 `[1, L]`。

`repeat(padding_size, *[1] * (len(v.shape) - 1))`：

- 对二维 tensor `[1, L]`，等价于 `.repeat(padding_size, 1)`。
- 得到 `[padding_size, L]`。

`torch.cat([v, pad_sequence], dim=0)`：

- 沿 batch 维拼接。
- 从 `[b, L]` 变成 `[b + padding_size, L]`。

```python
197        padded_active_batch = DataProto.from_dict(padded_batch)
198        for key in padded_active_batch.batch.keys():
199            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
```

构造补齐后的 `DataProto` 并确保 dtype 为 long。

```python
202        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
```

调用 worker group 生成。

典型输出字段：

| 字段 | 维度 |
| --- | --- |
| `responses` | `[b + padding_size, R]` |
| `input_ids` | `[b + padding_size, L + R]` |
| `attention_mask` | `[b + padding_size, L + R]` |
| `position_ids` | `[b + padding_size, L + R]` |

```python
205        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
```

去掉最后补齐的样本，恢复 batch size `b`。

```python
208        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
209            trimmed_meta = {}
210            for k, v in padded_output.meta_info.items():
211                if isinstance(v, torch.Tensor):
212                    trimmed_meta[k] = v[:-padding_size]
213                else:
214                    trimmed_meta[k] = v
215            padded_output.meta_info = trimmed_meta
```

如果 `meta_info` 里有 tensor，也裁掉补齐样本。非 tensor 的 meta 信息原样保留。

```python
217        padded_output.batch = trimmed_batch
218        return padded_output
```

返回恢复到 `[b, ...]` 的生成结果。

## 11. `run_llm_loop`，line 220-319

```python
220    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
221        """Run main LLM generation loop."""
```

函数作用：执行多轮搜索 agent 轨迹生成。每轮流程是：

```text
当前上下文 -> LLM 生成 action -> 截断 action -> 解析 action
          -> search/answer/invalid 分支
          -> observation tokenize
          -> 更新 rolling state 和最终 response 侧
```

输入：

| 输入 | 维度 | 说明 |
| --- | --- | --- |
| `gen_batch.batch["input_ids"]` | `[B, P]` | 初始 prompt |
| `gen_batch.batch["attention_mask"]` | `[B, P]` | 初始 prompt mask |
| `gen_batch.batch["position_ids"]` | `[B, P]` | 初始 prompt position |
| `initial_input_ids` | `[B, S]` 或 `[B, P]` | 调用方通常传 `input_ids[:, -S:]` |

```python
223        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
```

保存最终输出左侧 prompt。维度 `[B, S]`。

即使传入的 `initial_input_ids` 更长，也只保留最后 `S` 个 token。

```python
224        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
```

初始化最终输出右侧为空序列。

`initial_input_ids[:, []]` 是 PyTorch 高级索引，取空列，形状 `[B, 0]`。

两个字段：

- `responses`: 真实右侧 token。
- `responses_with_info_mask`: observation 被 pad 掉的右侧 token。

```python
226        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
227        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
228        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
229        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
```

初始化轨迹状态，长度都是 `[B]`。

- `active_mask`: True 表示样本还没结束。
- `turns_stats`: action 轮数统计，初始为 1。
- `valid_action_stats`: 有效 action 数。
- `valid_search_stats`: 有效 search 数。

```python
230        active_num_list = [active_mask.sum().item()]
231        rollings = gen_batch
```

`active_mask.sum()` 返回当前 active 数；`.item()` 转 Python 标量。

`rollings` 是每轮送给 LLM 的上下文，初始为原始 prompt。

### 11.1 主循环，line 233-277

```python
233        # Main generation loop
234        for step in range(self.config.max_turns):
235            if not active_mask.sum():
236                break
```

最多执行 `max_turns` 轮。如果没有 active 样本，提前退出。

```python
237            rollings.batch = self.tensor_fn.cut_to_effective_len(
238                rollings.batch,
239                keys=['input_ids', 'attention_mask', 'position_ids']
240            )
```

裁剪 rolling state 到当前 batch 最大有效长度。

`TensorHelper.cut_to_effective_len`：

- 计算 `effective_len = attention_mask.sum(dim=1).max()`。
- 默认 `cut_left=True`，保留每个 key 的最右 `effective_len` 个位置。

输入可能是 `[B, P]`，输出是 `[B, L]`，`L <= P`。

```python
243            rollings_active = DataProto.from_dict({
244                k: v[active_mask] for k, v in rollings.batch.items()
245            })            
```

只取 active 样本生成。

`v[active_mask]` 是布尔索引：

- 输入 `[B, L]`
- 输出 `[b, L]`

```python
246            gen_output = self._generate_with_gpu_padding(rollings_active)
```

生成当前 active 样本的输出。典型输出：

| 字段 | 维度 |
| --- | --- |
| `gen_output.batch["responses"]` | `[b, R]` |
| `gen_output.batch["input_ids"]` | `[b, L+R]` |
| `gen_output.batch["attention_mask"]` | `[b, L+R]` |
| `gen_output.batch["position_ids"]` | `[b, L+R]` |

```python
248            meta_info = gen_output.meta_info
```

保存本轮生成的 meta 信息。后面会追加统计字段。

```python
249            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
```

把 `[b, R]` 的 raw response 截断成一个 action，并重新 tokenize。

输出：

- `responses_ids`: `[b, r]`
- `responses_str`: 长度 `b`

```python
250            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
```

把 active-only 结果补回完整 batch。

`TensorHelper._example_level_pad`：

- 创建 `[B, r]` 的 pad tensor。
- 把 `responses_ids` 填到 active 对应行。
- 非 active 行保持 pad。
- 字符串同理，非 active 行是 `""`。

输出：

- `responses_ids`: `[B, r]`
- `responses_str`: 长度 `B`

```python
253            next_obs, dones, valid_action, is_search = self.execute_predictions(
254                responses_str, self.tokenizer.pad_token, active_mask
255            )
```

解析当前 action 并执行环境步骤。

输出都是长度 `B` 的 list：

- `next_obs`: 下轮 observation 文本。
- `dones`: 1 表示该轨迹结束。
- `valid_action`: 当前 action 是否有效。
- `is_search`: 当前 action 是否 search。

`self.tokenizer.pad_token` 当前没有在函数内部使用。

```python
257            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
258            active_mask = active_mask * curr_active_mask
```

`curr_active_mask` 表示这轮之后仍未结束的样本。

`active_mask * curr_active_mask` 对 bool tensor 相当于逐元素 AND。输出 `[B]`。

```python
259            active_num_list.append(active_mask.sum().item())
260            turns_stats[curr_active_mask] += 1
261            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
262            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
```

更新统计。

注意 line 260 用的是 `curr_active_mask`，不是更新后的 `active_mask`。这会给“本轮后仍未 done”的样本加 turn。

```python
264            next_obs_ids = self._process_next_obs(next_obs)
```

把 observation 文本 tokenize。

输出 `[B, o]`，`o <= O`。

```python
267            rollings = self._update_rolling_state(
268                rollings,
269                responses_ids,
270                next_obs_ids
271            )
```

更新下一轮 LLM 输入上下文：

```text
旧 rolling input_ids + 当前 action + 当前 observation
```

输出 `rollings.batch` 中每个字段是 `[B, L_new]`，`L_new <= P`。

```python
272            original_right_side = self._update_right_side(
273                original_right_side,
274                responses_ids,
275                next_obs_ids
276            )
```

更新最终训练样本的 response 侧：

```text
历史 responses + 当前 action + 当前 observation
```

并在 `responses_with_info_mask` 中把 observation 部分替换为 pad。

输出：

- `responses`: `[B, T_new]`
- `responses_with_info_mask`: `[B, T_new]`
- `T_new <= P`

### 11.2 final LLM rollout，line 278-310

```python
278        # final LLM rollout
279        if active_mask.sum():
```

主循环结束后，如果还有样本没有输出 answer，则再做一次最终 LLM 生成。

```python
280            rollings.batch = self.tensor_fn.cut_to_effective_len(
281                rollings.batch,
282                keys=['input_ids', 'attention_mask', 'position_ids']
283            )
```

裁剪当前 rolling state。

```python
286            rollings_active = DataProto.from_dict({
287                k: v[active_mask] for k, v in rollings.batch.items()
288            })            
289            gen_output = self._generate_with_gpu_padding(rollings_active)
```

只对仍 active 的样本生成最终输出。

```python
291            meta_info = gen_output.meta_info            
292            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
293            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
```

同主循环：截断 action，并补回 `[B, r_final]`。

```python
296            _, dones, valid_action, is_search = self.execute_predictions(
297                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
298            )
```

最终轮调用 `execute_predictions(..., do_search=False)`。

含义：

- 如果模型输出 `<answer>`，done。
- 如果模型输出 `<search>`，不会真的请求检索服务，而是把 search result 当空字符串。
- 当前 final 轮不会再把 observation 拼回输出。

```python
300            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
301            active_mask = active_mask * curr_active_mask
302            active_num_list.append(active_mask.sum().item())
303            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
304            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
```

更新最终统计。

```python
307            original_right_side = self._update_right_side(
308                original_right_side,
309                responses_ids,
310            )
```

把最终 action 追加到 response 侧。这里没有 `next_obs_ids`，所以不会追加 observation。

### 11.3 组合输出，line 312-319

```python
312        meta_info['turns_stats'] = turns_stats.tolist()
313        meta_info['active_mask'] = active_mask.tolist()
314        meta_info['valid_action_stats'] = valid_action_stats.tolist()
315        meta_info['valid_search_stats'] = valid_search_stats.tolist()
```

把统计 tensor 转成 Python list 写入 `meta_info`。

`Tensor.tolist()` 会把 tensor 转成嵌套 Python list 或标量 list，方便日志和跨进程传输。

```python
317        print("ACTIVE_TRAJ_NUM:", active_num_list)
319        return self._compose_final_output(original_left_side, original_right_side, meta_info)
```

打印每轮 active 数，并组合最终 `DataProto`。

`run_llm_loop` 最终输出来自 `_compose_final_output`，见下一节。

## 12. `_compose_final_output`，line 321-351

```python
321    def _compose_final_output(self, left_side: Dict,
322                            right_side: Dict,
323                            meta_info: Dict) -> Tuple[Dict, Dict]:
324        """Compose final generation output."""
```

函数作用：把左侧原始 prompt 和右侧多轮 response 拼成训练用 `DataProto`。

输入：

| 参数 | 字段 | 维度 |
| --- | --- | --- |
| `left_side` | `input_ids` | `[B, S]` |
| `right_side` | `responses` | `[B, T]` |
| `right_side` | `responses_with_info_mask` | `[B, T]` |

```python
325        final_output = right_side.copy()
326        final_output['prompts'] = left_side['input_ids']
```

复制右侧字段，并新增 `prompts`。

当前：

- `final_output["responses"]`: `[B, T]`
- `final_output["responses_with_info_mask"]`: `[B, T]`
- `final_output["prompts"]`: `[B, S]`

```python
329        final_output['input_ids'] = torch.cat([
330            left_side['input_ids'],
331            right_side['responses']
332        ], dim=1)
```

拼成模型训练所需完整序列。

输出：

- `input_ids`: `[B, S+T]`

```python
335        final_output['attention_mask'] = torch.cat([
336            self.tensor_fn.create_attention_mask(left_side['input_ids']),
337            self.tensor_fn.create_attention_mask(final_output['responses'])
338        ], dim=1)
```

为完整序列创建 attention mask。

- 左侧 mask: `[B, S]`
- 右侧 mask: `[B, T]`
- 拼接后 `attention_mask`: `[B, S+T]`

```python
339        final_output['info_mask'] = torch.cat([
340            self.tensor_fn.create_attention_mask(left_side['input_ids']),
341            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
342        ], dim=1)
```

创建 `info_mask`。

关键点：

- 左侧 prompt 全按正常 attention mask。
- 右侧使用 `responses_with_info_mask`。
- 因为 observation 区域已经被替换成 pad token，所以对应位置 mask 为 0。
- `<search> query </search>` 属于模型生成的 `response`，没有被替换成 pad，所以 mask 为 1。

输出：

- `info_mask`: `[B, S+T]`

后续训练中 `RayPPOTrainer._create_loss_mask` 会取：

```python
loss_mask = batch.batch["info_mask"][:, -response_length:]
```

也就是只在 response 侧屏蔽 observation token 的 actor loss。

```python
344        final_output['position_ids'] = self.tensor_fn.create_position_ids(
345            final_output['attention_mask']
346        )
```

根据完整 `attention_mask` 生成 position ids。

输出：

- `position_ids`: `[B, S+T]`

```python
348        final_output = DataProto.from_dict(final_output)
349        final_output.meta_info.update(meta_info)
351        return final_output
```

构造并返回 `DataProto`。

最终 batch 字段：

| 字段 | 维度 | 说明 |
| --- | --- | --- |
| `prompts` | `[B, S]` | 初始 prompt |
| `responses` | `[B, T]` | 多轮 action + observation + final action |
| `responses_with_info_mask` | `[B, T]` | observation 位置为 pad |
| `input_ids` | `[B, S+T]` | prompt + responses |
| `attention_mask` | `[B, S+T]` | 正常有效 token mask |
| `info_mask` | `[B, S+T]` | observation 位置为 0 |
| `position_ids` | `[B, S+T]` | position ids |

## 13. `execute_predictions`，line 353-405

```python
353    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
```

函数作用：把模型文本输出当作环境 action 执行。它支持三种情况：

- `<answer>...</answer>`：结束轨迹。
- `<search>...</search>`：调用检索器，并返回 `<information>...</information>` observation。
- 其它格式：无效 action，返回一段提示模型修正格式的文本。

输入：

| 参数 | 维度/类型 | 说明 |
| --- | --- | --- |
| `predictions` | `List[str]`, 长度 `B` | 当前轮模型输出文本 |
| `pad_token` | `str` | 当前未使用 |
| `active_mask` | `BoolTensor[B]` | 哪些样本仍 active |
| `do_search` | bool | 是否真实调用检索器 |

```python
367        cur_actions, contents = self.postprocess_predictions(predictions)
368        next_obs, dones, valid_action, is_search = [], [], [], []
```

先解析 action 和内容。

- `cur_actions`: 长度 `B`，元素是 `"search"`、`"answer"` 或 `None`。
- `contents`: 长度 `B`，对应标签内部文本。

```python
370        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
```

收集所有 search query，长度为当前 batch 内 search action 数，记作 `q`。

```python
371        if do_search:
372            search_results = self.batch_search(search_queries)
373            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
374        else:
375            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
```

如果 `do_search=True`，调用检索器。

`batch_search` 输入 `List[str]`，输出 `List[str]`，长度为 `q`。

如果 `do_search=False`，为每个 search action 放一个空结果，避免 final rollout 继续引入 observation。

```python
377        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
```

逐样本处理。

```python
379            if not active:
380                next_obs.append('')
381                dones.append(1)
382                valid_action.append(0)
383                is_search.append(0)
```

非 active 样本：

- observation 为空。
- done 为 1。
- 不计有效 action/search。

```python
385                if action == 'answer':
386                    next_obs.append('')
387                    dones.append(1)
388                    valid_action.append(1)
389                    is_search.append(0)
```

answer action：

- 轨迹结束。
- observation 为空。
- 有效 action。

```python
390                elif action == 'search':
391                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
392                    dones.append(0)
393                    valid_action.append(1)
394                    is_search.append(1)
```

search action：

- 从 `search_results` 队头取一个结果。
- `.strip()` 去掉结果首尾空白。
- 包装成 `<information>...</information>` observation。
- 轨迹不结束。
- 计作有效 action 和 search。

```python
395                else:
396                    next_obs.append(f'\nMy previous action is invalid. \
397 If I want to search, I should put the query between <search> and </search>. \
398 If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
399                    dones.append(0)
400                    valid_action.append(0)
401                    is_search.append(0)
```

无效 action：

- 返回格式修正提示作为 observation。
- 轨迹不结束。
- 不计有效 action/search。

```python
403        assert len(search_results) == 0
405        return next_obs, dones, valid_action, is_search
```

确保所有检索结果都被消费完。

输出：

| 返回值 | 类型 | 长度 |
| --- | --- | --- |
| `next_obs` | `List[str]` | `B` |
| `dones` | `List[int]` | `B` |
| `valid_action` | `List[int]` | `B` |
| `is_search` | `List[int]` | `B` |

## 14. `postprocess_predictions`，line 407-436

```python
407    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
```

函数作用：用正则从模型输出文本中提取 action 类型和内容。

输入：

- `predictions`: `List[Any]`，实际要求每个元素都是 `str`。

```python
417        actions = []
418        contents = []
```

准备输出 list。

```python
420        for prediction in predictions:
421            if isinstance(prediction, str): # for llm output
```

逐条解析。非字符串会抛错。

```python
422                pattern = r'<(search|answer)>(.*?)</\1>'
423                match = re.search(pattern, prediction, re.DOTALL)
```

正则说明：

- `<(search|answer)>`：匹配开始标签，捕获 action 名，必须是 `search` 或 `answer`。
- `(.*?)`：非贪婪匹配标签内内容。
- `</\1>`：结束标签必须和第一个捕获组一致，即 `<search>` 对 `</search>`，`<answer>` 对 `</answer>`。
- `re.DOTALL`：让 `.` 可以匹配换行符。

`re.search` 只返回第一个匹配。

```python
424                if match:
425                    content = match.group(2).strip()
426                    action = match.group(1)
```

匹配成功：

- `match.group(1)` 是 `"search"` 或 `"answer"`。
- `match.group(2)` 是标签里的内容。
- `.strip()` 去首尾空白。

```python
427                else:
428                    content = ''
429                    action = None
```

匹配失败，视为无效 action。

```python
430            else:
431                raise ValueError(f"Invalid prediction type: {type(prediction)}")
```

非字符串输入直接报错。

```python
433            actions.append(action)
434            contents.append(content)
436        return actions, contents
```

输出：

- `actions`: 长度 `B`，元素为 `"search"`、`"answer"` 或 `None`。
- `contents`: 长度 `B`，每个 action 的内部文本。

## 15. `batch_search`，line 438-448

```python
438    def batch_search(self, queries: List[str] = None) -> str:
```

函数作用：批量调用检索接口，并把每条 query 的检索结果格式化成字符串。

输入：

- `queries`: `List[str]`，长度 `q`。

```python
446        results = self._batch_search(queries)['result']
```

调用 `_batch_search` 发 HTTP 请求。

预期返回 JSON 结构包含：

```python
{
    "result": [
        [doc_item_1, doc_item_2, ...],
        ...
    ]
}
```

`results` 长度为 `q`，每个元素是一条 query 的文档列表。

```python
448        return [self._passages2string(result) for result in results]
```

逐条 query 格式化文档列表。

输出：

- `List[str]`，长度 `q`。
- 每个字符串会被 `execute_predictions` 包到 `<information>...</information>` 中。

## 16. `_batch_search`，line 450-458

```python
450    def _batch_search(self, queries):
```

函数作用：发送 HTTP POST 到检索服务。

```python
452        payload = {
453            "queries": queries,
454            "topk": self.config.topk,
455            "return_scores": True
456        }
```

构造请求体：

- `queries`: query 列表。
- `topk`: 每个 query 返回多少文档。
- `return_scores`: 是否返回检索分数。

```python
458        return requests.post(self.config.search_url, json=payload).json()
```

`requests.post(url, json=payload)` 用法：

- 发 POST 请求。
- `json=payload` 会自动将 payload 序列化为 JSON，并设置合适的请求头。

`.json()`：

- 把 HTTP 响应体解析为 Python dict/list。

注意：当前实现没有 timeout、重试和错误处理。如果检索服务不可用，这里会直接抛异常或在 `.json()` 解析时报错。

## 17. `_passages2string`，line 460-469

```python
460    def _passages2string(self, retrieval_result):
461        format_reference = ''
```

函数作用：把一条 query 返回的文档列表格式化为模型可读的纯文本。

输入：

- `retrieval_result`: 文档列表，长度一般为 `topk`。

```python
462        for idx, doc_item in enumerate(retrieval_result):
```

遍历每篇文档。

`enumerate` 返回 `(idx, item)`，其中 `idx` 从 0 开始。

```python
464            content = doc_item['document']['contents']
465            title = content.split("\n")[0]
466            text = "\n".join(content.split("\n")[1:])
```

预期文档格式：

```python
doc_item = {
    "document": {
        "contents": "title\nbody text ..."
    }
}
```

处理方式：

- 第一行作为 `title`。
- 剩余行拼回正文 `text`。

```python
467            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
```

每篇文档格式化为：

```text
Doc 1(Title: xxx) yyy
```

```python
469        return format_reference
```

输出：

- 一个字符串，包含 topk 篇文档。

之后会被包装为：

```text
<information>
Doc 1(...)
Doc 2(...)
</information>
```

## 18. 关键数据流总览

### 18.1 入口输入

`RayPPOTrainer.fit()` 中进入本类前，一般已经有：

```text
gen_batch.batch["input_ids"]      [B, P]
gen_batch.batch["attention_mask"] [B, P]
gen_batch.batch["position_ids"]   [B, P]
initial_input_ids                 [B, S]
```

### 18.2 每轮循环

```text
rollings.batch
  input_ids/attention_mask/position_ids: [B, L]
        |
        | active_mask 过滤
        v
rollings_active: [b, L]
        |
        | actor_rollout_wg.generate_sequences
        v
gen_output.responses: [b, R]
        |
        | batch_decode + 截断到 </search> 或 </answer> + tokenize
        v
responses_ids: [b, r]
        |
        | _example_level_pad
        v
responses_ids: [B, r]
        |
        | execute_predictions
        v
next_obs: List[str] length B
        |
        | tokenizer + truncate
        v
next_obs_ids: [B, o]
        |
        | _update_rolling_state
        v
下一轮 rollings: [B, L_new], L_new <= P
        |
        | _update_right_side
        v
original_right_side.responses: [B, T_new], T_new <= P
original_right_side.responses_with_info_mask: [B, T_new]
```

### 18.3 最终输出

```text
left_side["input_ids"]             [B, S]
right_side["responses"]            [B, T]
right_side["responses_with_info_mask"] [B, T]
        |
        v
final_output:
  prompts        [B, S]
  responses      [B, T]
  input_ids      [B, S+T]
  attention_mask [B, S+T]
  info_mask      [B, S+T]
  position_ids   [B, S+T]
```

其中：

- `attention_mask`：真实 prompt + response 的有效 token mask。
- `info_mask`：检索返回的 observation 位置为 0，其余有效 token 为 1。
- search query 属于模型生成的 `response`，不会被 `_info_masked_concatenate_with_padding` 置 pad。

## 19. 每个函数职责一句话汇总

| 函数 | 职责 |
| --- | --- |
| `__init__` | 保存 tokenizer、rollout worker、配置，并创建 tensor helper |
| `_batch_tokenize` | 批量把字符串转成 padded token ids |
| `_postprocess_responses` | 把 vLLM 输出截断为一个 search/answer action，并重新 tokenize |
| `_process_next_obs` | 把 observation 文本 tokenize，并裁剪到 `max_obs_length` |
| `_update_rolling_state` | 更新下一轮送给 LLM 的上下文 |
| `_info_masked_concatenate_with_padding` | 拼接右侧序列，并在 masked 版本中把 observation 置 pad |
| `_update_right_side` | 更新最终训练样本的 response 侧和 masked response 侧 |
| `_generate_with_gpu_padding` | 为多 GPU 生成补齐 batch，生成后裁掉补齐样本 |
| `run_llm_loop` | 主多轮 agent 循环，串联 LLM、搜索环境和状态更新 |
| `_compose_final_output` | 组合最终 `DataProto`，生成 `input_ids/attention_mask/info_mask/position_ids` |
| `execute_predictions` | 解析并执行 action，返回 observation/done/action 统计 |
| `postprocess_predictions` | 用正则从模型输出里提取 search/answer 标签和内容 |
| `batch_search` | 批量检索并格式化每个 query 的结果 |
| `_batch_search` | 向检索 HTTP 服务发送 POST 请求 |
| `_passages2string` | 把检索文档列表转成模型可读字符串 |


# Ray 与 veRL 入门教程

> 适合对象：刚开始接触分布式训练、强化学习框架、大模型 RLHF/RLVR 的同学。  
> 核心目标：弄清楚 Ray、RLlib、veRL、vLLM、FSDP/Megatron 之间的关系，并知道后续该怎么学习源码。

---

## 0. 总体关系

可以先记住这张图：

```text
Ray
│
├── Ray Core：分布式任务调度系统
│   ├── task：无状态远程函数
│   ├── actor：有状态远程进程
│   ├── object store：分布式对象存储
│   └── placement group：GPU/CPU 资源组调度
│
├── RLlib：Ray 官方传统强化学习库
│   ├── PPO / DQN / SAC / A3C 等
│   └── 更偏游戏、机器人、控制、仿真环境
│
└── veRL：大模型 RLHF / RLVR 后训练框架
    ├── PPO / GRPO / DAPO 等
    ├── actor / rollout / critic / reward / ref model
    ├── 底层用 Ray 调度 worker
    ├── 训练可接 FSDP / Megatron
    └── 推理可接 vLLM / SGLang / TGI
```

一句话概括：

**Ray 负责把任务分布式跑起来；RLlib 负责传统 RL；veRL 负责大模型后训练里的 RLHF/RLVR。**

你研究 LLM、RLHF、GRPO、vLLM、FSDP 时，重点应该放在：

```text
Ray Core + veRL
```

RLlib 可以暂时了解，不必一开始就深入。

---

## 1. Ray 到底是什么？

Ray 可以理解成一个 **Python 分布式运行时**。

普通 Python 是这样：

```python
result = f(x)
```

函数 `f` 只在当前 Python 进程里执行。

Ray 想做的是：

```python
result_ref = f.remote(x)
result = ray.get(result_ref)
```

这个函数可能被 Ray 调度到：

```text
本机 CPU
本机 GPU
另一台机器的 CPU
另一台机器的 GPU
```

你写的仍然是 Python，Ray 帮你处理进程创建、资源分配、远程调用、结果传输。

Ray Core 里最重要的几个概念是：

```text
task
actor
object store
placement group
```

---

## 1.1 Ray Task：无状态并行函数

安装：

```bash
pip install "ray[default]"
```

`remote` 修饰在**函数**上，这个函数被 `.remote()` 调用时，就是 **Task**。最小例子：

```python
import ray
import time

ray.init()

@ray.remote
def slow_square(x):
    time.sleep(1)
    return x * x

refs = [slow_square.remote(i) for i in range(8)]

results = ray.get(refs)

print(results)
```

普通 Python 跑 8 次可能要 8 秒。Ray 会把这些任务并行调度，理想情况下接近 1 秒多一点。

可以把：

```python
@ray.remote
```

理解成告诉 Ray：

```text
这个函数可以丢给别的 worker 执行。
```

---

## 1.2 Ray Actor：有状态远程对象

如果一个任务需要保存状态，比如模型、计数器、缓存、环境实例，就不能只用 task。

这时用 actor。修饰在**类**上，这个类被 `.remote()` 实例化时，就是 **Actor**。

```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def add(self):
        self.value += 1
        return self.value

counter = Counter.remote()

print(ray.get(counter.add.remote()))
print(ray.get(counter.add.remote()))
print(ray.get(counter.add.remote()))
```

输出：

```text
1
2
3
```

Actor 很适合表示：

```text
一个模型 worker
一个 rollout worker
一个 reward model worker
一个环境模拟器
一个长期服务
```

veRL 这类框架会大量使用 Ray Actor / WorkerGroup，原因就在这里。

---

## 1.3 Ray 的资源调度

Ray 可以声明资源：

```python
@ray.remote(num_cpus=2, num_gpus=1)
def train_one_model(config):
    ...
```

意思是：

```text
这个任务需要 2 个 CPU + 1 张 GPU。
Ray 只有在资源满足时才会调度它。
```

如果你有 8 张卡，可以启动很多 worker：

```text
actor worker：占 4 张 GPU
critic worker：占 2 张 GPU
reward worker：占 1 张 GPU
ref worker：占 1 张 GPU
```

Ray 会负责把这些 worker 放到合适的机器和 GPU 上。

---

## 2. RLlib 是什么？它和 veRL 有什么区别？

Ray 里面有一个官方强化学习库叫 **RLlib**。

它更偏传统 RL 场景，比如：

```text
CartPole
Atari
MuJoCo
机器人控制
多智能体仿真
离线 RL
工业控制
```

可以把 RLlib 理解成：

```text
传统 RL 算法库 + 分布式采样 + 分布式训练
```

veRL 面向的是：

```text
大模型 RLHF/RLVR 后训练框架 + Ray 调度 + FSDP/Megatron/vLLM/SGLang 集成
```

两者区别如下：

| 框架 | 主要对象 | 典型任务 |
|---|---|---|
| Ray Core | 分布式任务 / 进程 / 资源 | 任何 Python 分布式任务 |
| RLlib | 传统强化学习 | 游戏、机器人、仿真、控制 |
| veRL | 大模型后训练 | PPO、GRPO、数学推理、RLHF、RLVR |

所以，如果你研究大模型后训练，主线应该是：

```text
Ray Core → veRL → vLLM/FSDP/Megatron
```

---

## 3. veRL 到底是什么？

veRL 全称常写作：

```text
Volcano Engine Reinforcement Learning for LLMs
```

它是一个大模型后训练框架，可以理解成一个“大模型 RLHF/RLVR 流水线管理器”。

它管理的内容包括：

```text
prompt 数据怎么读
actor 怎么生成回答
vLLM 怎么 rollout
reference model 怎么算 log_prob
reward model / rule reward 怎么给分
critic 怎么估 value
advantage 怎么算
actor 怎么更新
critic 怎么更新
checkpoint 怎么保存
多 GPU 怎么分配
不同 worker 怎么通信
```

veRL 处理的并不只是 PPO 公式，还包括大模型 RL 训练中的整套工程流程。

---

## 4. veRL 为什么需要 Ray？

大模型 RLHF/RLVR 的流程比普通监督微调复杂很多。

监督微调大概是：

```text
输入 prompt + answer
直接算 loss
反向传播
更新模型
```

PPO / GRPO 这类强化学习流程更像：

```text
prompt
↓
actor 生成 response
↓
计算 actor log_prob
↓
reference model 计算 ref log_prob
↓
reward model / rule reward 给奖励
↓
critic 估计 value
↓
计算 advantage
↓
更新 actor
↓
更新 critic
```

这里面每一步都可能是一个很重的模型计算。

veRL 的设计思想可以理解为：

```text
单进程 controller 管控制流
多进程 worker 管模型计算
```

Ray 的作用就是帮助 veRL 创建和调度这些 worker。

整体关系是：

```text
Ray 负责创建和调度 worker
veRL 负责定义 RLHF/RLVR 流程
FSDP/Megatron 负责训练并行
vLLM/SGLang 负责高吞吐生成
```

---

## 5. veRL 里的核心角色

以 PPO 为例，veRL 里面常见角色有下面几个。

### 5.1 Actor

Actor 就是正在训练的大模型。

它负责：

```text
根据 prompt 生成 response
计算当前策略下 token 的 log_prob
通过 PPO/GRPO 更新参数
```

在 RLHF/RLVR 里，actor 可以理解成 policy model。

---

### 5.2 Rollout Engine

Rollout engine 负责高效生成样本。

常见后端有：

```text
vLLM
SGLang
TGI
```

你之前看到的 vLLM，在这里主要用来提高 response 生成吞吐。

---

### 5.3 Reference Model

Reference model 一般是训练前的 SFT 模型。

它的作用是计算：

```text
ref_log_prob
```

这个量通常用于 KL 约束，防止 actor 偏离原模型太远。

直观理解：

```text
actor 可以为了拿高 reward 做调整，但不能乱跑到完全不像原模型。
```

---

### 5.4 Reward Model / Rule Reward

Reward 用来给 response 打分。

在数学推理任务里，reward 可能是规则型：

```text
答案对：1
答案错：0
格式错：惩罚
```

在 RLHF 里，reward 可能来自 reward model。

---

### 5.5 Critic

PPO 里通常需要 critic 估计 value。

它帮助计算 advantage：

```text
A_t = R_t - V(s_t)
```

其中：

```text
R_t：实际回报
V(s_t)：critic 预测的价值
A_t：这个动作比预期好多少
```

GRPO 这类方法可以不显式使用 critic，而是对同一个 prompt 采样多个 response，用组内相对分数做 baseline。

---

### 5.6 Controller / Trainer

Controller / Trainer 是总指挥。

它负责：

```text
读数据
初始化 WorkerGroup
调度 rollout
调度 log_prob 计算
调度 reward 计算
调度 critic 计算
调度 actor / critic 更新
保存 checkpoint
记录日志
```

在 veRL 里，你经常会看到类似：

```text
RayPPOTrainer
WorkerGroup
ResourcePool
```

这些名字都和“如何组织分布式训练流程”有关。

---

## 6. PPO 在 veRL 里的完整流程

一次 PPO 迭代可以理解成下面这样：

```text
1. 读取一批 prompts

2. Actor/Rollout 生成 responses

3. Actor 计算当前策略 log_probs

4. Reference model 计算 ref_log_probs

5. Reward function / reward model 给 reward

6. Critic 计算 values

7. 根据 reward、values、KL 计算 advantage 和 return

8. PPO 更新 actor

9. PPO 更新 critic

10. 记录日志、保存 checkpoint、做验证
```

更像代码的话：

```python
for batch in dataloader:

    # 1. rollout
    responses = actor_rollout.generate_sequences(batch.prompts)

    # 2. log prob
    old_log_probs = actor.compute_log_prob(responses)
    ref_log_probs = ref.compute_log_prob(responses)

    # 3. reward
    rewards = reward_model.compute_reward(responses)

    # 4. critic
    values = critic.compute_values(responses)

    # 5. advantage
    advantages = compute_advantages(
        rewards=rewards,
        values=values,
        ref_log_probs=ref_log_probs,
        old_log_probs=old_log_probs,
    )

    # 6. update
    actor.update_actor(responses, advantages)
    critic.update_critic(responses, rewards)
```

这只是帮助理解的伪代码。真实 veRL 代码会处理更多工程细节：

```text
多 GPU
多进程
micro batch
mini batch
sequence parallel
FSDP / Megatron
vLLM rollout
模型权重同步
KL 控制
checkpoint
日志
```

---

## 7. Ray 和 veRL 的分工

| 问题 | 负责组件 |
|---|---|
| 哪个 worker 用哪张 GPU | Ray |
| 多个 worker 怎么启动 | Ray |
| worker 之间怎么通信 | Ray + torch distributed |
| actor / critic / ref / reward 怎么组织 | veRL |
| PPO / GRPO 训练流程 | veRL |
| 模型参数怎么切分训练 | FSDP / Megatron |
| response 怎么高吞吐生成 | vLLM / SGLang |
| tokenizer / model config | Hugging Face |
| 实验配置 | Hydra / YAML / 命令行 override |

所以你看到 veRL 代码里有很多层：

```text
trainer
worker
worker_group
resource_pool
actor_rollout_ref
critic
reward_fn
ray.remote
```

它们大致对应：

```text
trainer：算法主循环
worker：真正执行模型计算的进程
worker_group：一组 worker，通常对应一个模型角色
resource_pool：这一组 worker 用哪些 GPU
actor_rollout_ref：actor、rollout、reference 的组合配置
critic：value model
reward_fn：奖励函数
ray.remote：把函数/类交给 Ray 远程执行
```

---

## 8. veRL 的安装注意事项

veRL 对环境要求比较新，安装时需要重点关注：

```text
Python 版本
CUDA 版本
PyTorch 版本
vLLM 版本
transformers 版本
flash-attn 版本
```

如果你的机器是：

```text
Ubuntu 20.04
CUDA 11.8
RTX 3090
```

直接装最新版 veRL 可能会遇到版本冲突。更稳的策略是：

```text
方案 A：使用官方或社区 Docker 镜像
方案 B：找 veRL 旧版本 + 对应 vLLM/PyTorch/CUDA 组合
方案 C：先只学 Ray Core，不急着跑 veRL
```

源码安装大概长这样：

```bash
git clone https://github.com/verl-project/verl.git
cd verl
pip install --no-deps -e .
```

如果使用 vLLM 后端，通常还会涉及类似：

```bash
pip install -e ".[vllm]"
```

如果使用 SGLang 后端，可能是：

```bash
pip install -e ".[sglang]"
```

具体版本一定要以当前官方文档和项目 README 为准。

---

## 9. 一个 veRL PPO 命令长什么样？

veRL 的训练命令一般会很长，因为它要配置数据、模型、rollout、critic、算法参数和资源。

示例结构如下：

```bash
python3 -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  critic.optim.lr=1e-5 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=15
```

这些配置大致对应：

```text
data.*：数据
actor_rollout_ref.*：actor / rollout / reference model
critic.*：critic model
algorithm.*：PPO / KL / advantage 等算法设置
trainer.*：训练资源、日志、保存频率、epoch
```

---

## 10. 推荐学习路线

### 阶段 1：学 Ray Core

目标：知道 Ray 怎么启动 task 和 actor。

需要掌握：

```text
ray.init()
@ray.remote
xxx.remote()
ray.get()
ray.put()
num_cpus / num_gpus
Actor
```

练习：

```text
用 Ray 并行跑 8 个函数
用 Ray Actor 保存一个计数器状态
用 num_gpus=1 模拟一个 GPU worker
```

---

### 阶段 2：理解传统 RL 和大模型 RL 的差异

传统 RL 是：

```text
agent 在 environment 里 action
environment 返回 observation / reward
agent 更新 policy
```

LLM RLHF/RLVR 是：

```text
prompt 相当于状态
response 相当于动作序列
reward 来自规则 / reward model / verifier
policy 是大语言模型
```

传统 RL 里的动作可能是：

```text
向左 / 向右
```

大模型 RL 里的动作是：

```text
一个 token
一个 token
一个 token
...
```

所以 LLM RL 本质上是：

```text
超长离散动作空间里的序列决策问题
```

---

### 阶段 3：学 PPO / GRPO 的数据流

先搞清楚这些变量：

```text
prompt
response
log_prob
ref_log_prob
reward
value
advantage
loss
```

PPO 的核心是限制新旧策略变化太大：

```text
ratio = exp(new_log_prob - old_log_prob)
loss = min(ratio * advantage, clipped_ratio * advantage)
```

再加一个 KL 约束：

```text
KL(actor || reference)
```

GRPO 的核心是：

```text
同一个 prompt 采样多个 response
用组内相对分数做 advantage
减少对 critic 的依赖
```

---

### 阶段 4：读 veRL 的 main_ppo

读源码时建议按这个顺序：

```text
verl/trainer/main_ppo.py
↓
RayPPOTrainer
↓
数据加载 RLHFDataset
↓
WorkerGroup 初始化
↓
fit() 训练循环
↓
actor_rollout.generate_sequences
↓
compute_log_prob / compute_values / compute_reward
↓
update_actor / update_critic
```

不要一开始就钻 FSDP、Megatron、vLLM 权重同步。先把主流程摸清楚，再看底层实现。

---

### 阶段 5：理解 veRL + vLLM + FSDP 的参数同步

在 LLM RL 里，actor 有两个身份：

```text
训练时：FSDP / Megatron 管它
生成时：vLLM / SGLang 管它
```

所以每轮更新后，需要让 rollout engine 拿到新的 actor 参数。

你之前看到的这些内容就在这里出现：

```text
actor 参数同步到 vLLM
FSDP shard 转成 vLLM TP shard
rollout.generate_sequence
```

背后的原因是：

```text
训练系统和推理系统对参数的切分方式不同。
```

---

## 11. 学完后应该能看懂的流程

学完 Ray + veRL 后，你应该能看懂这种流程：

```python
batch = actor_rollout_wg.generate_sequences(batch)

batch = actor_rollout_wg.compute_log_prob(batch)

ref_log_prob = ref_wg.compute_ref_log_prob(batch)

values = critic_wg.compute_values(batch)

rewards = reward_fn(batch)

advantages = compute_advantage(batch)

actor_wg.update_actor(batch)

critic_wg.update_critic(batch)
```

你也应该知道：

```text
actor_rollout_wg 是一组 Ray worker
critic_wg 是另一组 Ray worker
ref_wg 可能和 actor 共用资源，也可能单独放
reward_fn 可能是本地函数，也可能是远程模型
Ray 负责调度
veRL 负责训练逻辑
vLLM 负责生成
FSDP/Megatron 负责训练并行
```

---

## 12. 最小理解版总结

### Ray：分布式调度员

它管：

```text
谁在哪张卡上跑
谁先跑谁后跑
远程函数怎么调用
远程 worker 怎么创建
结果怎么拿回来
```

### RLlib：Ray 自带的传统 RL 算法库

它适合：

```text
PPO 玩游戏
机器人控制
Gym 环境
多智能体仿真
```

### veRL：大模型 RLHF/RLVR 训练框架

它适合：

```text
LLM PPO
LLM GRPO
数学推理 RL
代码 RL
偏好对齐
规则奖励训练
```

### vLLM / SGLang：生成加速器

它管：

```text
高吞吐 rollout
快速生成 response
```

### FSDP / Megatron：训练并行系统

它管：

```text
参数怎么切
梯度怎么同步
显存怎么省
大模型怎么训
```

你现在最该掌握的主线是：

```text
Ray Core → veRL PPO 数据流 → WorkerGroup → actor/rollout/ref/critic/reward → 参数同步
```

这样再去看 veRL 源码，就不会觉得它是一团乱麻了。

---

## 13. 后续阅读建议

建议按下面顺序阅读：

```text
Ray Core 官方文档
↓
Ray Actor / Task / Placement Group 示例
↓
veRL README
↓
veRL trainer/main_ppo.py
↓
RayPPOTrainer
↓
WorkerGroup / ResourcePool
↓
actor_rollout_ref worker
↓
vLLM rollout 和参数同步相关代码
```

如果只是为了科研或工程复现，优先读：

```text
main_ppo.py
RayPPOTrainer.fit()
actor_rollout.generate_sequences()
update_actor()
update_critic()
```

这些地方会直接决定你是否能理解 veRL 的训练主线。


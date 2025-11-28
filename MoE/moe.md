# MOE架构

## 什么是 MoE？
MoE 全称 **Mixture of Experts（专家混合模型）**，是一种近年来在大规模模型中非常重要的架构，用来**提升模型能力、减少计算量**，并且能够在保持参数量巨大的情况下，让推理和训练成本保持可控。

## 核心思想
MoE 就像一个“大型专家团队”一起做决策，但**每次并不是所有专家都出动**。模型内部包含许多“专家”，通常是一些前馈网络。当输入一个 token 时，模型的一个专门模块（称为 **Router 或 Gating Network**）会判断：哪几个专家最适合处理这个 token？于是只有被选中的专家会被激活，其余专家不参与计算。这叫 **稀疏激活（Sparse Activation）**。

## 架构组件
- Experts: 多个独立FFN
- Router/Gating Network: 负责选择专家
  gate会输出一个概率分布，选出top-k个专家
- Sparse Routing: 稀疏路由
  对每个token
  - 计算它对每个专家对分数
  - 取top-k个专家
  - 只让这k个专家参与计算
  - 根据分数加权合并输出

## 负载均衡
在 MoE 中，每个输入 token 会被一个 gating network（门控网络）根据概率分配给若干个专家（Expert）。如果不做限制，gating network 很容易出现这样的情况：
- 某一个专家非常“受欢迎”，大量 token 都被分给它；
- 其他专家几乎没有工作量，甚至完全闲置。

这种不均衡会导致以下问题：
- 专家过载（overflow）：某些专家超量处理，导致训练不稳定甚至 OOM。
- 计算资源浪费：训练时大部分专家几乎闲置，无法发挥模型规模优势。
- 梯度学习不足：未被使用的专家无法得到梯度更新，模型能力下降。

因此，负载均衡的目标是让每个专家尽可能平均地分到 token 进行训练与推理，既避免过载也避免闲置。

为了解决负载均衡问题，常用的方法有：
- 损失函数中的负载均衡正则项（Load Balancing Loss）

这是最经典的做法，例如 Google Switch Transformer、GShard、DeepSpeed MoE 都采用这种方法。

它的核心思想是：

让门控网络不仅关注 token 与专家的适配度，还要考虑专家的使用比例是否均衡。

例如 GShard/Switch Transformer 中加入：

L_aux = n_experts × Σ(expert_load × expert_importance)

其中：
  - expert_load：某个专家接收的 token 比例
  - expert_importance：门控概率分配给该专家的平均值

最终的训练损失变成：

Loss_total = Loss_main + λ × L_aux

这个正则项会促使 gating network 自动学习更均衡的 token 分配方式。


- 专家容量限制（Capacity Factor）

每个专家能处理的最大 token 数称为 capacity。

MoE 会强制给每个专家设定：

capacity = ceil( token_batch / num_experts × capacity_factor )

如果某个专家的 token 超出 capacity：
  - Switch Transformer 直接丢弃多余 token（不会参与该层计算）。
  - GShard/Megatron MoE 会做 padding 或 routing 到下一个专家。

这样可避免专家超载，保证每个专家 workload 接近一致。


- Top-k Gating 保持专家数量可控

MoE 常用的 gating：
  - Top-1（Switch Transformer）
  - Top-2（GShard）

Top-k gating 会让 token 只分配给少数几个专家，而不是所有专家，从而简化分配过程并保持均衡。

- Softmax 门控概率平滑（Temperature / Entropy 正则）

调节 softmax 温度，或加入 entropy loss，避免门控网络过度集中：
  - 温度降低 → 更平均
  - entropy 增加 → 门控分布更平滑

这样 gating 不会把所有 token 都分给同一个专家。


- 工程调度策略（高性能训练框架）

例如 Megatron、DeepSpeed、Tutel 会在实际分布式训练中使用：
  - 专家并行（expert parallelism）
  - 全局 shuffle（token rebalancing）
  - 动态 batch 并行（动态路由批处理）


## 优缺点
- 优点
  - 参数量大但不增加计算量
  - 性能提升明显
  - 专家结构可扩展（增加专家不影响其他部分）

- 缺点
  - 路由通信代价高
  - 负载均衡难
  - 实现复杂（调优难度高）
  - 某些任务不如 dense 模型稳定


# References
- [awesome-moe](https://github.com/XueFuzhao/awesome-mixture-of-experts)
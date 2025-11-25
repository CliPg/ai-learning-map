# MOE架构

MoE 全称 **Mixture of Experts（专家混合模型）**，是一种近年来在大规模模型中非常重要的架构，用来**提升模型能力、减少计算量**，并且能够在保持参数量巨大的情况下，让推理和训练成本保持可控。

**MoE 的核心思想**：MoE 就像一个“大型专家团队”一起做决策，但**每次并不是所有专家都出动**。模型内部包含许多“专家”，通常是一些前馈网络。当输入一个 token 时，模型的一个专门模块（称为 **Router 或 Gating Network**）会判断：哪几个专家最适合处理这个 token？于是只有被选中的专家会被激活，其余专家不参与计算。这叫 **稀疏激活（Sparse Activation）**。

# References
- [awesome-moe](https://github.com/XueFuzhao/awesome-mixture-of-experts)
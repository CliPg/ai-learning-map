## Norm

### LayerNorm（层归一化）

LayerNorm 是 Transformer 里最经典的归一化方法，它对每一个样本的每一个 token 的所有隐藏维度做归一化。

1. 公式（重点）

对向量 $x = [x_1, x_2, ..., x_d]$：
$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i

\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 + \epsilon}

\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$
其中：
- $\mu$：均值
- $\sigma$：标准差
- $\gamma, \beta$：可学习参数


2. 关键特征
- 同时使用均值与方差进行归一化
- 会减去均值（centered）
- 是 Transformer 的默认归一化方式（例如 BERT、GPT-2）。


## RMSNorm（均方根归一化）

RMSNorm 是 2019 年提出的新型归一化方法，在 LLaMA、Mixtral、DeepSeek 中被大量使用。

1. 公式（重点）

它与 LayerNorm 最大的区别：

- 不减均值
- 只用元素平方的均值（RMS）进行缩放

对向量 x：
$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}

\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma
$$


2. 关键特征
- 不减均值（不中心化）
- 只用 L2 范数的均值来缩放
- 参数比 LayerNorm 更少（没有 β）
- 更简单，更快，更稳定


### RMSNorm vs LayerNorm：核心区别


1. 数学上最大的区别

归一化方法	是否减均值	是否用标准差	参数
LayerNorm	✔ 减均值	✔ 用标准差	γ、β
RMSNorm	❌ 不减均值	❌ 不用标准差，只用 RMS	只有 γ



2. 为什么 RMSNorm 更稳定？

因为：
- LayerNorm 的方差包含 (x - mean) 这个操作，会放大小数、削弱大数
- 而 RMSNorm 直接用值的平方，整体更稳定、更平滑


3. 计算开销对比
	•	LayerNorm：需要算均值、方差 → 更慢
	•	RMSNorm：只算平方 → 更快


4. 实际效果对比
	•	小模型：LayerNorm 稳定性更好
	•	大模型（>1B）：RMSNorm 效果更好、效率更高


### 直观理解

把一个向量看成一个学生的成绩：
- LayerNorm：
先看平均分，再看偏差，把成绩移动到均值附近再缩放。
- RMSNorm：
不看平均分，只看整体成绩的“能量大小”，然后缩放。

这就是为什么 RMSNorm 更简单但也更稳定。




### 总结

LayerNorm = (x − 均值) / 标准差
RMSNorm = x / RMS，不减均值，计算更简单、训练更稳定


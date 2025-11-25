# References
- [动手学深度学习-李沐](https://www.bilibili.com/video/BV1f54ZzGEMC)
- [cs224n-pytorch-tutorial](https://colab.research.google.com/drive/1Pz8b_h-W9zIBk1p2e6v-YFYThG1NkYeS?usp=sharing#scrollTo=Avy1fnyAvEcd)


# Basic

## 普通运算

加减乘除按位计算


## 广播机制

两个张量即使形状不同，也可以通过广播机制执行按元素操作。但是两个张量的某一维既不为1，又不相同，就无法广播计算。
从右往左看，每一维必须满足：一样或某一方为 1 

eg.
```py
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
a + b
```
a会广播成tensor([[0, 0],[1, 1],[2, 2]])
b会广播成tensor([[0, 1],[0, 1],[0, 1]])


## id
```py
before = id(a)
a = a + b
id(a) == before
```
在第二行代码中，a+b会得到一个新的对象，a被绑定到这个新的对象。before = 原来 a 的 id，id(a) = 新对象的 id

```py
z = torch.arange(24).reshape((2,3,4))
print(z)
z[:,1,:]
```
结果tensor([[ 4,  5,  6,  7],[16, 17, 18, 19]])，y只有一个维度。因为 1 是 整数索引（integer index），而 PyTorch 遇到整数索引时会：在该维度上移除（squeeze）这个维度。

```py
z = torch.zeros_like(a)
z[:] = a + b
```
这样可以执行原地操作。


## 线性代数

### 标量
```py
x = torch.tensor([3.0])
```

### 向量
```py
x = torch.arange(4)

x[3] # 通过索引访问
len(x) # 张量长度
x.shape # 张量形状
```

### 矩阵

#### **创建矩阵**
```py
A = torch.arange(20).reshape(5,4)
A
```

#### **矩阵转置**
```py
A.T
```

#### **哈达玛积**
矩阵按元素相乘
```py
A*B
```

#### **计算所有元素和**
sum永远是个标量
```
x.sum
```

#### **按某一个轴求和**
```
A_sum_axis0 = A.sum(axis=0)
```
按哪个维度求和，就消掉哪个维度

均值也是同理。

若要保持维度不变
```
sum_A = A.sum(axis=0, keepdims=True)
```

#### 点积
```
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4)
x, y, torch.dot(x, y)
```

#### 矩阵向量积
Ax是一个长度为m的列向量，其ith元素的点积是a_i^Tx
```
torch.mv(A,x)
```


#### L2范数

向量元素平方和的平方根
```
u = torch.tensor([3.0,-4.0])
torch.norm(u)
```


#### L1范数

向量元素绝对值之和
```py
torch.norm(u,1)
```


#### 矩阵求导
当f是标量，x是向量
$$
x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \frac {\partial f}{\partial x} = \begin{bmatrix} \frac{\partial f}{\partial x_1} , \frac{\partial f}{\partial x_2} , \frac{\partial f}{\partial x_3} \end{bmatrix}
$$

eg.
$$
f = x_1^2 + 2x_2^2 + 3x_3^2
$$

$$
\frac {\partial f}{\partial x} = \begin{bmatrix} 2x_1 , 4x_2 , 6x_3 \end{bmatrix}
$$

当f是向量，x是标量，则对f求导得到一个列向量。

当x，f都是向量，求导结果是一个矩阵。
$$
\frac {\partial f}{\partial x}  = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & ... & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & ... & \frac{\partial f_2}{\partial x_n} \\ ... & ... & ... & ... \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & ... & \frac{\partial f_m}{\partial x_n} \end{bmatrix}
$$

## 自动求导
```
x = torch.arange(4.0, requires_grad=True)

if x.grad is not None:
    x.grad.zero_()

y = x.sum()

y.backward()
```

x.grad的结果是tensor([1., 1., 1., 1.]),
因为y = x[0] + x[1] + x[2] + x[3]，因此y对x对每个分量求导的结果都是1。
如果不加x.grad.zero_()，则x.grad会累加之前的梯度。


## 线形回归

小批量随机梯度下降
- 在整个训练集上算梯度太贵
- 我们可以随机采样b个样本来近似损失,b表示批量大小
  - 批量太小，不适合并行来最大利用计算资源
  - 批量太大，内存消耗增加，浪费计算


## softmax

解决分类问题，将概率归一化

```
$$
softmax(\mathbf{z})_{ij} = \frac{e^{z_{ij}}}{\sum_{k=1}^{q}e^{z_{ik}}}
$$
```

## 感知机

$$
o = \sigma(\mathbf{w}^T\mathbf{x} + b) 
$$

$$
\sigma(x) = \begin{cases} 1, x \geq 0 \\ 0, otherwise \end{cases}
$$

这是一个二分类。

**训练感知机**
```
initialize w = 0 and b = 0
repeat
  # y_i 是真实值， w^Tx_i + b 是预测值，如果相乘为负，说明预测错误
  if  y_i[w^Tx_i + b] <= 0 
    w = w + x_i y_i
    b = b + y_i
until converged 
```

**xor问题**

感知机不能拟合xor函数，只能产生线性分割面


## 多层感知机

e.g.

1、4一类，2、3一类

-------
|   |  1 | 2 | 3 | 4 |
|---|----|---|---|---|
|   | +  | - | + | - |
|   | +  | + | - | - |
|product| +  | - | - | + |

**单隐藏层**
```py
h = σ(XW1 + b1)
o = hW2 + b2
```
这里的激活函数需要非线形的，如果为线形，则相当于单层感知机，无法拟合xor问题。

**Sigmoid激活函数**
$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$
将输入投影到(0,1)

**Tanh激活函数**
$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
将输入投影到(-1,1)

**ReLU激活函数**
$$
ReLU(x) = max(0,x)
$$

### 多类分类
$$
h = σ(XW_1 + b_1) \\
o = hW_2 + b_2 \\
y = softmax(o)
$$

多层感知机的本质就是构建多个隐藏层。
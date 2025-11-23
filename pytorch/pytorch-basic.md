# References
- [动手学深度学习-李沐](https://www.bilibili.com/video/BV1f54ZzGEMC)


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

标量
```py
x = torch.tensor([3.0])
```

向量
```py
x = torch.arange(4)

x[3] #通过索引访问
```
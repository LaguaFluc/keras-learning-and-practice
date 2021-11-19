# 2021.11.15总结

Done：

- [ ] 程序正确运行，跑完200个epochs
- [ ] 写完了关于数据输出的修正代码（需要后序的验证
- [ ] 发现了数据中存在的规律，模型大同小异，现在的建模需要往**数据处理**上走
- [ ] 发现：1可以尝试使用故障检测的方法来对数据进行预测，2尝试在进行故障检测之后对故障发生时刻进行预测



TODO:

- [ ] 验证，使用后序的修正代码
- [ ] 阅读3个kaggle中使用LSTM预测的代码
- [ ] 找到3个PHM期刊中的LSTM预测数据的，数据处理方法
- [ ] 对数据可视化之后的特征进行处理：考虑进行特征组合，特征筛选来处理特征
- [ ] 不使variancethreshold来筛选低方差变量，因为阈值设置差异

# 2021.11.16 总结

过拟合：[深度学习知识点总结四（防止过拟合的方法） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/97326991)

寿命预测比赛top分享：[科大讯飞2019机械核心部件寿命预测，亚军方案_datayx的文章-CSDN博客](https://blog.csdn.net/demm868/article/details/103171227)



遇到的问题：

1. Accuracy不高，0.0038，20类，1/20 = 0.05比猜还差
2. 特征都是连续型变量，且呈现出大致的指数分布，应该怎么进行特征工程处理数据
3. 数据效果不好，猜测可能是数据的问题，关于如何使用模型处理数据，有待考究



针对1：

1. 模型没有拟合，没有收敛，发现模型中有dropout层，这层本来是对于模型过拟合会产生一定的缓解作用，但是在这里没有拟合的情况下，就不需要对模型进行过多的dropout操作。
2. 删除了一个LSTM层，增加了一个Dense层，将模型简单话，因为对于这个数据不需要过于复杂的模型



## 阅读笔记

### [「AI不惑境」学习率和batchsize如何影响模型的性能？_收敛 (sohu.com)](https://www.sohu.com/a/333348924_120233360)

**1 为什么说学习率和batchsize**
$$
w_{t+1} = w_t - \eta \cfrac{1}{n} \sum_{x \in \mathcal{B} } \triangledown l(x, w_t)
$$
n: batch_size, $\eta$是学习率

lr: 影响模型收敛状态，

batch_size: 影响模型泛化性能



**2 学习率如何影响模型性能？**

学习率满足条件：
$$
\begin{aligned}
\sum_{i=0}^{\infty} \epsilon_i = \infty \\
\sum_{i=1}^{\infty}\epsilon_i^2 < \infty
\end{aligned}
$$
第一个：不管初始状态距离最优状态多元，总是可以收敛，

第二个：约束学习速率随着训练进行有效地降低，保证收敛稳定性（事实上，各个自适应的学习算法本质上就是再不断调整各个时刻地学习率。

学习率：决定权重迭代地步长

对模型性能地影响体现在两个方面：

- 初始学习率的大小
- 学习率的变换方案

**2.1 初始学习率对模型性能的影响**

不同学习率大对模型loss的影响

不考虑差异，如何确定最佳的初始学习率？

简单搜索发：从小到达开始训练模型，记录损失的变化，得到曲线

经验：0.1, 0.01

学习率增加，模型达到欠拟合或者过拟合状态，带那个数据集上会更加明显

**2.2 学习率变换策略对模型性能的影响**

lr很少有不变的，

两种更改方式：

- 设规则lr变化
- 自适应学习率变换

**2.2.1 预设规则学习率变化法**

常用： fixed, step, exp, inv, multistep, poly, sigmoid

做实验得到结论：step, multistep方法的收敛效果最好

其次，而exp, poly

最次，inv, fixed

新方法：cyclical learning rate

设置上下界，让学习了在其中变化

优点：在模型迭代后期更有利于克服因为lr不够而无法跳出鞍点的情况

确定上下界的方法： LR range test,使用不同的学习率得到精度曲线，获得精度升高和下降的两个拐点，作为上下界

SGDR: cuclical learning rate 变化更加平缓的周期性变化方法

**2.2.2 自适应学习率变化法**

Adagrad, Adam

现象：原理上，改进的自适应学习率算法都比SGD算法更有利于性能的提升；

实际上：精细调优过的SGD算法可以取得更好的结果

**2.3 小结**

不考虑其它因素，学习率的大小和迭代方法本身是一个非常敏感的参数；

如果经验不足，考虑从adam系列的默认参数开始；经验丰富，尝试更多的试验配置



**3 Batchsize如何影响模型性能**

先言：对batchsize没有lr那么敏感，进一步提升，需要调节batchsize

**3.1 大的batchsize减少训练时间，提高稳定性**

一样epoch，大的batchsize需要的batch数目减少，可以减少训练时间，目前多篇论文在1h内训练完ImageNet数据集，另一方面，大的batch size梯度的计算更加稳定，因为模型训练曲线会更加平滑，在微调的时候，大的batchsize 可能会取得更好的结果

**3.2 大的batchsize泛化能力下降**

**在一定范围内**，增加batch size有卒于收敛的稳定性，但是随着batchsize的增加，模型的性能会下降



研究[6] 表明大的batchsize 收敛到sharp minimum， 小的batchsize 收敛到flat minimum, 

结论： 后者泛化能力更强

区别：变化的趋势，一个快，一个慢

原因：（主要）小的batchsize带来的噪声有助于逃离sharp minimum

hoffer[7] 表明，大的batchsize性能下降时因为时间不够长，本质上并不是batchsize的问题，同样epochs, 参数更新变少，需要更长的迭代次数

**3.3 小结**

batchsize在变得很多时，会降低模型的泛化能力，再次之下，模型的性能变换随batchsize通常没有lr敏感

**4 学习率和batchsize的关系**

增加batchsize为原来的N倍，

保证更新后权重相等：线性缩放，lr = lr * N

保证更新后权重方差相等：lr = lr * sqrt(N)

前者用的多

常见的调整策略来看，lr 与 batchsize同时增加

lr: 非常敏感，不能太大，否则模型不收敛

batchsize: 也有影响

研究[8]： 衰减学习率可以通过增加batchszie来实现类似效果，从SGD的权重更新式子可以看出两者时等价的，文中有验证

研究[9]：对于fixed lr, 存在最优的batchsize 能够最大化测试精度，精度与batchsize和lr以及训练集的大小正相关

两建议：

- if 增加lr, batch size 最好也跟着增加，这样收敛更加稳定
- 尽量使用大的lr, 因为很多研究表明更大的lr有利于提高泛化能力。如果真的要衰减，尝试其它方法：eg增加batchsize, **lr 对模型的收敛影响真的很大，慎重调整**

# 2021.11.18 记录

今天白天第一次训练，epochs=250, lr=0.01, 得到的最好结果为0.7左右，且

![image-20211118162337711](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211118162337711.png)

有问题

1. loss始终不收敛，一会增大，一会儿变小
2. 精度大部分时候很小，少部分时候很大
3. 测试集的loss也随着训练集一样，变化趋势差不多的增大，变小

模型上的问题

1. 如何选择学习率
2. 进行数据处理的方法（如何有数据处理的直觉）



最后做了什么？

1. 尝试了不同的学习率

   lr=0.1, epochs =300, loss波动幅度比较大，总是先上升再急剧下降，accuracy也不大，最多在0.2附近

   lr=0.001，epochs=300,  loss平缓下降，accuracy缓慢上升，总体loss上升和下降的趋势稳定，但是变动速度实在是太慢；且accuracy取值在0.2左右，差不多在80个epochs左右才开始变到0.01

   lr=0.05， epochs=189，loss下降稳定，速度也保持的可以，总体loss和accuracy呈现的趋势正常，即loss稳定下降，类似于指数曲线；accuracy稳定上升，不过类似于线性曲线，accuracy最后最大值在0.6下面一点

还想尝试的地方：

1. 改变optimizer=RMSP, 设置初始学习率
2. 尝试step-wise的lr递减方式
3. 尝试添加batch_normalization层，减少batch，层之间的变换

还需学习的新知识：

1. Adam, RMSP 原理
2. Batch Normalization原理



# 2021.11.19

## GD, SGD 学习

[为什么说随机最速下降法 (SGD) 是一个很好的方法？ | 雷锋网 (leiphone.com)](https://www.leiphone.com/category/yanxishe/c7nM342MTsWgau9f.html)

GD为什么要到SGD?
$$
x_{t+1} = x_t - \eta_t g_t
$$


- GD，梯度算的精确，慢

- GD容易陷入鞍点（局部最小点），SGD可以帮助跳出鞍点，
  $$
  x_{t+1} = x_t - \eta_t g_t, E[g_t] = \triangledown f(x_t)
  $$

- SGD算出来的导数是大概的，GD算出来的导数是精确的，相当于说SGD的导数是加了噪声的



鞍点的表达：

考虑导数为0的点，成为Stationary points, 稳定点。可以是局部最小值，局部最大值，也可以是鞍点。

判断：使用Hessian矩阵

- H福鼎，说明所有的特征值都是负的。这个时候，无论往什么方向走，导数都会变负，函数值会下降。所以，这是局部最大值
- H正定，说明所有特征值整的。这个时候，无论往什么方向走，导数都会变正，函数之会上升。所以，这是局部最小值。
- H即包含正的特征值，又包含负的，这个稳定点是鞍点；某些方向函数值会上升，某些会下降
- H可能包含特征值为0；无法判断稳定点属于哪一类，需参照更高维的导数；

只考虑，前三种，第四种被称为退化的情况，考虑非退化的



非退化，考虑struct saddle

特点：对每个点x

- 要么x的导数比较大
- 要么x的Hessian矩阵包含一个负的特征值
- 要么x已经离某一个局部最小值很近



两篇论文：

- 在鞍点加上扰动，能顺着负的特征值方向滑下去
- 跑若干步GD, 再跑一步SGD（导数比较小，很长时间没有跑SGD)，



## 炼丹实验室

深度学习网络训练技巧汇总 - 萧瑟的文章 - 知乎 https://zhuanlan.zhihu.com/p/20767428

参数初始化

一定要搞参数的初始化，Xavier, He

- uniform
- normal
- svd

数据预处理方式

- zero-center

  X -= np.mean(X, axis=0) # zero-center

  X /= np.std(X, axis=0) # normalize

- PCA witening

训练技巧

- 梯度归一化

- clip c（梯度裁剪）：限制最大梯度

- dropout 对小数据防止过拟合有很好效果，一般设置为0.5

  小数据，dropout + SGD，效果提升明显

  dropout输入位置：建议放到输入->RNN与RNN->输出的位置. 还有一片参考论文讲述如何配置dropout的位置

- adam, adadelta等，在小数据上，效果不如sgd, sgd慢，但是收敛后的效果一般都比较好。



作者：乔卡
链接：https://www.zhihu.com/question/274788355/answer/1071886489
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```python
tbCallBack = TensorBoard(
    log_dir='./woca_logs',  # log 目录                         
    histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算            
    batch_size=32,     # 用多大量的数据计算直方图                         
    write_graph=True,  # 是否存储网络结构图                         
    write_grads=True,  # 是否可视化梯度直方图                         
    write_images=True,  # 是否可视化参数                         
    embeddings_freq=0,                         
    embeddings_layer_names=None,                         
    embeddings_metadata=None
)
```



### batch_size

[谈谈深度学习中的 Batch_Size_听雨322的博客-CSDN博客_深度学习中的batch](https://blog.csdn.net/qq_22080019/article/details/81357245)

为什么要使用？

首先决定的是下降的方向。

如果数据集比较小，完全可以采用全数据集的形式

好处：

- 全数据集确定的方向能够更好的代表样本总体 
- 由于不同权重的梯度差别巨大，选择一个全局的学习率很困难
- full batch learning 可以使用Rprop只基于梯度符号并且针对性单独胡更新各权值

大数据集

好处 -> 坏处

- 数据海量增长 +  内存限制，一次性载入不可行

- Rprop的方式迭代，会由于各个batch之间的采样差异性，歌词梯度修正值相互抵消，无法修正。于是衍生出RMSprop的妥协方案

  Rprop 弹性反向传播

  均根反向传播：RMSprop

- 

### batch_size Gradient Descent

[批量梯度下降(BGD)、随机梯度下降(SGD)以及小批量梯度下降(MBGD)的理解 - LLLiuye - 博客园 (cnblogs.com)](https://www.cnblogs.com/lliuye/p/9451903.html)

> 注：损失和梯度更新一起计算，用多少数据计算损失，就用多少数据计算梯度，并且进行更新

batch gradient descent: 使用全量数据进行迭代计算梯度，找到最小值，可能需要迭代10次，那么计算量为10 * 30w

stochastic gradient descent: 一次参数更新只使用一个样本，若使用30w各样本进行参数更新，参数会被更新（迭代）30w次。这期间，SGD能保证收敛到一个合适的最小值。在收敛是，BGS计算10 * 30w次，而SGD计算1 * 30w次。

mini batch gradient descent: 每次使用一个atch可以大大减小收敛所需要的迭代次数，同时可以使收敛到的结果更加接近梯度下降的效果（batch_size = 100, 迭代3000次，院校书SGD 30w次）

### optimizer中的动量

[深度学习各类优化器详解（动量、NAG、adam、Adagrad、adadelta、RMSprop、adaMax、Nadam、AMSGrad）_恩泽君的博客-CSDN博客_动量优化器](https://blog.csdn.net/qq_42109740/article/details/105401197)

动量：

针对不同维度的梯度大小差异（有的维度梯度变化很大，另一个维度变化比较小），这样会使得维度梯度在梯度大的维度上震荡，另外一个维度更新缓慢，添加动量，来使得各个梯度的更新变得更加均匀。

是这样做的：

一阶动量的更新：
$$
m_t = \beta m_{t-1} + (1 - \beta) \cfrac{\partial(Loss)}{\partial \theta_t}
$$
二阶动量的更新：对于动量$V=1$
$$
\theta_{t+1} = \theta_t - \eta m_t
$$

> 这篇文章讲的太好啦


# 2021.11.15总结

Done：

- [x] 程序正确运行，跑完200个epochs
- [x] 写完了关于数据输出的修正代码（需要后序的验证
- [x] 发现了数据中存在的规律，模型大同小异，现在的建模需要往**数据处理**上走
- [ ] 发现：1可以尝试使用故障检测的方法来对数据进行预测，2尝试在进行故障检测之后对故障发生时刻进行预测



TODO:

- [x] 验证，使用后序的修正代码
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

![image-20211118162337711](image/image-20211118162337711.png)

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





# 2021.11.22

[深度学习: 参数初始化 - 云+社区 - 腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1347911?from=article.detail.1508512)

[深度学习策略选择 | 优化器选择，权重初始化，损失函数选择 - 简书 (jianshu.com)](https://www.jianshu.com/p/dcd35af75b8e)

# 2021.11.29


2021.11.29.RMSprop(0.01),loss_weight:0.95, 0.05, epochs=300,batch_size=64
<img src="./image/image-20211129153728357.png" alt="0.95, 0.05" style="zoom:70%;" />

<img src="./image/image-20211129153741960.png" alt="more 100 epochs" style="zoom:70%;" />

## 0.95, 0.05, RMSProp(0.03), 200, 64

<img src="./image/image-20211129180405031.png" alt="3.1" style="zoom:70%;" />

### 200(50)

<img src="./image/image-20211129180552833.png" alt="3.1" style="zoom:70%;" />

<img src="./image/image-20211129180606682.png" alt="3.1" style="zoom:70%;" />

### 200(50(50))

<img src="./image/image-20211129180639866.png" alt="3.1" style="zoom:70%;" />

<img src="./image/image-20211129180654862.png" alt="3.1" style="zoom:70%;" />

# 2021.11.30

<img src="./image/image-20211130111910413.png" alt="3.1" style="zoom:70%;" />

<img src="./image/image-20211130111924142.png" alt="3.1" style="zoom:70%;" />

## 2021.12.6

RMSProp(0.001), 200, 64, 0.95, 0.05

<img src="./image/image-20211206105303635.png" alt="3.1" style="zoom:70%;" />

<img src="image/image-20211206105318678.png" alt="image-20211206105318678" style="zoom:70%;" />


## 再跑100epochs
<img src="./image/image-20211206113113300.png" alt="3.1" style="zoom:70%;" />

<img src="./image/image-20211206113126295.png" alt="3.1" style="zoom:70%;" />



## RMSprop 0.001， 450epochs

<img src="./image/image-20211206162247871.png" alt="3.1" style="zoom:70%;" />

<img src="./image/image-20211206162301076.png" alt="3.1" style="zoom:70%;" />

# 2021.12.7

## RMSprop(0.005), 0.95:0.05, 400epochs, 

![image-20211207113634947](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207113634947.png)

![image-20211207113648971](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207113648971.png)

## 50-400epochs

![image-20211207113706219](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207113706219.png)

![image-20211207113856688](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207113856688.png)



## RMSprop(0.003), epochs=400, 0.95:0.05

![image-20211207165748604](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207165748604.png)

![image-20211207165801243](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207165801243.png)

### error(train) distribution

![image-20211207165821267](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207165821267.png)

![image-20211207171026712](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207171026712.png)

### the curve of test data

![image-20211207182033009](C:\Users\laguarange\AppData\Roaming\Typora\typora-user-images\image-20211207182033009.png)

## 简单写写今天学习到了什么

1 首先是rnn的基本结构，权值共享，多个rnn

2 再次是关于数据的处理，对输出的处理

3 学习了一下矩阵求导的知识，按行求导，按列求导等等，标量对向量求导、向量对向量求导、函数对矩阵求导、向量对矩阵求导。之前觉得一直要记忆的东西，竟然可以通过理解原理来习得，还是感觉听神奇的。

4 搞好了误差的分布，分别查看了训练集和测试集的



# 2021.12.8

## RMSprop(0.005), 0.95: 0.05, epochs=400

到400个epochs，能够达到0.63的正确率



## 在看的文章

[零基础入门深度学习(5) - 循环神经网络 - 作业部落 Cmd Markdown 编辑阅读器 (zybuluo.com)](https://zybuluo.com/hanbingtao/note/541458)



参考的文章

[雅可比矩阵_百度百科 (baidu.com)](https://baike.baidu.com/item/雅可比矩阵/10753754?fr=aladdin)

## 学习的知识

1 雅可比矩阵

2 RNN BPTT理解，矩阵求导

3 colab跑代码操作

## 关于一些尝试

- [x] 1 跑一遍修正后的数据
- [ ] 2 测试得到的最优thr_1, thr_2
  - [x] 测试过10， 20
  - [x] 待测试修正后的相对误差0.15, 0.3
  - [ ] 再测试两组阈值参数，例如0.1， 0.2
- [ ] 3 keras 日志管理
  - [ ] 未查找网页
- [x] 4 RNN BPTT
  - [x] 简单了解步骤
  - [ ] 跟写代码
- [ ] 5 LSTM basic
  - [x] 知道LSTM是从结构上解决RNN梯度消失问题，通过增加三个门（忘记，留存，输出）
  - [ ] 了解LSTM的应用
  - [ ] LSTM优缺点



## 关于thr_1 如何设置

有三种办法

1 直接加上训练过后的误差

2 加上误差的平均数

3 加上一个自设定的数（固定的），当作是超参数进行调整

最后：

(a) 将误差从原来的简单相对，变成
$$
error = \cfrac{y_{pred} - y_{true}}{y_{pred}}
$$
(1) 当$thr_1 < error < thr_2$时，

当$error > 0$，则使用$y_{true} = (1 + e) * y_{true}$​进行更正，否则不操作

(2) 当$error > thr_2$​， 作删除操作



# 2021.12.9

Done:

- [x] 切分验证集、测试集代码
- [x] 处理测试集（test）的输出
- [x] 跑了一次数据修正，发现比原来的结果好（loss平滑，到200个epochs精度高）
- [x] 再看一遍BPTT（RNN）
- [x] 看了作业部落--感知机

TODO：

- [ ] 看作业部落--线性单元
- [ ] 看作业部落--激活函数和梯度下降

# 2021.12.10

最后，我先给自己一些忠告

1 多寄希望于自己，自己给自己成功，自己给自己失败，凡事都从自身做起。

2 慢慢来，去搞清楚事物发展的规律。你不可能要求自己将别人写了近一个星期的文章，在一个半天的时间里完全搞懂。学习知识的规律在这里：曲线生长，你不可能一下子就能够直冲上天。如果有一天你真的做到了这一点，恐怕你此刻已经在云霄之巅了。

3 多写作，多运动，如果不能跟别人有深度交流的话，不妨试着「精神分裂」一下，尝试着去跟自己深度对话，或者换个角度来了解自己。只有自己学会爱自己，学会疏导自己的情绪，才能够去疏导其它人的情绪。

4 事情不能够看一个人怎么说，而是要看一个人怎么做。这点，对于自己也适用。

5 尝试给未来一个月的自己写信，尝试着给自己一些期待，给一些现在的教训。

6 真的没有那么难，找到问题的所在，或许需要尝试5遍6遍，这在工程实践中都是非常正常的。

7 尝试看书，当发觉自己进入到一种思想的胡同时，最好的方法是看书，发现胡同中的隐藏小路，那时便会有豁然开朗的感觉。当然，这个过程也是非常缓慢的。

8 可以发脾气，但是一定要有所控制，刻意练习，并不是达不到。

9 每个周一，尝试给过去写个总结（3句），未来三句话。

10 脑子不是自动随着时间的流逝而变灵活的，而是说随着自己慢慢地进行思考地练习，从而变得更加灵活的。



# 2021.12.16

- [x] 尝试跑对比

  - [x] 尝试跑400个没有对原始训练数据进行处理的模型

    得到结果：

    1 没有处理，确实效果会差一点，最后只能到0.61

    处理机器

    2 根据训练集的y来处理训练数据：0.6487

    3 对训练集的特征分布来处理训练集中异常特征：0.6550

    仅仅根据原始结果处理训练数据

    4 单纯对原始数据的训练集按照预测的输出，对训练集的y进行修正，得到结果最差：0.59

- [x] 邱锡鹏《神经网路与深度学习》

  - [x] 看chapt4， 主要讲述的是神经网络的入门，感知机，ANN, Activations, loss, 机器学习的分类等等知识
  - [x] 看chap5 1/2， 了解什么是卷积（信号处理），CNN中的卷积跟信号处理中的卷积有什么区别

- [x] 查看别人的实践

  - [x] 看了2个kaggle比赛输出处理和模型，主要是进行特征处理，看都尝试了很多的方法，
    - 没有学习到很多调参知识
    - 了解了特征生成，可以根据历史事实

- [x] 自己实践

  - [x] 写好了Perceptron的代码框架
    - 自上而下，先从同一层开始思考，再慢慢向下写函数



# 2021.12.20

- [x] 大致看了5个kaggle上的数据处理，以及回归的流程。

  分为数据描述、数据处理、特征工程、建模、得到结论。

  只看见一篇文章讲述循环数据处理、建模的流程，其它的都是kernel吗？感觉都好简单

- [x] 做了三篇笔记(English)



今日疑惑：

我们学习了这么多的方法，数据处理的方法那么多，我们真的能够找到那个最优的方法，去达到最优的结果吗？

还有一个概率统计上的疑问：p值小于0.05原假设真的可以被拒绝吗？

还看见了一个LR在测试集上也能达到0.99的预测准确率，真的这么强吗？我也没看着有对数据进行很复杂的特征处理鸭。

To be explored.

# 2021.12.22(Wrapper Feature Engineering)

- [x] 写好了特征工程筛选（使用随机森林，fit生成特征重要性），查看模型的结果是否会发生改变

- [x] 了解pandas处理组内数据，学会map函数（用在group之后的数据），apply函数的使用（用在series）

  [超好用的 pandas 之 groupby - 简书 (jianshu.com)](https://www.jianshu.com/p/42f1d2909bb6)



今天所做的事情分为两部步，

首先需要处理输出，可以输入道RandomForestClassifier中去，这里主要是处理训练数据，得到RUL的列。

> 这里新新学习了pandas的groupyby操作，还了解了一下agg的操作，
>
> [Group by: split-apply-combine — pandas 1.3.5 documentation (pydata.org)](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)
>
> 总是要复习的pandas的列合并
>
> [pandas实现两个dataframe数据的合并：按行和按列_美丽心灵的博客-CSDN博客_dataframe按列合并](https://blog.csdn.net/weixin_44208569/article/details/89676843)

第二，放到RandomForest模型中去，得到比较大的特征排名（大于0.01）

得到的特征分别是 var_ + [2,3,4,6,8,9,10,12,13,14,15,16]

去除的特征分别是 var_ + [0,1,5,7,11]

第三，得到上述的特征，

>(a)准备训练数据
>
>(b)进行标准化处理
>
>(c)去除无用特征
>
>(d)prepare_data()
>
>(e)处理test的输入输出
>
>(f)fit模型

# 2021.12.23(Feature engineering Test)

方法参考：[scikit-learn中的特征选择方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/141506312)



[tensorflow2知识总结（杂）---3、如何提高网络的拟合能力 - 范仁义 - 博客园 (cnblogs.com)](https://www.cnblogs.com/Renyi-Fan/p/13388877.html)

[loss问题汇总（不收敛、震荡、nan） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/420053831)

- [x] 跑了四次的模型
  - [x] 只使用RandomForest跑，小于0.01的去除
  - [x] 使用RandomForest之后的特征，等于0的去除
  - [x] 使用`sklearn.feature_selection.f_classif`得到12个特征（`get_support`获得特征名）
  - [x] 使用12个特征，并且增加神经元的个数（50-》128）



# 2021.12.24（想要尝试的

- [x] 尝试跑出600个epochs（原来不处理特征的）
- [x] 对比处理特征之后，跑600个epochs
- [x] 尝试找其它的方法来提高模型的准确率
  - [x] 尝试减少训练数据集合

关于训练集只选择一部分：

[训练模型时，损失下降到某个值附近之后便不再下降，远没有达到理想的情况。请问后续应该如何调整参数？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/437040334/answer/1651769469)



|                            | train_acc | train_mae | valid_acc | valid_mae | test_acc | test_mae | epochs                                            |
| -------------------------- | --------- | --------- | --------- | --------- | -------- | -------- | ------------------------------------------------- |
| FE-去除小于0.001           | 0.5454    | 2.4200    | 0.5134    | 27.4259   | 0.4795   | 22.7204  | 400                                               |
| FE-去除等于0               | 0.5897    | 2.0852    | 0.5354    | 27.7301   | 0.5338   | 23.3396  | 400                                               |
| 神经元-50->128             | 0.6522    | 2.3165    | 0.6136    | 26.6359   | 0.5586   | 18.0637  | 600                                               |
| 10000条数据                | 0.6944    | 1.3867    | 0.6863    | 15.8345   | 0.6129   | 15.2838  | 600                                               |
| 10000条（第一次）          | 0.7013    | 1.3792    | 0.6645    | 12.1412   |          |          | 400+200（后面200开始牵几个epoch是用全部数据训练的 |
| 15000                      | 0.6888    | 1.8886    | 0.7140    | 38.3320   | 0.6129   | 15.5615  | 600                                               |
| batch_size = 128           | 0.6202    | 1.6491    | 0.5952    | 22.4944   | 0.5571   | 17.6558  | 600                                               |
| FE-retry( importance <= 0) | 0.1398    | 28.6129   | 0.1389    | 20.9569   | 0.0924   | 20.4364  | 600                                               |
| 后13000                    | 0.6477    | 1.6275    | 0.5924    | 33.7079   | 0.5529   | 21.9910  | 600                                               |
| 去除68,91,95机器（时间     | 0.6561    | 1.7960    | 0.5584    | 20.9602   | 0.5471   | 14.8190  | 600                                               |
| 去除8,32,77机器（特征      | 0.6576    | 2.5846    | 0.6351    | 26.4505   | 0.5933   | 20.5571  | 600                                               |

Note: 

```cmd
Epoch 491/600 316/316 [==============================] - 16s 51ms/step - loss: 0.8765 - output_1_loss: 0.3608 - output_2_loss: 10.6744 - output_1_accuracy: 0.5748 - output_2_mean_absolute_error: 2.1558 - val_loss: 278.9924 - val_output_1_loss: 21.2340 - val_output_2_loss: 5176.4019 - val_output_1_accuracy: 0.5348 - val_output_2_mean_absolute_error: 42.1584 Epoch 492/600
```





好的文章：如何成为一名合格的深度学习工程师？ - xingxing的回答 - 知乎 https://www.zhihu.com/question/302304559/answer/645365600

# 2021.12.27

TODO:

- [x] 跟导师交流上周的成果，记录下这周要尝试的
- [ ] 看完chap5，有个基本的了解，不要求全部看懂
- [ ] 两个Leetcode，
- [ ] 写下Leetcode刷题计划（未来要复习的，和要新刷的，设置deadline，并且设置要求程度）



尝试的方向

- [x] FE-重新进行特征工程，去除掉特征重要性为0的特征
- [x] data-try latter half data, such as 13000
- [ ] learning rate, optimizers, update strategy. 
  - [ ] Try Adam, 
  - [ ] learning_rate decay
- [x] batch_size 128, not better than data processing.

tomorrow TODO

- [ ] dataset, check whether exist new

  [hustcxl/Rotating-machine-fault-data-set: Open rotating mechanical fault datasets (开源旋转机械故障数据集整理) (github.com)](https://github.com/hustcxl/Rotating-machine-fault-data-set)

- [ ] others Code, check evaluate

- [ ] learning_rate decay, keras, try

  - [ ] interpret "step" in keras LearningRateDecay class

    [Keras learning rate schedules and decay - PyImageSearch](https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/)

- [ ] Check Code, process y.

  [RUL-Net/data_processing.py at master · LahiruJayasinghe/RUL-Net (github.com)](https://github.com/LahiruJayasinghe/RUL-Net/blob/master/data_processing.py)

  处理y，用分段函数来表述剩余寿命

# 2021.12.28

- [ ] dataset, check whether exist new

  [hustcxl/Rotating-machine-fault-data-set: Open rotating mechanical fault datasets (开源旋转机械故障数据集整理) (github.com)](https://github.com/hustcxl/Rotating-machine-fault-data-set)

  [CMAPSS的个人理解和CMAPSS、PHM08、09、12下载地址_分类保的博客-CSDN博客_cmapss](https://blog.csdn.net/qq_37117980/article/details/95343811)

- [ ] others Code, check evaluate

- [ ] learning_rate decay, keras, try

  - [x] interpret "step" in keras LearningRateDecay class

    [Keras learning rate schedules and decay - PyImageSearch](https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/)
    
  - [ ] pratise it with my own juypter notebook.

- [ ] Check Code, process y.

  [RUL-Net/data_processing.py at master · LahiruJayasinghe/RUL-Net (github.com)](https://github.com/LahiruJayasinghe/RUL-Net/blob/master/data_processing.py)

  处理y，用分段函数来表述剩余寿命



# 2021.12.29

今天开始正式查看最后搜寻的资料

查看别人的数据处理，别人的评价方式，有的是使用

- 简单得MAE，bias，也就是预测值和真实值的平均



# 2021.12.30

今天主要还是去了解数据、加上数据处理、数据y值(RUL)的处理、模型的评价

> 1. 数据如何处理，关注特征
> 2. y值（RUL）如何处理
> 3. 如何评价模型

不搞好这些，就要继续看，不能够中途放弃任何一个想要学习的项目。

[Ali-Alhamaly/Turbofan_usefull_life_prediction: given run to failure measurements of various sensors on a sample of similar jet engines, estimate the remaining useful life (RUL) of a new jet engine that has measurements of the same sensor for a period of time equal to its current operational time. (github.com)](https://github.com/Ali-Alhamaly/Turbofan_usefull_life_prediction)



Data Exploration

1 sn_9, sn_14 indicates that the trnd depends on the specific engine. 

2 other columns show an apparent trend as the fault prpagate throughout the engine cycles

linear trending

**1、如何数据处理**

---

3 optional settings, altitude(0-42k ft.), mach number (0 - 0.84), and throttle resolver angle(TRA)(20-100)

21 sensors

contaminated with sensor noise.

**problem**: **fused** these sensors into a condition indicator or a health index that help in identifying the occurrence of a failure

**method:** compare how similar the testing fused signal to the raining fused signal.

correlation, displot(discrete variable),

linear trending,

Dimensionality reduction PCA

PCA -> PCA

21 -> 6 -> 3

summary of data exploration and dimensionality reduction

1. sensors that do not change with time are dropped since they do not offer any information toward the end of life.
2. sensors, not have apparent trend are droped
3. linear regression, 6 sensors kept, 
4. further, take 3 principle components for the data

**Fusing Sensors:**

Health Index (HI) sensor, 

RUL_high = 300, RUL_low = 5

how to fuse HI, linear and logistic model

HI: noisy, processed it with Savitzky-Golay(Sav_gol) filter 

2、y值如何处理

---

> 使用HI作为health indicator 来表明剩余寿命，表明当前的寿命指数

**Fitting the Model:**
$$
y = a [\exp(b * t) - 1] = \mathrm{HI}
$$
steps:

1. removing low variance sensors
2. normalize x using StandardScaler
3. find linear trend with each xi
4. from xnorm, find a subset of r sensors(r highest aboslute linear slope), xslope
5. perform PCA on xslope to reduce dimensionally to space of n columns. xpca


$$
f(x)=\left\{
\begin{aligned}
&1, y >= 1 \\
&y , otherwise\\
&0  ,y<=0
\end{aligned}
\right.
$$

$$
y_{fused} = \theta^T \mathrm{x}_{pca} + \theta_0
$$

# 2021.12.31

1. 处理数据
2. 去除几个不变变量，取6个sensors作为主成分的输入变量
3. 拿到排名前90%的主成分作为最后的三个变量
4. 输入这三个变量，到一个模型中去融合
5. 得到一个变量，这个变量作为HI（使用指数函数来拟合的）
6. 取HI变量作为一个变量，与RUL对应起来

关于模型的建立，如何输入数据到模型中，如何进行预测，得到我们想要的RUL值

这个键盘是我用过最好的键盘，这个键盘的弹力实在是太强了，虽然说我没用过别的键盘，但是现在这个巧克力键盘是当前我用到的键盘中感受最好的。

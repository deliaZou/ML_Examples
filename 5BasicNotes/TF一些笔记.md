#### TF一些笔记

##### TF安装

https://www.cnblogs.com/lvsling/p/8672404.html

1，conda安装tensorflow虚拟环境

2，在tensor虚拟环境中安装（切换国内镜像源）

【待添加】

##### 张量

- 通常定义张量的物理学或传统数学方法，是把张量看成一个多维数组，当变换坐标或变换基底时，其分量会按照一定规则进行变换，这些规则有两种：即协变或逆变转换。
- 通常现代数学中的方法，是把张量定义成某个矢量空间或其对偶空间上的多重线性映射，这矢量空间在需要引入基底之前不固定任何坐标系统。例如协变矢量，可以描述为 1-形式，或者作为逆变矢量的对偶空间的元素。

通俗：如果现在重新描述向量和矩阵，就可以是：一阶张量为向量，二阶张量为矩阵。当然，零阶张量也就是标量，而更重要的是 N 阶张量，也就是 N 维数组。

##### TF中的张量定义

在 TensorFlow 中，每一个 Tensor 都具备两个基础属性：数据类型（默认：float32）和形状

##### TF张量的数据类型

|  Tensor 类型   |         描述          |
| :------------: | :-------------------: |
|  `tf.float32`  |      32 位浮点数      |
|  `tf.float64`  |      64 位浮点数      |
|   `tf.int64`   |    64 位有符号整型    |
|   `tf.int32`   |    32 位有符号整型    |
|   `tf.int16`   |    16 位有符号整型    |
|   `tf.int8`    |    8 位有符号整型     |
|   `tf.uint8`   |    8 位无符号整型     |
|  `tf.string`   |  可变长度的字节数组   |
|   `tf.bool`    |        布尔型         |
| `tf.complex64` | 实数和虚数Tensor 类型 |

##### TF张量的形状、阶数和维数

|        形状        |  阶  | 维数 |                示例                |
| :----------------: | :--: | :--: | :--------------------------------: |
|         []         |  0   | 0-D  |          0 维张量。标量。          |
|        [D0]        |  1   | 1-D  |      形状为 [5] 的 1 维张量。      |
|      [D0, D1]      |  2   | 2-D  |    形状为 [3, 4] 的 2 维张量。     |
|    [D0, D1, D2]    |  3   | 3-D  |   形状为 [1, 4, 3] 的 3 维张量。   |
| [D0, D1, ... Dn-1] |  n   | n-D  | 形状为 [D0, D1, ... Dn-1] 的张量。 |

##### TF的示例

`x = tf.Variable([1., 2., 3., 4.])  # 形状为 4 的一维变量`

`sess = tf.Session()  # 建立会话`
`sess.run(x.initializer)  # 初始化变量`
`sess.run(x)  # 得到变量的值`

##### TF张量的大致4个类型

- `tf.Variable` ：变量 Tensor，需要指定初始值，常用于定义可变参数，例如神经网络的权重。
- `tf.constant` ：常量 Tensor，需要指定初始值，定义不变化的张量。
- `tf.placeholder` ：占位 Tensor，不必指定初始值，可在运行时传入数值。
- `tf.SparseTensor` ：稀疏 Tensor，不常用。

##### TF的访问

`sess = tf.Session()  # 建立会话`
`sess.run(x.initializer)  # 初始化变量 #常量不需要初始化`
`sess.run(x)  # 得到变量的值`

##### 计算图（Data Flow Graphs）

`init = tf.global_variables_initializer()  # 初始化全部变量`

`with tf.Session() as sess:  # 使用 with 语句建立会话无需手动关闭`
    `init.run()  # 执行变量初始化`
    `result = sess.run(f)  # 改为 sess.run 计算`

##### 占位符

`import numpy as np`

`x = tf.placeholder(tf.float32, shape=(3, 3))  # 创建占位符张量`
`y = tf.matmul(x, x)  # 乘法计算`

`with tf.Session() as sess:`

​	`x_test = np.random.rand(3, 3)  # 生成 numpy 数组作为示例传入数据`
​    `result = sess.run(y, feed_dict={x: x_test})  # 通过 feed_dict 把测试数据传给占位符张量`

##### TF一些设计初衷

为了构建和训练由图构建的模型，Python 程序首先构建一个表示计算的图，然后调用 Session.run 来发送该图，以便在基于 C++ 的运行时上执行。这种方式具有以下优势：

- 使用静态 autodiff 进行自动微分。
- 可轻松地部署到独立于平台的服务器。
- 基于图的优化（常见的子表达式消除、常量折叠等）。
- 编译和内核融合。
- 自动分发和复制（在分布式系统上放置节点）。

##### Eager Execution 特性

TensorFlow 后续版本新增了 Eager Execution 特性来帮助初学者以及简化代码调试。Eager Execution  是一种命令式编程环境，可像 NumPy 那样立即评估操作，而无需构建图。操作会返回具体的值，而不是构建以后再运行的计算图。

##### TF构建神经网络

参见jupyter notebook

##### TF.nn下的激活函数参考

```
- `tf.nn.relu`
- `tf.nn.relu6`
- `tf.nn.crelu`
- `tf.nn.elu`
- `tf.nn.selu`
- `tf.nn.softplus`
- `tf.nn.softsign`
- `tf.nn.dropout`
- `tf.nn.bias_add`
- `tf.nn.sigmoid`
- `tf.nn.tanh`
#卷积神经网络下的卷积层
- `tf.nn.convolution`
- `tf.nn.conv2d`
- `tf.nn.depthwise_conv2d`
- `tf.nn.depthwise_conv2d_native`
- `tf.nn.separable_conv2d`
- `tf.nn.atrous_conv2d`
- `tf.nn.atrous_conv2d_transpose`
- `tf.nn.conv2d_transpose`
- `tf.nn.conv1d`
- `tf.nn.conv3d`
- `tf.nn.conv3d_transpose`
- `tf.nn.conv2d_backprop_filter`
- `tf.nn.conv2d_backprop_input`
- `tf.nn.conv3d_backprop_filter_v2`
- `tf.nn.depthwise_conv2d_native_backprop_filter`
- `tf.nn.depthwise_conv2d_native_backprop_input`
```

##### 优化器

在 NumPy  实现神经网络的更新中：手动计算梯度，然后让参数验证梯度的反方向以一定比率（学习率）进行更新，也就是之前所说的梯度下降法。实际上，梯度下降完成的就是优化迭代过程。手动计算梯度 + 权重更新非常麻烦，深度学习通过高阶 API 封装优化器可以自动完成参数优化，也就是优化器。

优化器一般放在 `tf.train` 模块中（eg：tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

设置好优化器后就可以进行神经网络训练了（参考jupyter notebook）

##### Adam优化器（更常用）

tf.train.AdamOptimizer(0.01).minimize(loss)

##### TF和Numpy构建NN

    NumPy 构建神经网络：定义数据 → 前向传播 → 反向传播 → 更新权重 → 迭代优化。
    TensorFlow 构建神经网络：定义张量 → 前向传播计算图 → 定义损失函数 → 定义优化器 → 迭代优化。
TensorFlow 省掉了推导反向传播更新参数的过程，若使用 Keras 等更高阶 API ，会更简单。

##### 深度神经网络常用损失函数

交叉熵损失函数本质上就是对数损失函数。交叉熵主要用于度量两个概率分布间的差异性信息，交叉熵损失函数会随着正确类别的概率不断降低，返回的损失值越来越大。交叉熵损失函数公式如下：
$$
H_{y^{\prime}}(y)=-\sum_{i} y_{i}^{\prime} \log \left(y_{i}\right)
$$
#yi' 是预测的概率分布，而 yi 是实际的概率分布，即独热编码处理后的标签矩阵。

Softmax 函数（通过该函数转换为概率）：
$$
\operatorname{softmax}(x)_{i}=\frac{\exp \left(x_{i}\right)}{\sum_{j} \exp \left(x_{j}\right)}
$$
√ 使用 Softmax 函数对全连接层输出进行概率处理，并最终计算交叉熵损失

TensorFlow 中给出了交叉熵损失函数 + Softmax 函数二合一 API：`tf.nn.softmax_cross_entropy_with_logits_v2`

##### Mini Batch

将整个数据分成一些小批次放进模型里进行训练。

scikit-learn ： [ *K 折交叉验证*](https://zh.wikipedia.org/wiki/交叉驗證)  sklearn.model_selection.KFold
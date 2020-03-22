## 线性回归

##### 一次函数定义公式

$$
y(x,w)=w_0+w_1*x
$$

##### 平方损失函数公式

$$
\sum_{i=1}^{n}(y_i−(w_0+w_1*x_i))^2
$$

##### 最小二乘法代数解（推导略）

$$
w_1=\frac{(n∑x_iy_i−(∑x_i∑y_i))}{n∑x_i^2−(∑xi)^2}
$$

$$
w_0=\frac{∑x_i^2∑y_i−∑x_i∑x_iy_i}{n∑x_i^2−(∑xi)^2}
$$

##### 最小二乘法矩阵解（推导略）

$$
W=(X^TX)^{-1})X^Ty
$$

### 多项式回归

##### 一元高阶多项式定义

$$
y(x,w)=w_0+w_1x+w_2x^2+…+w_mx^m=\sum_{j=0}^{m}w_jx^j
$$

##### 损失函数

$$
\sum_{i=1}^{n}y_i−y'_i
$$

### 岭回归

##### 线性回归的损失函数   (在简单版本上加上了 L2 正则项（2-范数） )

$$
F_{Ridge}=\sum_{i=1}^{n}(y_i−w^Tx)^2+λ\sum_{i=1}^{n}(w_i)^2
$$

##### 损失函数最优解（#λ:正则化强度，上同）

$$
w_{Ridge}=(X^TX+λI)^{−1}X^TY
$$

### Lasso回归

##### 损失函数 （(在简单版本上加上了 L1 正则项（1-范数） )

$$
F_{LASSO}=\sum_{i=1}^{n}(y_i−w^Tx)^2+λ\sum_{i=1}^{n}|w_i|
$$

### 模型评估检验

##### MAE（绝对误差平均值）

$$
\operatorname{MAE}(y, \hat{y})=\frac{1}{n} \sum_{i=1}^{n}\left|y_{i}-\hat{y}_{i}\right|
$$

##### MSE（ 误差的平方的期望值 ）

$$
\textrm{MSE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^{2}
$$

##### MAPE （ 预测结果较真实结果平均偏离 ）

$$
\textrm{MAPE}(y, \hat{y} ) = \frac{\sum_{i=1}^{n}{|\frac{y_{i}-\hat y_{i}}{y_{i}}|}}{n} \times 100
$$



##### 拟合优度检验

实际值vs实际值平均值vs预测线上的值
$$
TSS=\sum_{i=1}^{n} Y_{i}^{2}=\sum_{i=1}^{n}\left(y_{i}-\overline{y}\right)^{2}
$$

$$
E S S=\sum_{i=1}^{n} \hat{Y}_{i}^{2}=\sum_{i=1}^{n}\left(\hat{y}_{i}-\overline{y}\right)^{2}
$$

$$
RSS=\sum_{i=1}^{n} e_{i}^{2}=\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}
$$

以上三式，可得
$$
TSS=ESS+RSS
$$
 TSS 总体平方和 Total Sum of Squares，ESS 回归平方和 Explained Sum of Squares， RSS 残差平方和 Residual Sum of Squares 

###### 拟合优度：

$$
R^{2}=\frac{ESS}{TSS}=1-\frac{RSS}{TSS}
$$

#当 RSS 越小时，R2R^{2}R2 就越趋近于 1，那么代表模型的解释力越强。反之，模型的解释力就越弱。所以，一般情况下，R2R^{2}R2 的有效取值范围在 [0,1][0, 1][0,1] 之间。值越大，就代表模型的拟合优度越好。 

### 逻辑回归

由2个函数构成1，回归函数2，逻辑函数（Sigmoid）
$$
z_{i} = {w_0}{x_0} + {w_1}{x_1} + \cdots + {w_i}{x_i} = {w^T}x
$$

$$
f(z_{i})=\frac{1}{1+e^{-z_{i}}}
$$



### 感知机

##### Sign函数

$$
\operatorname{sign}(x)=\left\{\begin{array}{ll}{+1,} & {\text { if } x \geq 0} \\ {-1,} & {\text { if }  x<0}\end{array}\right.
$$

##### 损失函数

$$
J(W,b) = - \sum_{x_i\epsilon M} y_i(W*x_{i}+b)
$$
[TOC]

### 配对交易 Pair Trading

Mean-Reverting 均值回归思想，两者价差过大时，会收敛。会有风险。

范围举例：

- AH股：差价套利

- 内外盘：跨市场套利
- 同行业两只股票：两只股票价差过大，有理由相信会回归
- 两只ETF

同行业2个股票A，B历史经验价差10±元，（B-A），当价差变15元，shortB，longA，当价差变为10，就盈利。为统计套利。当AB价差不回归时，就会亏。

#### pair trading的数据性质：

①stationary平稳性，μ和σ稳定 （不要random walk）

②Non-stationary非平稳    

​		a,  差分diff：δ=y(t)-y(t-1) 	#股价非平稳，股价增长率可能平稳

​		b, 协整关系Co-intergration：找到平稳的两只股票比较难，故找协整关系。假设A，B2只股票，如果ax+by           		    （线性组合）=>stationary，那ax和by存在协整关系。

##### stationary检验：

1. 肉眼查看，均值回归图像√，非均值回归图像×
2. 模型检验：AR；MA；ARMA（应用：可以根据不同时间序列使用不同模型检验，并得出结果）
   - AR（AutoRegression）自回归：y(t) = b0 + b1*y(t-1) + ξ               #拿今天的自己检验昨天的自己
     - Unit root（单位根检验）：如果b1=1 ---> Non-stationary （使用假设检验）
     - DF检验：一阶回归。检验b1在(0,1)or(1,+∞) ，希望拒绝原假设，平稳      #更希望(0,1)
     - ADF检验：多阶回归。#拿N天前的自己检验今天的自己。拒绝原假设（p-value<0.05)，平稳
   - MA
   - ARMA

##### Non-sationary检验：

1. 差分
2. 协整关系 
   - DFEG检验

### 举例检验

600199   600702

#### 两种思路

- 最简单的方法（有问题，说明思想用）

  价差spread=2，上偏1，下偏1。

  价差标准化standardied = (spread - spread.mean())/spread.sigma

   [Pair trading策略.ipynb](http://localhost:8888/notebooks/jupyter/配对交易/2.配对交易_金程教育/Pair trading策略.ipynb) 

  #策略问题：1，假设价差符合正太分布

  ​			          2，回测，用的历史数据，正常情况下，不知道价差

  解决问题：不断rebalancing

- 协整思想


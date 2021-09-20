## 第二章 感知机算法

- 数据的线性可分性

	给定一个数据集 $T$，如果存在超平面：
$$
w \cdot x+b = 0
$$
能将正实例和负实例完全、正确地划分到超平面两侧，即对所有 $y_{i}= +1$ 的实例有$w \cdot x_{i}+b > 0$，对所有 $y_{i}= -1$ 的实例有$w \cdot x_{i}+b < 0$，则称数据集 $T$ 线性可分，否则称数据集 $T$ 线性不可分

	- 感知机模型的**使用前提**

- 感知机模型的定义

	感知机是输入空间到输出空间的函数：
$$
f(x)=\operatorname{sign}(w \cdot x+b)
$$
其中输入的特征向量 $x\in \mathcal{X},\mathcal{X} \subseteq \mathbf{R}^{n}$，输出 $y \in \mathcal{Y}, \mathcal{Y} = \{+1, -1\}$。$w$ 和 $b$ 是模型参数。$w \in \mathbf{R}^{n}$ 叫做权值（weight），$b \in \mathbf{R}$ 叫做偏置（bias），$w \cdot x$ 表示二者内积，$sign$ 是符号函数：
$$
\operatorname{sign}(x)=\left\{\begin{array}{ll}{+1,} & {x \geqslant 0} \\ {-1,} & {x<0}\end{array}\right.
$$

	- **一种线性分类模型，属于判别模型**
	- 假设空间：函数集 $\{f | f(x)=w \cdot x+b\}$
	- 按照书中$x$是列向量的定义，$w$应为为行向量

- 感知机的学习策略（损失函数）

	- 空间中任一点到超平面的距离

	$$
\frac{1}{\|w\|}\left|w \cdot x_{0}+b\right|
$$

	- 选择误分类点到超平面的距离总和作为损失函数

		$$
-\frac{1}{\|w\|} \sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)
$$
$M$ 为误分类的点集。不考虑 $\frac{1}{\|w\|}$，就得到了感知机的损失函数：
$$
L(w, b)=-\sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)
$$

	 - 定义损失函数的思想是极小化损失函数，从而将预测问题转为优化方法
	 - 不直接优化模型函数的原因是，$sign$函数不可导，无法直接进行优化
	 - 对于误分类的点来说，因为预测值和真实值总是相反的，所以

		$$
-y_{i}\left(w \cdot x_{i}+b\right)>0
$$

- 学习算法

	- 经过上述过程，原问题被转化为以下优化问题
    
    给定训练数据集
$$\begin{align*} \\& T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\} \end{align*}$$   
其中，$x_{i} \in \mathcal{X} = R^{n}, y_{i} \in \mathcal{Y} = \left\{ +1, -1 \right\}, i = 1, 2, \cdots, N$。求参数$w$和$b$，使其为以下损失函数极小化问题的解
$$\begin{align*} \\& \min_{w,b} L \left( w, b \right) =  -\sum_{x_{i} \in M} y_{i} \left( w \cdot x_{i} + b \right) \end{align*}$$   
其中，$M$为误分类点的集合。

	- 采用随机梯度下降法进行优化

		求解 $w$，$b$ 就是最优化问题：
    $$
    \min _{w, b} L(w, b)=-\sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)
    $$
    任意选取一个超平面 $w_{0}$， $b_0$，采用梯度下降法（stochastic gradient descent）不断地极小化损失函数直到收敛。

    损失函数 $L(w, b)$ 的梯度由
    $$
    \begin{array}{c}{\nabla_{w} L(w, b)=-\sum_{x_{i} \in M} y_{i} x_{i}} \\ {\nabla_{b} L(w, b)=-\sum_{x_{i} \in M} y_{i}}\end{array}
    $$
    给出，随机选取一个误分类点 $(x_{i},y_{i})$，更新 $w$，$b$：
    $$
    \begin{array}{c}{w \leftarrow w+\eta y_{i} x_{i}} \\ {b \leftarrow b+\eta y_{i}}\end{array}
    $$
    $\eta$（$0<\eta \leqslant 1$）是步长，也叫学习率(learning rate)，这样不断迭代直到损失为0。

   	 - 几何上表示为，起始随意选取一个超平面，然后随机根据一个误分类点调整 $w$ 和 $b$ ，使得超平面向该点靠近，直到越过该点使分类正确。
   	 - 感知机算法的解不唯一，往往存在多个符合条件的超平面，依赖于初始值的选择以及误分类点的选取顺序。

- 感知机算法流程

	感知机算法（原始形式）：  
    
    输入：训练数据集$T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\}$，其中$x_{i} \in \mathcal{X} = R^{n}, y_{i} \in \mathcal{Y} = \left\{ +1, -1 \right\}, i = 1, 2, \cdots, N $；学习率$\eta \left( 0 < \eta \leq 1 \right)$。  
    
    输出：$w,b$；感知机模型$f \left( x \right) = sign \left( w \cdot x + b \right)$
   
    1. 选取初值$w_{0},b_{0}$ 
    2. 在训练集中选取数据$\left( x_{i}, y_{i} \right)$  
    3. 如果$y_{i} \left( w \cdot x_{i} + b \right) \leq 0$  
    $$\begin{align*} \\& w \leftarrow w + \eta y_{i} x_{i} 
    \\ & b \leftarrow b + \eta y_{i} \end{align*}$$  
    4. 转至2，直至训练集中没有误分类点。

- 感知机的对偶形式

	设初始化初值$w_{0}=0,b_{0}=0$，设$w,b$修改$n$次，则有：
    则$w,b$关于$\left( x_{i}, y_{i} \right)$的增量分别是$\alpha_{i} y_{i} x_{i}$和$\alpha_{i} y_{i}$，其中$\alpha_{i} = n_{i} \eta$。$w,b$可表示为
$$\begin{align*} \\& w = \sum_{i=1}^{N} \alpha_{i} y_{i} x_{i} 
\\ & b = \sum_{i=1}^{N} \alpha_{i} y_{i} \end{align*}$$    
其中，$\alpha_{i} \geq 0, i=1,2, \cdots, N$

	- $\alpha_{i}$ 主要取决于实例更新的次数，更新的次数越多，对应的样本点$(x_i,y_i)$离超平面越近，越难正确分类, 当 $\eta = 1$ 时，$\alpha_{i}$ 表示为第 $i$ 个实例由于误分类而进行更新的次数。
	- $\sum_i^N n_i = n$

- 对偶形式的感知机算法

	感知机算法（对偶形式）：  
    
    输入：训练数据集$T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\}$，其中$x_{i} \in \mathcal{X} = R^{n}, y_{i} \in \mathcal{Y} = \left\{ +1, -1 \right\}, i = 1, 2, \cdots, N $；学习率$\eta \left( 0 < \eta \leq 1 \right)$。               
    
    输出：$\alpha,b$；感知机模型$f \left( x \right) = sign \left( \sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x + b \right)$  ，其中$\alpha = \left( \alpha_{1}, \alpha_{2}, \cdots, \alpha_{N} \right)^{T}$
    1. $\alpha \leftarrow 0, b \leftarrow 0$ 
    2. 在训练集中选取数据$\left( x_{i}, y_{i} \right)$  
    3. 如果$y_{i} \left( \sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x_{i} + b \right) \leq 0$  
    $$\begin{align*} \\& \alpha_{i} \leftarrow \alpha_{i} + \eta
    \\ & b \leftarrow b + \eta y_{i} \end{align*}$$  
    4. 转至2，直至训练集中没有误分类点。

- $x_{j}\cdot x_{i}$ 实际上是$x_{j}^T \cdot x_{i}$，是一个数
- 感知机计算的优势，內积矩阵的计算
$$\begin{align*} \\& G = \left[ x_{i}^T \cdot x_{j} \right]_{N \times N} \end{align*}$$ 
	
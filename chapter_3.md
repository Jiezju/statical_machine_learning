## KNN算法

- 算法流程

    输入：
    
    &ensp;&ensp;&ensp;&ensp;&ensp;**训练数据集**
    $$T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\}$$
    
    &ensp; &ensp; &ensp; &ensp; 其中，
    
    $$x_{i} \in \mathcal{X} \subseteq R^{n}$$
    &ensp; &ensp; &ensp; &ensp; 是实例的特征向量，$x_{i} = \left( x_{i}^{\left( 1 \right)},x_{i}^{\left( 2 \right) },\cdots,x_{i}^{\left( n \right) } \right)^{T}$
    
    $$ y_{i} \in \mathcal{Y} = \left\{ c_{1}, c_{2}, \cdots, c_{K} \right\}$$
    &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;是实例的类别，$ i = 1, 2, \cdots, N$；
    
    &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**实例特征向量** $x$  
输出：实例 $x$ 所属的类 $y$  
	1. 根据给定的距离度量，在训练集 $T$ 中找出与 $x$ 最近邻的 $k$ 个点，涵盖这 $k$ 点的 $x$ 的邻域记 作$N_{k} \left( x \right)$；  
	2. 在$N_{k} \left( x \right)$中根据分类决策规则决定 $x$ 的类别 $y$：
$$\begin{align*}  \\ & y = \arg \max_{c_{j}} \sum_{x_{i} \in N_{k} \left( x \right)} I \left( y_{i} = c_{j} \right), \quad i=1,2, \cdots, N; \quad j=1,2,\cdots,K \end{align*}$$   

	表决 $x$ 的类别。其中 $I$ 是指示函数，即当  $y_{i} = c_{i}$ 时 $I$ 为 1，否则为 0
    
- 算法核心

	- **距离度量**

		特征空间中两个点的距离是两个实例相似程度的反映。
        
        设特征空间 $\mathcal{X}$ 是 $n$ 维实数向量空间 $R^n$，$x_{i}, x_{j} \in \mathcal{X}, x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(n)}\right)^{\mathrm{T}}$，$x_{j}=\left(x_{j}^{(1)}, x_{j}^{(2)}, \cdots, x_{j}^{(n)}\right)^{\mathrm{T}}$，$x_{i},x_{j}$ 间的距离 $L_{p}$ 定义为：
$$
L_{p}\left(x_{i}, x_{j}\right)=\left(\sum_{l=1}^{n}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|^{p}\right)^{\frac{1}{p}}
$$
这里 $p \geqslant 1$ ，当 $p = 2$ ，成为**欧氏距离**（Euclidean distance）
$$
L_{2}\left(x_{i}, x_{j}\right)=\left(\sum_{l=1}^{n}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|^{2}\right)^{\frac{1}{2}}
$$
当 $p = 1$，称为**曼哈顿距离**（Manhattan distance）
$$
L_{1}\left(x_{i}, x_{j}\right)=\sum_{l=1}^{n}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|
$$
当 $p = \infty$，它是各个坐标距离的最大值
$$
L_{\infty}\left(x_{i}, x_{j}\right)=\max _{l}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|
$$

 - **k 值的选择**

	- k 值较小，模型复杂

		只在比较小的邻域范围内“学习”，**近似误差（approximation error）**小，估计误差大（estimation error），容易过拟合
        
   - k 值较大，模型简单

		在大范围的邻域内”学习“，近似误差大，估计误差小
        
     **一般取一个较小的数值，通过交叉验证选择最优 $k$ 值**

 - **多数表决原则**

	对给定的实例$x \in \mathcal{X}$，其最近邻的$k$个训练实例点构成的集合$N_{k} \left( x \right)$。如果涵盖$N_{k} \left( x \right)$的区域的类别是$c_{j}$，则误分类率
$$\begin{align*}  \\ & \dfrac{1}{k} \sum_{x_{i} \in N_{k} \left( x \right)} I \left( y_{i} \neq c_{j}\right) = 1 -\dfrac{1}{k} \sum_{x_{i} \in N_{k} \left( x \right)} I \left( y_{i} = c_{j}\right) \end{align*}$$
即经验风险最小化等价于多数表决规则。

	- 解析： 

		多数表决，就是使预测区域集合$N_{k} \left( x \right)$都预测为类别最多的类别，则$\sum_{x_{i} \in N_{k} \left( x \right)} I \left( y_{i} = c_{j}\right)$ 最大，所以多数表决等价于经验风险最小化
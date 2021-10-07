## 朴素贝叶斯

- 背景知识

	已知数据集
$$\begin{align*} \\& T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\} \end{align*} $$
由$P \left( X, Y \right)$独立同分布产生。其中，
$$x_{i} \in \mathcal{X} \subseteq R^{n}, y_{i} \in \mathcal{Y} = \left\{ c_{1}, c_{2}, \cdots, c_{K} \right\}, i = 1, 2, \cdots, N$$

  $x_{i}$ 为第 $i$ 个特征向量（实例），$y_{i}$ 为 $x_{i}$ 的类标记，
$X$是定义在输入空间 $\mathcal{X}$ 上的随机向量，$Y$ 是定义在输出空间$\mathcal{Y}$ 上的随机变量。$P \left( X, Y \right)$ 是 $X$ 和 $Y$ 的联合概率分布。

- 条件概率公式以及贝叶斯公式

	$$P(Y|X) = \frac {P(X,Y)} {P(X)} = \frac {P(X|Y) \cdot P(Y)}{P(X)} = \frac {P(X|Y) \cdot P(Y)}{\sum_{Y} P(X|Y) \cdot P(Y)}$$
    
    - $\sum_{Y} P(X|Y) \cdot P(Y)$表示所有 $Y$ 情形下的 $X$ 发生的概率即全概率
	- 这个公式是用**贝叶斯公式解决问题的常用公式**

- 条件独立性假设

	$$\begin{align*} \\& P \left( X = x | Y = c_{k} \right) = P \left( X^{\left( 1 \right)} = x^{\left( 1 \right)} , \cdots, X^{\left( n \right)} = x^{\left( n \right)} | Y = c_{k}\right) 
\\ & \quad\quad\quad\quad\quad\quad = \prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right) \end{align*}$$   
即，用于分类的特征在类确定的条件下都是条件独立的（朴素的内涵）。

- 贝叶斯算法推导

	- 问题定义

		类别预测就是已知实例 $x$ 下的类别 $y$ 的条件概率 $P(y|X=x)$ 的确定，理论基础就是贝叶斯概率公式以及条件独立性假设
    
    - 后验概率公式 $P(y|X=x)$ 的推导
   
   $$\begin{align*} \\ & P \left( Y = c_{k}| X = x \right) = \dfrac{P \left(X = x | Y = c_{k} \right) P \left( Y = c_{k} \right)}{P \left( X = x  \right)} 
\\ & \quad\quad\quad\quad\quad\quad = \dfrac{P \left(X = x | Y = c_{k} \right) P \left( Y = c_{k} \right)}{\sum_{Y} P \left( X = x, Y = c_{k}  \right)}
\\ & \quad\quad\quad\quad\quad\quad = \dfrac{P \left(X = x | Y = c_{k} \right) P \left( Y = c_{k} \right)}{\sum_{Y} P \left(X = x | Y = c_{k} \right) P \left( Y = c_{k} \right)}
\\ & \quad\quad\quad\quad\quad\quad = \dfrac{ P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right)}{\sum_{Y} P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right)}\end{align*}$$ 
		- 公式解析

			- 分子的 $Y=c_k$ 表示指定事件 $Y=c_k$，分母 $Y=c_k$ 泛指 $Y$  的所有可能取值的概率，用于求和
			- 上述公式表明，要获取 $x$ 的最终类别，其实就是通过数据集 $T$ 学习到了条件概率分布 $P \left( Y| X \right)$， 从公式推导看出实际上来自于**对 $P(X,Y)$的学习**，也就是朴素贝叶斯是**生成模型**的原因。

	- 朴素贝叶斯分类器的表示
	
    	$$\begin{align*} \\& y = f \left( x \right) = \arg \max_{c_{k}} \dfrac{ P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right)}{\sum_{Y} P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right)}
\\ & \quad\quad\quad = \arg \max_{c_{k}} P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right)\end{align*} $$

		- 解析

			- 分母其实就是$P(X)$，与最终求解的变量 $c_k$ 无关，所以可以不考虑
			- 贝叶斯分类器的意义是在各个类别 $y$ 下，已知 $x$ 的各个特征的概率之积得到贝叶斯后验概率分布，取最大概率对应类别 $c_k$ 即可

- 贝叶斯算法

	 输入：
     
     &ensp; &ensp; &ensp;**线性可分训练数据集**
     $$T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\}$$  
     &ensp; &ensp; &ensp; &ensp;其中
     $$x_{i}＝ \left( x_{i}^{\left(1\right)},x_{i}^{\left(2\right)},\cdots x_{i}^{\left(n\right)} \right)^{T}$$
     &ensp; &ensp; &ensp;$x_{i}^{\left( j \right)}$是第$i$个样本的第$j$个特征，$x_{i}^{\left( j \right)} \in \left\{ a_{j1}, a_{j2}, \cdots, a_{j S_{j}} \right\}$
     
     &ensp; &ensp; &ensp;$a_{jl}$是第$j$个特征可能取的第$l$个值， 
     
     &ensp; &ensp; &ensp;$j = 1, 2, \cdots, n; l = 1, 2, \cdots, S_{j}$
     
    &ensp; &ensp; &ensp; $y_{i} \in  \left\{ c_{1}, c_{2}, \cdots, c_{K} \right\}$；
    
    &ensp; &ensp; &ensp;**实例 $x$**              
输出：实例$x$的分类

	1. 计算先验概率及条件概率
$$\begin{align*}  \\ & P \left( Y = c_{k} \right) = \dfrac{\sum_{i=1}^{N} I \left( y_{i} = c_{k} \right)}{N} \quad  k = 1, 2, \cdots, K
\\ & P \left( X^{\left( j \right)} = a_{jl} | Y = c_{k} \right) ＝ \dfrac{\sum_{i=1}^{N} I \left(x_{i}^{\left( j \right)}=a_{jl}, y_{i} = c_{k} \right)}{\sum_{i=1}^{N} I \left( y_{i} = c_{k} \right)}
\\ & j = 1, 2, \cdots, n;\quad l = 1, 2, \cdots, S_{j};\quad k = 1, 2, \cdots, K\end{align*}$$     

	2. 对于给定的实例$x=\left( x^{\left( 1 \right)}, x^{\left( 2 \right)}, \cdots, x^{\left( n \right)}\right)^{T}$，计算
$$\begin{align*}  \\ & P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right) \quad  k=1,2,\cdots,K\end{align*}$$   

	3. 确定实例$x$的类别  
$$\begin{align*} \\& y = f \left( x \right) = \arg \max_{c_{k}} P \left( Y = c_{k} \right)\prod_{j=1}^{n} P \left( X^{\left( j \right)} = x^{\left( j \right)} | Y = c_{k} \right)  \end{align*}$$  
        
## 逻辑斯蒂回归模型

- 二项逻辑斯蒂回归模型

	$$
\begin{array}{l}{P(Y=1 | x)=\frac{\exp (w \cdot x)}{1+\exp (w \cdot x)}} \\ \\ {P(Y=0 | x)=\frac{1}{1+\exp (w \cdot x)}}\end{array}
$$

	$Y$ 是输出，$w = (w^{(1)},w^{(2)},\ldots,w^{(n)},b)$，$x = (x^{(1)},x^{(2)},\ldots,x^{(n)},1)^{T}$，$b$ 是偏置
    
    - 逻辑斯蒂回归模型输出 $𝑌=1$ 的对数几率是输入 $𝑥$ 的线性函数

		$$
\log \frac{P(Y=1 | x)}{1-P(Y=1 | x)}=w \cdot x
$$

- 建立最大似然估计模型

	**1. 准备数据集-给定训练数据集**
$$\begin{align*} \\& T = \left\{ \left( x_{1}, y_{1} \right), \left( x_{2}, y_{2} \right), \cdots, \left( x_{N}, y_{N} \right) \right\} \end{align*}$$   
其中，$x_{i} \in R^{n+1}, y_{i} \in \left\{ 0, 1 \right\}, i = 1, 2, \cdots, N$。

	令
    $$
    P(Y=1|x) = \pi(x),\quad P(Y=0|x)=1 - \pi(x)
    $$
    
    **2. 计算似然函数**
    
    $$\begin{align*} \\& l \left( w \right) = \prod_{i=1}^{N} P \left( y_{i} | x_{i} \right) 
\\ & = P \left( Y = 1 | x_{i} , w \right) \cdot P \left( Y = 0 | x_{i}, w \right) 
\\ & = \prod_{i=1}^{N} \left[ \pi \left( x_{i} \right) \right]^{y_{i}}\left[ 1 - \pi \left( x_{i} \right) \right]^{1 - y_{i}}\end{align*}$$ 

	**3. 计算对数似然函数**
    
    $$\begin{align*} \\& L \left( w \right) = \log l \left( w \right) 
\\ & = \sum_{i=1}^{N} \left[ y_{i} \log \pi \left( x_{i} \right) + \left( 1 - y_{i} \right) \log \left( 1 - \pi \left( x_{i} \right) \right) \right]
\\ & = \sum_{i=1}^{N} \left[ y_{i} \log \dfrac{\pi \left( x_{i} \right)}{1- \pi \left( x_{i} \right)} + \log \left( 1 - \pi \left( x_{i} \right) \right) \right]
\\ & = \sum_{i=1}^{N} \left[ y_{i} \left( w \cdot x_{i} \right) - \log \left( 1 + \exp \left( w \cdot x \right)  \right) \right]\end{align*}$$ 

	**4. 似然函数梯度计算**
    
    $$
    \frac {\partial L(w)}{\partial w} = \sum_{i}^{N}(y_i \cdot x_i - \frac{x_i \cdot e^{w \cdot x_i}}{1 + e^{w \cdot x_i}})
    $$
    
    

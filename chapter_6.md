## 逻辑斯蒂回归模型

- **二项逻辑斯蒂回归模型**

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

- **最大熵模型**

    - 最大熵原理

		最大熵原理认为在给定约束条件下的所有可能模型中，熵最大的模型是最好的模型。
        
	随机变量 $X$ 的熵表示为 $H(P)=-\sum_{x} P(x) \log P(x)$。熵都满足不等式：
$$
0 \leqslant H(P) \leqslant \log |X|
$$
$|X|$ 表示 $X$ 的取值个数，当且仅当 $X$ 服从均匀分布，各个取值机会均等的时候，右边等号成立，熵最大。

	- **总结**

		概率模型能给定已有的约束条件就给定，不能的话就认为等可能性。
        
  - 最大熵模型的定义     

	给定数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$
    
    最大熵模型的学习等价于约束最优化问题：
$$
\begin{array}{ll}{\max _{P \in \mathbb{C}}} & {H(P)=-\sum_{x, y} \tilde{P}(x) P(y | x) \log P(y | x)} \\ {\text { s.t. }} & {E_{P}\left(f_{i}\right)=E_{\tilde{P}}\left(f_{i}\right), \quad i=1,2, \cdots, n} \\ {} & {\sum_{y} P(y | x)=1}\end{array}
$$
改成就最小值的形式：
$$
\begin{array}{ll}{\min _{P \in \mathbb{C}}} & {-H(P)=\sum_{x, y} \tilde{P}(x) P(y | x) \log P(y | x)} \\ {\text { s.t. }} & {E_{P}\left(f_{i}\right)-E_{\tilde{P}}\left(f_{i}\right)=0, \quad i=1,2, \cdots, n} \\ {} & {\sum_{y} P(y | x)=1}\end{array}
$$

	其中， 
    
    - 经验分布 
$$
\begin{array}{l}{\tilde{P}(X=x, Y=y)=\frac{\nu(X=x, Y=y)}{N}} \\ {\tilde{P}(X=x)=\frac{\nu(X=x)}{N}}\end{array}
$$
$v(x)$ 表示样本中 $x$ 出现的频次。

	- 样本数据模型预测的期望值 $E_{\tilde{P}}(f)$ 和我们直观观测到的期望值 $E_{P}(f)$ 应该相同。
$$
E_{\tilde{P}}(f)=\sum_{x, y} \tilde{P}(x, y) f(x, y) \\
E_{P}(f)=\sum_{x, y} \tilde{P}(x) P(y | x) f(x, y)
$$
其中 $f(x,y)$ 是特征函数（feature function）
$$
f(x, y)=\left\{\begin{array}{ll}{1,} & x 与 y满足某一事实 \\ {0,} & 否则\end{array}\right.
$$

 - 问题求解：采用拉格朗日乘数法

	- 构建拉格朗日函数

		引入拉格朗日乘子 $w_{0},w_{1},\ldots,w_{n}$，得到拉格朗日函数 $L(P,w)$
$$
\begin{aligned} L(P, w) \equiv &-H(P)+w_{0}\left(1-\sum_{y} P(y | x)\right)+\sum_{i=1}^{n} w_{i}\left(E_{\tilde{P}}\left(f_{i}\right)-E_{P}\left(f_{i}\right)\right) \\=& \sum_{x, y} \tilde{P}(x) P(y | x) \log P(y | x)+w_{0}\left(1-\sum_{y} P(y | x)\right)+\\ & \sum_{i=1}^{n} w_{i}\left(\sum_{x, y} \tilde{P}(x, y) f_{i}(x, y)-\sum_{x, y} \tilde{P}(x) P(y | x) f_{i}(x, y)\right) \end{aligned}
$$

	- 转换为对偶问题

		原始最优化问题 $\min _{P \in \mathbf{C}} \max _{w} L(P, w)$ 的对偶问题是 $\max _{\boldsymbol{w}} \min _{P \in \mathbf{C}} L(P, w)$。因为函数 $L(P,w)$ 是 $P(y|x)$ 的凸函数，所以原始问题和对偶问题的解是等价的。
        
    - 拉格朗日问题求解

	 - **引入拉格朗日乘子$w_{i}, i = 0,1, \cdots, n$，定义拉格朗日函数$L \left( P, w \right)$**
    $$\begin{align*} \\& L \left( P, w \right) = - H \left( P \right) + w_{0} \left( 1 - \sum_{y} P \left( y | x \right) \right) + \sum_{i=1}^{n} w_{i} \left( E_{P} \left( f_{i} \right) - E_{\tilde{P}} \left( f_{i} \right) \right) 
    \\ & = \sum_{x,y} \tilde{P} \left( x \right) P \left( y | x \right) \log P \left( y | x \right) + w_{0} \left( 1 - \sum_{y} P \left( y | x \right) \right) 
    \\ & \quad + \sum_{i=1}^{n} w_{i} \left( \sum_{x, y} \tilde{P} \left( x \right) P \left( y | x \right) f_{i} \left( x, y \right) - \sum_{x, y} \tilde{P} \left( x, y \right) f_{i} \left( x, y \right) \right) \end{align*} $$
	   
	  - **求$\min_{P \in \mathcal{C} } L \left( P, w \right)$**:  
		
       记对偶函数$\Psi \left( w \right) = min_{P \in \mathcal{C} } L \left( P, w \right) = L \left( P_{w}, w \right)$，其解记$P_{w} = \arg \min_{P \in \mathcal{C} } L \left( P, w \right) = P_{w} \left( y | x \right)$
	$$\begin{align*} \\& \dfrac {\partial L \left( P, w \right)} {\partial P \left( y | x \right)} = \sum_{x,y} \tilde{P} \left( x \right) \left( \log P \left( y | x \right) + 1 \right) - \sum_{y} w_{0} - \sum_{x,y} \left( \tilde{P} \left( x \right) \sum_{i=1}^{n} w_{i}  f_{i} \left( x, y \right) \right) 
\\ & \quad = \sum_{x,y} \tilde{P} \left( x \right) \left( \log P \left( y | x \right) + 1 \right) - \sum_{x,y} P \left( x \right) w_{0} - \sum_{x,y} \left( \tilde{P} \left( x \right) \sum_{i=1}^{n} w_{i}  f_{i} \left( x, y \right) \right) 
\\ & \quad = \sum_{x,y} \tilde{P} \left( x \right) \left( \log  P \left( y | x \right) + 1 - w_{0} - \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right) = 0\end{align*}$$   
由于$\tilde{P} \left( x \right) > 0 $，得
$$\begin{align*} \\ & \log  P \left( y | x \right) + 1 - w_{0} - \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right)=0
\\ &  P \left( y | x \right) = \exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) + w_{0} -1 \right) = \dfrac{ \exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right) }{ \exp  \left( 1 - w_{0} \right)}\end{align*}$$  
由于  
$$\begin{align*} \\& \sum_{y} P \left( y | x \right) = 1 \end{align*} $$
则
$$\begin{align*}  \\ & \sum_{y} P \left( y | x \right) = \sum_{y} \dfrac{ \exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right) }{ \exp  \left( 1 - w_{0} \right)} = 1 
\\ & \sum_{y} \exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right) = \exp  \left( 1 - w_{0} \right)\end{align*}$$   
代入，得
$$\begin{align*}  \\ & P \left( y | x \right) = \dfrac{1 }{Z_{w} \left( x \right)}\exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right) \end{align*}$$   
其中
$$\begin{align*} Z_{w} = \sum_{y} \exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right)  \end{align*}$$  
$Z_{w}$称为规范化因子；$f_{i} \left( x, y \right)$是特征函数；$w_{i}$是特征的权值。

	 - 求$\max_{w} \Psi \left( w \right)$  
	
    	将其解记为$w^{*}$，即
$$\begin{align*} w^{*} = \arg \max_{w} \Psi \left( w \right) \end{align*}$$  
        
   -  对偶函数 $\Psi(w)$ 的极大化求解

		- 原理：**最大熵模型（条件熵）的极大似然估计等价于对偶函数极大化**
        
        所以原问题转为对 $\Psi(w)$ 的极大似然估计
        
        - 证明：
        
        	在经验概率分布$\tilde{P}(X,Y)$ 上，条件概率分布 $P(Y|X)$ 的极大似然函数 $L_{1}$ 为
        $$
        L_{1}=\prod_{i=1}^{n} P\left(y_{i} | x_{i}\right)=P\left(y_{1} | x_{1}\right) \cdots P\left(y_{n} | x_{n}\right)
        $$
$n$ 个样例里难免会有相同样例$(x^{i},y^{j})$，于是 $L_{1}$ 可以写为
        $$
        {L_{1}=P\left(y^{1} | x^{1}\right)^{v\left(x^{1}, y^{1}\right)} \cdots P\left(y^{r} | x^{m}\right)^{v\left(x^{m}, y^{r}\right)}}=\prod_{y, x} P(y | x)^{v(x, y)}
        $$
$v$ 表示共现数量，而此时 $L_1$ 的极大化等价于下式极大化
            $$
    L_{2}=\prod_{y, x} P(y | x)^{\frac{v(x, y)}{N}}=\prod_{y, x} P(y | x)^{\tilde{P}(x, y)}
    $$
  		依据上述原理进行求解 $\Psi(w)$
$$\begin{align*} \\ & L_{\tilde{P}} \left( P_{w} \right) = \log \prod_{x,y} P \left( y | x \right)^{\tilde{P} \left( x, y \right)} 
\\ & = \sum_{x,y} \tilde{P} \left( x, y \right) \log P \left( y | x \right)
\\ & = \sum_{x,y} \tilde{P} \left( x, y \right) \log \dfrac{\exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right)}{Z_{w} \left( x \right) }
\\ & = \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) - \sum_{x,y} \tilde{P} \left( x, y \right) \log Z_{w} \left( x \right) 
\\ & = \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) - \sum_{x} \tilde{P} \left( x \right) \log Z_{w} \left( x \right)\end{align*}$$    
对偶函数
$$\begin{align*} \\ & \Psi \left( w \right) = min_{P \in \mathcal{C} } L \left( P, w \right) = L \left( P_{w}, w \right) \\ & = - H \left( P_{w} \right) + w_{0} \left( 1 - \sum_{y} P_{w} \left( y | x \right) \right) + \sum_{i=1}^{n} w_{i} \left( E_{\tilde{P}} \left( f_{i} \right) - E_{P_{w}} \left( f_{i} \right) \right)  
\\ & = \sum_{x,y} \tilde{P} \left( x \right) P_{w} \left( y | x \right) \log P_{w} \left( y | x \right)
\\& \quad\quad\quad + w_{0} \left( 1 - \sum_{y} \dfrac{1 }{Z_{w} \left( x \right)}\exp \left( \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) \right) \right)
\\ & \quad\quad\quad + \sum_{i=1}^{n} w_{i} \left( \sum_{x, y} \tilde{P} \left( x, y \right) f_{i} \left( x, y \right) - \sum_{x, y} \tilde{P} \left( x \right) P_{w} \left( y | x \right) f_{i} \left( x, y \right) \right)  
\\ & = \sum_{x, y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} w_{i}   f_{i} \left( x, y \right) + \sum_{x,y} \tilde{P} \left( x \right) P_{w} \left( y | x \right) \left( \log P_{w} \left( y | x \right) - \sum_{i=1}^{n} w_{i} f_{i} \left(x, y \right) \right) 
\\ & = \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) - \sum_{x,y} \tilde{P} \left( x, y \right) \log Z_{w} \left( x \right) 
\\ & = \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} w_{i} f_{i} \left( x, y \right) - \sum_{x} \tilde{P} \left( x \right) \log Z_{w} \left( x \right)\end{align*}$$   
得
$$\begin{align*} \\ & L_{\tilde{P}} \left( P_{w} \right) = \Psi \left( w \right)\end{align*}$$

 - 优化最大熵模型

	最大熵模型都可以归结为似然函数为目标函数的最优化问题，且目标函数还是凸函数。所以优化方法有很多，常用的有**改良迭代尺度法，梯度下降法，牛顿法或拟牛顿法**。后两者收敛更快些（二阶导）
	
    - 改进的迭代尺度法

		- 算法思想: 如果能找到一种更新参数向量 𝑤 的方法 **𝜏:𝑤→𝑤+𝛿**，因为加了 𝛿 使得似然函数变大，那么不断重复这个过程就能找到似然函数的最大值。

		- 获取优化目标函数下界
	
		对数似然函数变大可以表示为增量 $L(w+\delta)-L(w) > 0$，具体的
$$
\begin{aligned} L(w+\delta)-L(w) &=\sum_{x, y} \tilde{P}(x, y) \log P_{w+\delta}(y | x)-\sum_{x, y} \tilde{P}(x, y) \log P_{w}(y | x) \\ &=\sum_{x, y} \tilde{P}(x, y) \sum_{i=1}^{n} \delta_{i} f_{i}(x, y)-\sum_{x} \tilde{P}(x) \log \frac{Z_{w+\delta}(x)}{Z_{w}(x)} \end{aligned}
$$
其中$P_w(y|x)$是最大熵模型，$Z_{w}(x)$ 是规范化因子 
$$
P_{w}(y | x)=\frac{\exp \left(\sum_{i=1}^{n} w_{i} f_{i}(x, y)\right)}{\sum_{y}\exp \left(\sum_{i=1}^{n} w_{i} f_{i}(x, y)\right)}=\frac{\exp \left(\sum_{i=1}^{n} w_{i} f_{i}(x, y)\right)}{Z_{w}(x)}
$$
利用不等式 $-\log \alpha \geqslant 1- \alpha,\alpha > 0$，可得
$$
\begin{aligned} L(w+\delta)-L(w) & \geqslant \sum_{x, y} \tilde{P}(x, y) \sum_{i=1}^{n} \delta_{i} f_{i}(x, y)+1-\sum_{x} \tilde{P}(x) \frac{Z_{w+\delta}(x)}{Z_{w}(x)} \\ &=\sum_{x, y} \tilde{P}(x, y) \sum_{i=1}^{n} \delta_{i} f_{i}(x, y)+1-\sum_{x} \tilde{P}(x) \sum_{y} P_{w}(y | x) \exp \sum_{i=1}^{n} \delta_{i} f_{i}(x, y) \end{aligned}
$$
	记
$$\begin{align*} \\ & A \left( \delta | w \right) = \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} \delta_{i} f_{i} \left( x, y \right) + 1 - \sum_{x} \tilde{P} \left( x \right) \sum_{y} P_{w} \left( y | x \right) \exp \left( \sum_{i=1}^{n}  \delta_{i}  f_{i} \left( x, y \right) \right)\end{align*}$$     
则   
$$\begin{align*}  \\ & L \left( w + \delta \right) - L \left( w \right) \geq A \left( \delta | w \right)\end{align*}$$   
即$ A \left( \delta | w \right)$是对数似然函数改变量的一个下界。

		**如果下界越大，似然函数也就越大**，但不等式右边的 $\delta$ 是向量，变量太多。所以 **IIS 试图只优化一个变量 $\delta_i$**，其他变量不动，获取一个更低但好计算的下界

		引入 $f^\#(x,y) = \sum_i f_i(x,y)$，表示所有特征为$(x,y)$的样本数。这样
$$
\begin{aligned} A(\delta | w)=& \sum_{x, y} \tilde{P}(x, y) \sum_{i=1}^{n} \delta_{i} f_{i}(x, y)+1-\sum_{x} \tilde{P}(x) \sum_{y} P_{w}(y | x) \exp \left(f^{\#}(x, y) \sum_{i=1}^{n} \frac{\delta_{i} f_{i}(x, y)}{f^{\#}(x, y)}\right) \end{aligned}
$$

	 - 使用 **Jensen 不等式** 进一步确定下界

		- **Jensen 不等式**
	
            过一个凸函数上任意两点所作割线一定在这两点间的函数图象的上方，即
        $$
        t f\left(x_{1}\right)+(1-t) f\left(x_{2}\right) \geqslant f\left(t x_{1}+(1-t) x_{2}\right), 0 \leqslant t \leqslant 1
        $$
        泛化形式为，对点集 $\{x_i\}$，如果 $\lambda_i \geqslant 0,\sum_i \lambda_i = 1$，则有 $f\left(\sum_{i=1}^{M} \lambda_{i} x_{i}\right) \leqslant \sum_{i=1}^{M} \lambda_{i} f\left(x_{i}\right)$

        	将 $\lambda$ 视作概率分布的话，那么就可以写做 $f(E[x]) \leqslant E[f(x)]$
       
       - 代入 $ A \left( \delta | w \right)$

            对任意$i$，有$\dfrac{f_{i} \left( x, y \right)}{f^{\#} \left( x, y \right)} \geq 0$且$\sum_{i=1}^{n} \dfrac{f_{i} \left( x, y \right)}{f^{\#} \left( x, y \right)} = 1$,  
    根据Jensen不等式，得  
    $$\begin{align*}  \\ &  \exp \left( \sum_{i=1}^{n} \dfrac{f_{i} \left( x, y \right)}{f^{\#} \left( x, y \right)} \delta_{i} f_{\#} \left( x, y \right) ) \right)  \leq \sum_{i=1}^{n} \dfrac{f_{i} \left( x, y \right)}{f^{\#} \left( x, y \right)} \exp \left( \delta_{i} f^{\#} \left(x, y\right) \right)\end{align*}$$     
    则
    $$\begin{align*}  \\ & A \left( \delta | w \right) \geq \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} \delta_{i} f_{i} \left( x, y \right) + 1 - \sum_{x} \tilde{P} \left( x \right) \sum_{y} P_{w} \left( y | x \right)  \sum_{i=1}^{n} \left( \dfrac{f_{i} \left( x, y \right)}{f^{\#} \left( x, y \right)} \right) \exp \left( \delta_{i} f^{\#} \left(x, y\right) \right)\end{align*}$$
	
			记
$$\begin{align*} \\ & B \left( \delta | w \right) = \sum_{x,y} \tilde{P} \left( x, y \right) \sum_{i=1}^{n} \delta_{i} f_{i} \left( x, y \right) + 1 - \sum_{x} \tilde{P} \left( x \right) \sum_{y} P_{w} \left( y | x \right)  \sum_{i=1}^{n} \left( \dfrac{f_{i} \left( x, y \right)}{f^{\#} \left( x, y \right)} \right) \exp \left( \delta_{i} f^{\#} \left(x, y\right) \right)\end{align*}$$     
则   
$$\begin{align*}  \\ & L \left( w + \delta \right) - L \left( w \right) \geq A \left( \delta | w \right) \geq B \left( \delta | w \right)\end{align*}$$   
即$ B \left( \delta | w \right)$是对数似然函数改变量的一个新的（相对不紧的）下界。

		- 求解$ B \left( \delta | w \right)$的最优解
		
        	对 $\delta_i$ 求偏导得
        $$
        \frac{\partial B(\delta | w)}{\partial \delta_{i}}=\sum_{x, y} \tilde{P}(x, y) f_{i}(x, y)-\sum_{x} \tilde{P}(x) \sum_{y} P_{w}(y | x) f_{i}(x, y) \exp \left(\delta_{i} f^{\#}(x, y)\right)
        $$
	令导数为 0 ，得到只关于 $\delta_i$ 的方程
    $$\sum_{x, y} \tilde{P}(x) P_{w}(y | x) f_{i}(x, y) \exp \left(\delta_{i} f^{\#}(x, y)\right)=E_{\tilde{P}}\left(f_{i}\right)\tag{1}$$
	这样我们就可以计算 $\delta$ ，更新 $w$ 
        - 采用改进的迭代尺度算法更新 参数 $w$ 

           - 输入：
           	
            特征函数$f_{i},i=1, 2, \cdots, n$，经验分布$\tilde{P} \left( x, y \right)$，模型$P_{w} \left( y | x \right)$  
           - 输出：最优参数值$w_{i}^{*}$；最优模型$P_{w^{*}}$ 
        
           - 算法流程	
          
          	- 对所有$i \in \left\{ 1, 2, \cdots, n \right\}$，取$w_{i} = 0$；
        
        	- 对每一$i \in \left\{ 1, 2, \cdots, n \right\}$  
        		-  令$\delta_{i}$是方程
        	$$\begin{align*} \\ & \sum_{x,y} \tilde{P} \left( x, y \right) f_{i} \left( x, y \right) = \sum_{x, y} \tilde{P} \left( x \right) P_{w} \left( y | x \right) f_{i} \left( x, y \right) \exp \left( \delta_{i} f^{\#} \left(x, y\right) \right)  \end{align*}$$ 的解  
        		- 更新$w_{i}$的值
        		$$\begin{align*} \\ & w_{i} \leftarrow w_{i} + \delta_{i}\end{align*}$$ 
        		
                - 如果不是所有$w_{i}$都收敛，重复步骤2.

		- 使用**拟牛顿法**求解关于$\delta_{i}$的方程
			
          当 $f^{\#}(x, y)=M$，$\delta_i$ 可表示为 $\delta_{i}=\frac{1}{M} \log \frac{E_{\tilde{P}}\left(f_{i}\right)}{E_{P}\left(f_{i}\right)}$
          
          当 $f^{\#}(x, y)$ 不是常数，用$g(\delta_i)=0$ 表示，使用牛顿法迭代求解得到 $g(\delta^*)=0$，迭代公式为
        $$
        \delta_{i}^{(k+1)}=\delta_{i}^{(k)}-\frac{g\left(\delta_{i}^{(k)}\right)}{g^{\prime}\left(\delta_{i}^{(k)}\right)}
        $$


- 最大熵模型与逻辑斯蒂模型的关系        	
        
   当分类任务为二分类 $Y=\{y_1,y_2\}$，且特征函数为
    $$
    f_{i}(\mathbf{x}, y)=\left\{\begin{array}{ll}{x_{i},} & {\text { if } y=y_{1}} \\ {0,} & 其他\end{array}\right.
    $$
   有
  $$
    P\left(y_{1} | \mathbf{x}\right)=\frac{e^{\sum_{i=1}^{n} w_{i} x_{i}}}{1+e^{\sum_{i=1}^{n} w_{i} x_{i}}}=\frac{e^{\mathbf{w}^{T} \mathbf{x}}}{1+e^{\mathbf{w}^{T} \mathbf{x}}} \\
    P\left(y_{0} | \mathbf{x}\right)=\frac{1}{1+e^{\mathbf{w}^{T} \mathbf{x}}}
    $$
	**此时最大熵模型退化成为逻辑斯蒂回归模型**
        
        

		      
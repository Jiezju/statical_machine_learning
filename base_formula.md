## 统计学习方法基本公式

### 概率论

- 样本概念标记

	**输入**：定义在输入空间上的随机变量的取值。输入变量通常记作 $X$,取值记作 $x$

	**输出**：定义在输出空间上的随机变量的取值。输出变量通常记作 $Y$,取值记作 $y$
	
    注：输入输出都可以是一维到多维，是标量或是向量，默认为**列向量**。
    
    **输入实例**：$x=\left(x^{(1)}, x^{(2)}, \cdots, x^{(i)}, \cdots, x^{(n)}\right)^{T}$，注意 $x^{(i)}$ 表示 $x$ 的第 $i$ 个特征
    
    **样本（sample）**：也称样本点，由输入和输出构成的数据对。表示为 $\left(x_{i}, y_{i}\right)$，$x_{i}$ 表示若干输入变量中的第 $i$ 个变量，即 $x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(n)}\right)^{\mathrm{T}}$ 

	**训练集（training data)**：用于模型学习的数据集合，通常记作 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$
    
- 样本矩阵

	$$X_{n*m} = [x_1,x_2,...,x_m]=\begin{bmatrix}
{x_{1}^{(1)}}&{x_{2}^{(1)}}&{\cdots}&{x_{m}^{(1)}}\\
{x_{1}^{(2)}}&{x_{2}^{(2)}}&{\cdots}&{x_{m}^{(2)}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{x_{1}^{(n)}}&{x_{2}^{(n)}}&{\cdots}&{x_{m}^{(n)}}
\end{bmatrix}_{n*m}$$

	- $n$维特征，$m$个样本

- 数字特征计算

	- 期望计算

		$$E(X)=\frac {1}{N}\sum_{i=0}^{m}x_i$$
        
        - 按特征维度求均值（按$X$行求均值）

		$$\mu =\begin{bmatrix}
           \mu_1 \\
           \mu_2 \\
           {\vdots}\\
           \mu_n 
        \end{bmatrix}=\begin{bmatrix}
       \frac {1}{m}\sum_{i=0}^{m}x_i^{(1)} \\
       \\
        \frac {1}{m}\sum_{i=0}^{m}x_i^{(2)} \\
        \\
        {\vdots}\\
        \\
         \frac {1}{m}\sum_{i=0}^{m}x_i^{(n)} 
        \end{bmatrix}_{n*1}$$
        
  - 协方差计算 
        
       $$\Sigma = E[(X-E(X))(X-E(X))^T] = \begin{bmatrix}
{\sigma_{11}}&{\sigma_{12}}&{\cdots}&{\sigma_{1n}}\\
{\sigma_{21}}&{\sigma_{22}}&{\cdots}&{\sigma_{2n}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{\sigma_{n1}}&{\sigma_{n2}}&{\cdots}&{\sigma_{nn}}
\end{bmatrix}_{n*n}$$
       
       - 此处计算涉及到广播计算，因为$X_{n*m}$维度与$E(X)_{n*1}$，则
       
       	$$E(X) = [\mu, \mu, \cdot\cdot\cdot, \mu]$$
        
		$$ E[(X-E(X)] = [x_1-\mu,x_2-\mu, \cdot\cdot\cdot,x_m-\mu]_{n*m}$$
    
    - 协方差$[\sigma_{ij}]$描述了$n$个维度特征之间相关关系
    - 协方差对角线$[\sigma_{11}, \sigma_{22}, ...., \sigma_{nn}]$为各个特征的方差
    - 协方差非对角线元素$\sigma_{ij}$描述特征$x^{(i)}$与$x^{(j)}$的相关关系
    	- $x^{(i)}$与$x^{(j)}$独立，则$\sigma_{ij}$=0
    	- $\sigma_{ij}$=0，则$x^{(i)}$与$x^{(j)}$非线性相关（包含独立）
    	- $\sigma_{ij}$>0, 则$x^{(i)}$与$x^{(j)}$正相关
    	- $\sigma_{ij}$<0, 则$x^{(i)}$与$x^{(j)}$负相关

 - 实例

	若样本矩阵
    $$X= \begin{bmatrix}
1&2&1\\
0&0&1
\end{bmatrix}$$
	样本特征数为2，样本数为3
    
    则：
    $$均值：\mu =\begin{bmatrix}
           \mu_1 \\
           \mu_2 
        \end{bmatrix} = \begin{bmatrix}
           4/3 \\
           1/3 
        \end{bmatrix} $$

	$$协方差：\Sigma = \frac{1}{m}[x_1-\mu,x_2-\mu, \cdot\cdot\cdot,x_m-\mu]\cdot [x_1-\mu,x_2-\mu, \cdot\cdot\cdot,x_m-\mu]^T$$
    $$= \frac {1}{3}\begin{bmatrix}
-1/3&2/3&-1/3\\
-1/3&-1/3&2/3
\end{bmatrix} \cdot \begin{bmatrix}
-1/3&-1/3\\
-1/3&2/3\\
2/3&-1/3
\end{bmatrix}$$	
$$=\frac {1}{3}\begin{bmatrix}
1/9&6/9\\
6/9&1/9
\end{bmatrix}$$

- 马氏距离计算

	$$D_M(x)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}$$
    
    - 这个值是一个标量

- 多维高斯分布

	$$p(x)=\frac{1}{(2\pi)^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$
    
    ![](../img/chapter_1/1.png)
    ![](../img/chapter_1/2.png)
    
    - 以二维特征为例
    	- 协方差矩阵为单位阵则为单位圆
    	- 只有对角线元素的协方差阵为正向椭圆，一般情况为椭圆

- 离散随机变量的概率公式

	- 加法公式
		
        $$P(x) = \sum_yP(x,y)$$
    
    - 乘法公式

		$$P(x) = P(x)P(y|x)$$
        
    - 全概率公式

		$$P(x|\theta) = \sum_{\theta_i}P(x|\theta_i)$$
        
        - 全概率公式表明一个事件发生的概率是所有相关条件满足情况下的该事件发生概率之和

	- 贝叶斯公式

		$$P(\theta|D) = \frac{P(\theta)P(D|\theta)}{P(D)}$$
        
        - 先验概率$P(\theta)$
        - 后验概率$P(\theta|D)$
        - 类条件概率$P(D|\theta)$

	- 贝叶斯推断(全概率公式的另一种写法)
		
		$$P(x,D) = \sum_{\theta_i} P(x|\theta_i,D)P(\theta_i|D)$$
        
- 估计公式

	假设估计模型$P_\theta(X)$（视具体情形看），预先假设含参数$\theta$分布$P_\theta(X)$的具体形式，如高斯分布等，基于独立同分布假设
    
    - 极大似然估计
		预先对总体采样，获取样本集合$\{X_i\}$
        
        极大化估计公式为
        
        $$
\begin{aligned} \theta &=\underset{\theta}{\arg \max } \prod_{i=1}^{N}P_{\theta}(X_{i})\end{aligned} = \underset{\theta}{\arg \max } \prod_{i=1}^{N}P(X_{i}|\theta)
$$

	- 贝叶斯估计
	
    	贝叶斯估计公式为
		$$
        \theta =\underset{\theta}{\arg \max } P(\theta|X) <=> \underset{\theta}{\arg \max } \prod_{i=1}^{N}P(X_i|\theta)\cdot P(\theta)
        $$
        
        预先假设 $P(\theta)$分布，代入即可
        

### 机器学习公式

- 期望损失
    $$
    \begin{aligned} R_{\mathrm{ exp }}(f) &=E_{P}[L(Y, f(X))] \\ &=\int_{\mathcal{X} \times \mathcal{Y}} L(y, f(x)) P(x, y) \mathrm{d} x \mathrm{d} y \end{aligned}
    $$
    
- 经验损失

	$$
R_{\mathrm{emp}}(f)=\frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, f\left(x_{i}\right)\right)
$$
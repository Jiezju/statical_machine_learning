## 决策树

- 决策树模型的定义

	决策树的每个内部结点（internal node）表示一个特征或属性，叶结点（leaf node）表示一个类。

每次从根结点开始，对某项特征进行测试，根据结果分配到子结点，每个子结点对应一个特征取值，递归直到抵达某个叶结点。

- 熵的定义

	熵表示**随机变量不确定性**的度量。  

	设$X$是一个取有限个值的离散随机变量，其概率分布为
$$\begin{align*} P \left( X = x_{i} \right) = p_{i}, \quad i =1, 2, \cdots, n \end{align*}$$
则随机变量$X$的熵
$$\begin{align*} H \left( X \right) = H \left( p \right) = - \sum_{i=1}^{n} p_{i} \log p_{i} \end{align*}$$   
其中，若$p_{i}=0$，则定义$0 \log 0 = 0$

- 熵的有界性

	若
$$\begin{align*} p_{i} = \dfrac{1}{n} \end{align*}$$   
则
$$\begin{align*} \\ & H \left( p \right) = - \sum_{i=1}^{n} p_{i} \log p_{i} 
\\ & = - \sum_{i=1}^{n} \dfrac{1}{n} \log \dfrac{1}{n}
\\ & = \log n\end{align*}$$ 
由定义，得
$$\begin{align*} \\ & 0 \leq H \left( p \right) \leq \log n\end{align*} $$

	由此可得，**随机变量的均匀分布熵最大**
    
- 条件熵

	随机变量$X$给定的条件下随机变量$Y$的条件熵
$$\begin{align*} \\ & H \left( Y | X \right) = \sum_{i=1}^{n} p_{i} H \left( Y | X = x_{i} \right) \end{align*}$$  
即，$X$给定条件下$Y$的条件概率分布的熵对$X$的数学期望。其中，
$$p_{i}=P \left( X = x_{i} \right), i= 1,2,\cdots,n。$$  
**条件熵$H \left( Y | X \right)$表示在已知随机变量$X$的条件下随机变量$Y$的不确定性。**

- 信息增益

	特征$A$对训练集$D$的信息增益
$$\begin{align*} \\ & g \left( D, A \right) = H \left( D \right) - H \left( D | A \right) \end{align*}$$   
即，集合$D$的经验熵$H \left( D \right)$与特征$A$给定条件下$D$的经验条件熵$H \left( D | A \right)$之差。  
其中，**当熵和条件熵由数据估计（极大似然估计）得到时，对应的熵和条件熵分别称为经验熵和经验条件熵。**  
信息增益$g \left( X , Y \right)$表示已知特征$X$的信息而使得类$Y$的信息的不确定性减少的程度。

- **信息增益算法**

    设训练数据集为$D$，$\left| D \right|$表示其样本容量，即样本个数。  
    设有$K$个类$C_{k}, k=1,2,\cdots,K$，**$\left| C_{k} \right|$为属于类$C_{k}$的样本的个数**，
    $$\sum_{k=1}^{K} \left| C_{k} \right| = \left| D \right|。$$  
    设特征$A$有$n$个不同的特征取值$\left\{ a_{1},a_{2},\cdots,a_{n}\right\}$，根据特征$A$的取值将$D$划分为$n$个子集$D_{1},D_{2},\cdots,D_{n}$，$\left| D_{i} \right|$为$D_{i}$的样本数，$\sum_{i=1}^{n}\left| D_{i} \right| = \left| D \right|$。  
    记子集$D_{i}$中属于类$C_{k}$的样本的集合为$D_{ik}$，即$D_{ik} = D_{i} \cap C_{k}$，$\left| D_{ik} \right|$为$D_{ik}$的样本个数。
 
   -  输入：训练数据集$D$和特征$A$ 
   -  输出：特征$A$对训练数据集$D$的信息增益$g \left( D, A \right) $
   -  算法：

		1. 计算数据集$D$的经验熵$H\left(D\right)$  
$$\begin{align*} \\ &  H \left( D \right) = -\sum_{k=1}^{K} \dfrac{\left|C_{k}\right|}{\left| D \right|}\log_{2}\dfrac{\left|C_{k}\right|}{\left| D \right|} \end{align*}$$
		2. 计算特征$A$对数据集$D$的经验条件熵$H \left( D | A \right)$
$$\begin{align*} \\ & H \left( D | A \right) = \sum_{i=1}^{n} \dfrac{\left| D_{i} \right|}{\left| D \right|} H \left( D_{i} \right)
\\ & = \sum_{i=1}^{n} \dfrac{\left| D_{i} \right|}{\left| D \right|} \sum_{k=1}^{K} \dfrac{\left| D_{ik} \right|}{\left| D_{i} \right|} \log_{2} \dfrac{\left| D_{ik} \right|}{\left| D_{i} \right|}\end{align*}$$
		3. 计算信息增益
$$\begin{align*} \\ & g \left( D, A \right) = H \left( D \right) - H \left( D | A \right) \end{align*}$$


- 信息增益比

	特征$A$对训练集$D$的信息增益比
$$\begin{align*} \\ & g_{R} \left( D, A \right) = \dfrac{g \left( D, A \right)}{H_{A} \left(D\right)}\end{align*}$$   
即，信息增益$g\left( D, A \right)$与训练数据集$D$关于特征$A$的经验熵$H_{A}\left(D\right)$之比。  
其中，
$$\begin{align*} \\ & H_{A} \left( D \right) = -\sum_{i=1}^{n} \dfrac{\left|D_{i}\right|}{\left|D\right|}\log_{2}\dfrac{\left|D_{i}\right|}{\left|D\right|}\end{align*}$$  
$i$表示特征$A$的取值个数,$|D_i|$表示 $D$ 中 特征取 $A_i$ 的个数
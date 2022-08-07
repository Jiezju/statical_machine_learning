'''
构造平衡kd树算法

输入：𝑘 维空间数据集𝑇＝{𝑥1，𝑥2,…,𝑥𝑁} ，其中𝑥𝑖=(𝑥(1)𝑖,𝑥(2)𝑖,⋯,𝑥(𝑘)𝑖)T ，𝑖＝1,2,…,𝑁

；

输出：kd树。

（1）开始：构造根结点，根结点对应于包含 𝑇 的 𝑘 维空间的超矩形区域。

         选择 𝑥(1) 为坐标轴，以 T 中所有实例的 𝑥(1) 坐标的中位数为切分点，将根结点对应的超矩形区域切分为两个子区域。
         切分由通过切分点并与坐标轴 𝑥(1) 垂直的超平面实现。

         由根结点生成深度为 1 的左、右子结点：左子结点对应坐标 𝑥(1) 小于切分点的子区域， 右子结点对应于坐标 𝑥(1) 大于切分点的子区域。
         将落在切分超平面上的实例点保存在根结点。

（2）重复：对深度为 𝑗 的结点，选择𝑥(1)为切分的坐标轴，𝑙＝𝑗(𝑚𝑜𝑑𝑘)+1，以该结点的区域中所有实例的 𝑥(1) 坐标的中位数为切分点，

         将该结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴 𝑥(1) 垂直的超平面实现。

         由该结点生成深度为 𝑗+1 的左、右子结点：左子结点对应坐标 𝑥(1) 小于切分点的子区域，右子结点对应坐标 𝑥(1) 大于切分点的子区域。

         将落在切分超平面上的实例点保存在该结点。

（3）直到两个子区域没有实例存在时停止。从而形成kd树的区域划分。
'''

from math import sqrt
from collections import namedtuple

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple",
                    "nearest_point  nearest_dist  nodes_visited")

# kd-tree每个结点中主要包含的数据结构如下
# 每个 样本点 作为 节点
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k 维向量节点(k 维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号，实际上记录的是 树 的 层数）
        self.left = left  # 该结点分割超平面左子空间构成的 kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的 kd-tree


# 构建 Kd 树
class KdTree(object):
    def __init__(self, data):
        k = len(data[0])  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集exset创建KdNode
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # //为Python中的整数除法
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates

            # 递归的创建kd树
            return KdNode(
                median,
                split,
                CreateNode(split_next, data_set[:split_pos]),  # 创建左子树
                CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)


def find_nearest(tree, point):
    # 树 的 层数
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"),
                          0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        # ----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


if __name__ == '__main__':
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = KdTree(data)
    preorder(kd.root)
    ret = find_nearest(kd, [3, 4.5])
    print(ret)

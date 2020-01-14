"""
参考地址：
https://blog.csdn.net/tudaodiaozhale/article/details/77327003
https://sine-x.com/statistical-learning-method/#%E7%AC%AC4%E7%AB%A0-%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B3%95

"""

import numpy as np


class KDTree:
    """
    构造kd树
    """

    def __init__(self, data=None):
        self.data = data
        self.l_child = None
        self.r_child = None

    def create(self, data_set, depth=0):
        """
        创建kd树，返回根结点
        :param data_set:
        :param depth:
        :return:
        """
        if len(data_set) > 0:
            m, n = np.shape(data_set)  # 求出样本行，列
            midIndex = int(m / 2)  # 中间数的索引位置
            axis = depth % n  # 判断以哪个轴划分数据

            # 按指定列排序
            ind = list(range(n))

            ind[axis] = n - 1
            ind[-1] = axis
            data_array = np.array(data_set)
            # 排序后的列表
            sorted_dataSet = data_array[np.lexsort(data_array[:, ind].T)].tolist()

            # 将节点数据域设置为中位数，具体参考下书本
            node = KDTree(sorted_dataSet[midIndex])

            left_dataSet = sorted_dataSet[: midIndex]
            right_dataSet = sorted_dataSet[midIndex + 1:]
            # print(leftDataSet)
            # print(rightDataSet)

            node.l_child = self.create(left_dataSet, depth + 1)
            node.r_child = self.create(right_dataSet, depth + 1)
            return node
        else:
            return None

    def preOrder(self, node):
        """
        遍历kd树
        :param node:
        :return:
        """
        if node is not None:
            print("NODE->%s" % node.data)
            self.preOrder(node.l_child)
            self.preOrder(node.r_child)

    def dist(self, x1, x2):
        """
        欧式距离计算
        :param x1:
        :param x2:
        :return:
        """
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5

    def search_knn(self, tree, x, k):
        """
        k近邻
        :param tree:
        :param x:
        :param k:
        :return:
        """

        kn_point = []
        kn_dist = []

        def travel(node, depth=0):  # 递归搜索

            # 特征个数
            n = len(x)
            # 计算划分的轴
            axis = depth % n

            if node is not None:  # 递归终止条件
                if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
                    travel(node.l_child, depth + 1)
                else:
                    travel(node.r_child, depth + 1)

                # 递归完毕后，往父结点方向回朔，对应算法3.3(3)
                distance = self.dist(x, node.data)  # 目标和节点的距离
                if len(kn_point) < k:  # 维护一个长度为k的列表
                    kn_point.append(node.data)
                    kn_dist.append(distance)
                elif max(kn_dist) > distance:  # 列表中最大的距离，小于当前距离
                    ind = kn_dist.index(max(kn_dist))
                    kn_point[ind] = node.data
                    kn_dist[ind] = distance

                # print(node.data, self.kn_dist, node.data[axis], x[axis])
                if abs(x[axis] - node.data[axis]) <= max(kn_dist):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                    if x[axis] < node.data[axis]:
                        travel(node.r_child, depth + 1)
                    else:
                        travel(node.l_child, depth + 1)

        travel(tree)
        return kn_point


# class KDSearch():
#
#     def __init__(self, tree=None, x=None, k=2):
#         self.tree = tree
#         self.x = x  # 待搜索的点
#         # 最近邻
#         self.nearest_point = None  # 保存最近的点
#         self.nearest_dist = 0  # 保存最近的值
#
#         # k近邻
#         self.k = k
#         self.kn_point = []
#         self.kn_dist = []
#
#     def dist(self, x1, x2):
#         """
#         欧式距离计算
#         :param x1:
#         :param x2:
#         :return:
#         """
#         return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5
#
#     def search_nn(self, node, depth=0):  # 递归搜索
#
#         # 特征个数
#         n = len(self.x)
#         # 计算划分的轴
#         axis = depth % n
#
#         if node is not None:  # 递归终止条件
#             if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
#                 self.search_nn(node.l_child, depth + 1)
#             else:
#                 self.search_nn(node.r_child, depth + 1)
#
#             # 递归完毕后，往父结点方向回朔，对应算法3.3(3)
#             distance = self.dist(x, node.data)  # 目标和节点的距离判断
#             if self.nearest_point is None:  # 确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
#                 self.nearest_point = node.data
#                 self.nearest_dist = distance
#             elif self.nearest_dist > distance:
#                 self.nearest_point = node.data
#                 self.nearest_dist = distance
#
#             print(node.data, self.nearest_dist, node.data[axis], x[axis])
#             if abs(x[axis] - node.data[axis]) <= self.nearest_dist:  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
#                 if x[axis] < node.data[axis]:
#                     self.search_nn(node.r_child, depth + 1)
#                 else:
#                     self.search_nn(node.l_child, depth + 1)
#
#         return self.nearest_point
#
#     def search_kn(self, node, depth=0):  # 递归搜索
#
#         # 特征个数
#         n = len(self.x)
#         # 计算划分的轴
#         axis = depth % n
#
#         if node is not None:  # 递归终止条件
#             if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
#                 self.search_kn(node.l_child, depth + 1)
#             else:
#                 self.search_kn(node.r_child, depth + 1)
#
#             # 递归完毕后，往父结点方向回朔，对应算法3.3(3)
#             distance = self.dist(x, node.data)  # 目标和节点的距离
#             if len(self.kn_point) < self.k:
#                 self.kn_point.append(node.data)
#                 self.kn_dist.append(distance)
#             elif max(self.kn_dist) > distance:
#                 ind = self.kn_dist.index(max(self.kn_dist))
#                 self.kn_point[ind] = node.data
#                 self.kn_dist[ind] = distance
#
#             # print(node.data, self.kn_dist, node.data[axis], x[axis])
#             if abs(x[axis] - node.data[axis]) <= max(self.kn_dist):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
#                 if x[axis] < node.data[axis]:
#                     self.search_kn(node.r_child, depth + 1)
#                 else:
#                     self.search_kn(node.l_child, depth + 1)
#
#         return self.kn_point


if __name__ == '__main__':
    dataSet = [[4, 3],
               [2, 3],
               [5, 4],
               [9, 6],
               [4, 7],
               [8, 1],
               [7, 2]]
    x = [5, 3]

    kdtree = KDTree()
    tree = kdtree.create(dataSet)

    kdtree.preOrder(tree)
    print(kdtree.search_knn(tree, [5, 3], 1))

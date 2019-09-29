import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(6, 4))

# # cmap用cubehelix map颜色
# # cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
# pt = np.array([2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4]).reshape(4, 4)
# # sns.heatmap(pt, linewidths=0.05, ax=ax1, vmax=5, vmin=0, cmap=cmap)
# # ax1.set_title('cubehelix map')
# # ax1.set_xlabel('')
# # ax1.set_xticklabels([])  # 设置x轴图例为空值
# # ax1.set_ylabel('kind')
#
# # cmap用matplotlib colormap
# sns.heatmap(pt, linewidths=0.05, vmax=5, vmin=0, cmap='rainbow')
# # rainbow为 matplotlib 的colormap名称
# # plt.set_title('matplotlib colormap')
# # plt.set_xlabel('region')
# # plt.set_ylabel('kind')
# plt.show()


a = [1, 3, 4, 3, 2, 3]
b = [3, 4, 1, 2, 3, 1]
c = [3, 4, 1, 3, 3, 6]
plt.plot(a, label='a')
plt.plot(b, label='b')
plt.plot(c, label='c')
plt.show()

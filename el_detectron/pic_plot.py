import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(6, 4))

# pt = np.array([2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4]).reshape(4, 4)
pt = np.zeros((500, 500)) + 10
# sns.heatmap(pt, linewidths=0.05, vmax=15, vmin=0, cmap='rainbow')
# plt.show()


# a = [1, 3, 4, 3, 2, 3]
# b = [3, 4, 1, 2, 3, 1]
# c = [3, 4, 1, 3, 3, 6]
# plt.plot(a, label='a')
# plt.plot(b, label='b')
# plt.plot(c, label='c')
# plt.show()

plt.pcolor(pt, cmap=plt.cm.Reds)
plt.show()
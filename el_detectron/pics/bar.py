import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

x = [1, 2, 3, 4, 5]
# x = ["0.9", "0.92", "0.94", "0.96", "0.98"]
cmap1 = plt.get_cmap("Pastel1")

y1 = np.array([2, 2, 2, 2, 3])
y2 = np.array([3, 1, 0, 2, 0])
y3 = np.array([2, 2, 2, 2, 3])
plt.bar(x, y1, align="center", tick_label=["0.9", "0.92", "0.94", "0.96", "0.98"], color=cmap1(4), label="miss")
plt.bar(x, y2, align="center", bottom=y1, color=cmap1(1), label="overkill")
plt.bar(x, y3, align="center", bottom=y1 + y2, color=cmap1(2), label="ng")
plt.plot(range(-1, 7), [2] * 8, 'r--', label="aa")

plt.xlabel("置信度")
plt.ylabel("个数")
plt.xlim(0, 6)

plt.legend()

plt.show()

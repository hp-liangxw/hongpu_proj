import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

"""
https://matplotlib.org/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
https://matplotlib.org/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
https://matplotlib.org/tutorials/colors/colormaps.html?highlight=pastel1
"""

plt.rcParams['font.family'] = 'STSong'

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 6, figure=fig)

ax1 = fig.add_subplot(gs[0, :-1])
ax2 = fig.add_subplot(gs[0, -1])

cmap1 = plt.get_cmap("Pastel1")
cmap2 = plt.get_cmap("tab20c")

vals1 = [40, 80, 120, 30, 20]
vals2 = [40, 80, 120, 50]
vals3 = [1]

ax1.pie(vals1, labels=["", 80, 120, 30, 20], colors=cmap1([19, 0, 1, 0, 1]), labeldistance=0.75, radius=1,
        wedgeprops=dict(edgecolor='w'))
ax1.pie(vals2, labels=[40, 40, 90, 10], colors=cmap2([11, 11, 11, 11]), labeldistance=0.6, radius=0.66,
        wedgeprops=dict(edgecolor='w'))
ax1.pie(vals3, radius=0.33, colors='w')

ax2.hlines(0.5, 0.7, 0.9, color=cmap2(11), linewidth=12)
ax2.text(0.95, 0.5 - 0.01, "组件", fontsize=10)
ax2.hlines(0.45, 0.7, 0.9, colors=cmap1(19), linewidth=12)
ax2.text(1, 0.45 - 0.01, "无缺陷", fontsize=10)
ax2.hlines(0.4, 0.7, 0.9, colors=cmap1(0), linewidth=12)
ax2.text(1, 0.4 - 0.01, "yinlie", fontsize=10)
ax2.hlines(0.35, 0.7, 0.9, colors=cmap1(1), linewidth=12)
ax2.text(1, 0.35 - 0.01, "xuhan", fontsize=10)

ax2.yaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax2.set_axis_off()
ax2.set_xlim([0.7, 1])
ax2.set_ylim([0, 0.9])


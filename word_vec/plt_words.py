import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl


zhfont = matplotlib.font_manager.FontProperties(fname=r'C:\Nuance\python_env\basic_dev\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\msyh.ttf')
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# x = np.arange(0, 5, 0.1)
# # y = 2 * x
# # plt.plot(x, y)

print(matplotlib.matplotlib_fname())
plt.xlabel(u"横轴")
# fig = plt.figure()
# ax = fig.add_subplot(111)
x = np.arange(-5, 5, 0.1)
y = x
plt.plot(x, y)
plt.text(1.33333, -0.5222, '你好', color='b', fontproperties=zhfont)

# plt.axis([40, 160, 0, 0.03])
plt.show()
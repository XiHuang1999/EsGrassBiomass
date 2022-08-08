# -*- coding: utf-8 -*-

# @Time : 2022-08-08 11:22
# @Author : XiHuang O.Y.
# @Site : 
# @File : DrawingPIC_MeanAndDistribution.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 读取CN-border-La.dat文件
with open('D:/绘制地图/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
# 设置画图各种参数
fig = plt.figure(figsize=[8, 8])
# 设置投影类型和经纬度
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=90,
                                               central_longitude=105))
# 画海，陆地，河流，湖泊
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.RIVERS.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'))
# 画国界
for line in borders:
    ax.plot(line[0::2], line[1::2], '-', color='gray', transform=ccrs.Geodetic())
# 画经纬度网格
ax.gridlines(linestyle='--')
# 框出区域
ax.set_extent([80, 140, 13, 55])
# 画南海，这一步是新建一个ax，设置投影
sub_ax = fig.add_axes([0.741, 0.11, 0.14, 0.155],
                      projection=ccrs.LambertConformal(central_latitude=90,
                                                       central_longitude=115))
# 画海，陆地，河流，湖泊
sub_ax.add_feature(cfeature.OCEAN.with_scale('50m'))
sub_ax.add_feature(cfeature.LAND.with_scale('50m'))
sub_ax.add_feature(cfeature.RIVERS.with_scale('50m'))
sub_ax.add_feature(cfeature.LAKES.with_scale('50m'))
# 画边界
for line in borders:
    sub_ax.plot(line[0::2], line[1::2], '-', color='gray',
                transform=ccrs.Geodetic())
# 框区域
sub_ax.set_extent([105, 125, 0, 25])
# 显示
plt.show()
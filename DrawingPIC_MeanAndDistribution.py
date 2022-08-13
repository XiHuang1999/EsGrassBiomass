# -*- coding: utf-8 -*-

# @Time : 2022-08-08 11:22
# @Author : XiHuang O.Y.
# @Site : 
# @File : DrawingPIC_MeanAndDistribution.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader, natural_earth
import numpy as np
from osgeo import gdal
from osgeo import osr
from cartopy.crs import epsg
from matplotlib.image import imread
from scipy.interpolate import make_interp_spline

def getSRSPair(dataset):
    '''
    得到给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据肯定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2] #关于coords = ct.TransformPoint(px,py)的介绍:coords是一个Tuple类型的变量包含3个元素，coords[0]为纬度，coords[1]为经度，coords[2]为高度

def read_img(filename):
    """
    filename: str,要读取的栅格文件
    return: 函数输入图像所在的路径，输出图像的投影，仿射矩阵以及图像数组\n
    """
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    # # If you have projection coordinates, you need to run the following two lines
    # startY, startX = geo2lonlat(dataset,im_geotrans[0],im_geotrans[3])
    # endY, endX = geo2lonlat(dataset,im_geotrans[0] + im_geotrans[1] * im_data.shape[1],im_geotrans[3] + im_geotrans[5] * im_data.shape[0])
    #
    # # Calculate Lon Lat Grid
    # Lons = np.linspace(start=startX,    # 左上角x坐标
    #                    stop=endX,       # 右下角x坐标 + 东西方向像素分辨率 * 列
    #                    num=im_data.shape[1],
    #                    endpoint=True)
    # Lats = np.linspace(start=startY,    # 左上角y坐标
    #                    stop=endY,       # 右下角y坐标 + 南北方向像素分辨率 * 行
    #                    num=im_data.shape[0],
    #                    endpoint=True)
    # Lons, Lats = np.meshgrid(Lons, Lats)  # 构建经纬网
    return dataset,im_proj,im_geotrans,im_data

def smooth_xy(lx, ly):
    """数据平滑处理
    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth

# region Parameters
# tif相关参数
tif = r'G:\1_BeiJingUP\CommonData\temp\111Pro.tif'
CHNshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\全国无子区域.shp'
NHshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\南沙群岛海上国境线.shp'
Alberts_China = ccrs.AlbersEqualArea(central_longitude=110, central_latitude=0, standard_parallels=(25, 47))

tifRange = [0,300]
colorBarSectionsNum = 16

# fig相关参数
row,col = 4,4
startLon = 70
endLon = 138
startLat = 15
endLat = 50
extents = [startLon, endLon, startLat, endLat]  # [-180,180,-90,90]#
res = '50m' # BaceMap Resolution
geo = ccrs.Geodetic()
projAlb = ccrs.AlbersEqualArea()
projPlc = ccrs.PlateCarree() #central_longitude=110
projMct = ccrs.Mercator()
# endregion

# region Drawing
fig = plt.figure(figsize=(10,7.5))  # 创建Figure对象
gs = fig.add_gridspec(row,col) # 添加子图
# endregion

# region Main Pic
ax1 = fig.add_subplot(gs[1:, 0:col-1], projection=projPlc)  # 通过添加projection参数 创建geoaxes对象
# ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
gl.top_labels = False
gl.right_labels = False
# ax1.stock_img()
ax1.add_feature(cfeature.OCEAN.with_scale(res))
ax1.add_feature(cfeature.LAND.with_scale(res), edgecolor='gray')
ax1.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
ax1.add_feature(cfeature.RIVERS.with_scale(res))
# ax1.add_feature(cfeature.BORDERS.with_scale(res))

# 栅格
ds,img_proj,img_geotrans,img_data = read_img(tif)
img_data[img_data<0] = np.nan
# SHP
CHN = cfeat.ShapelyFeature(Reader(CHNshp).geometries(),projPlc, edgecolor='k', facecolor='none')
HN = cfeat.ShapelyFeature(Reader(NHshp).geometries(),projPlc, edgecolor='k',facecolor='none')

clevs = np.linspace(tifRange[0], tifRange[1], colorBarSectionsNum)        # 对颜色进行分段
# 栅格可视化，同时根据像元值设置图片颜色
# Rs = ax1.contourf(geoGrid[0], geoGrid[1], img_data, transform=Alberts_China, cmap=plt.cm.jet, levels=clevs, extend='both') #clevs,#transform=Alberts_China,cmap=plt.cm.jet,zorder=10)
tifExtent = geo.transform_points(projPlc,
                                np.array([img_geotrans[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1]]),
                                np.array([img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[3]]))
tifExtent = (tifExtent[0][0],tifExtent[1][0],tifExtent[0][1],tifExtent[1][1])
# Or use tifExtent = (img_geotrans[0], img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1], img_geotrans[3])

ax1.imshow(img_data,extent=tifExtent, origin='upper',transform=projPlc)
# ax1.add_feature(CHNshp, linewidth=2,alpha=0.9, zorder=1)
# ax1.add_feature(NHshp, linewidth=2,alpha=0.9, zorder=1)
ax1.set_extent(extents)     # ax1.set_global()
plt.show()

# 绘制色带
# cbar = fig.colorbar(ax1, orientation='vertical',
#                     pad=0.08, aspect=20, shrink=0.65)
# cbar.set_ticks([int(i) for i in np.linspace(tifRange[0], tifRange[1], 11)])
# cbar.set_label('AGB Value (gDM/$^{-2}$)') #我的数据没有负值，所以快乐值都是正值，祝见者天天开心

# #设置南海子图的坐标
# left, bottom, width, height = 0.67, 0.14, 0.21, 0.23
# axNH = fig.add_axes(
#     [left, bottom, width, height],
#     projection=projMct
# )
# #添加南海子图的详细内容
# axNH.add_feature(CHN, linewidth=2,alpha=0.9, zorder=1)
# axNH.add_feature(NH, linewidth=2,alpha=0.9, zorder=1)
# axNH.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2,zorder=1)    #加载分辨率为50的海岸线
# axNH.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=2,zorder=1)       #加载分辨率为50的河流
# axNH.add_feature(cfeature.LAKES.with_scale('50m'), zorder=1)                    #加载分辨率为50的湖泊
#
# #设置南海子图的坐标范围，和边框linewidth
# axNH.set_extent([105, 125, 0, 25])
# axNH.outline_patch.set_linewidth(2)

# endregion

# region top Pic
ax2 = fig.add_subplot(gs[0, 0:col-1])
ax2.plot([1, 2], [3, 4])
# endregion

# region Right Pic
vx = np.linspace(start=startLat+111*tifExtent[2],stop=startLat+111*tifExtent[2]+img_data.shape[0], num=img_data.shape[0], endpoint=False)
vm = np.nanmean(img_data,axis=1)
vm[np.isnan(vm)]=0
vx, vm = smooth_xy(vx, vm)
ax3 = fig.add_subplot(gs[1:, col-1])
ax3.plot(vm,vx)
# ax3.set_ylim([1,2])
# endregion

# region Adjust Pic Show
# plt.subplots_adjust(left=0.01, right=1, bottom=0.01, top=1, wspace=0.35, hspace=0.35)
plt.subplots_adjust(wspace=0.5, hspace=0.6)
plt.tight_layout()
# endregion


plt.show()
print()

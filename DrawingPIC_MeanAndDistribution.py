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
import matplotlib.ticker as mticker
import os
from tqdm import tqdm

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
    x_smooth = np.linspace(x.min(), x.max(), 140)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth

# region Parameters
# tif相关参数
tif = r'G:\1_BeiJingUP\CommonData\temp\111Pro.tif'
CHNshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\全国无子区域.shp'
NHshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\南沙群岛海上国境线.shp'
Alberts_China = ccrs.AlbersEqualArea(central_longitude=110, central_latitude=0, standard_parallels=(25, 47))

tifRange = [0,150]
colorBarSectionsNum = 10
cbar = 'Spectral'

# fig相关参数
row,col = 4,4
startLon = 70
endLon = 138
startLat = 15
endLat = 50
res = '50m' # BaceMap Resolution
moveWindow = 3
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rc('font',family='Times New Roman')
extents = (startLon, endLon, startLat, endLat)  # [-180,180,-90,90]#
geo = ccrs.Geodetic()
projAlb = ccrs.AlbersEqualArea()
projPlc = ccrs.PlateCarree() #central_longitude=110
projMct = ccrs.Mercator()
# endregion

# region Drawing
fig = plt.figure(figsize=(9.6,7.5))  # 创建Figure对象
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
ax1.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke', edgecolor='gray')
ax1.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
ax1.add_feature(cfeature.RIVERS.with_scale(res))
# ax1.add_feature(cfeature.BORDERS.with_scale(res))

# 栅格
ds,img_proj,img_geotrans,img_data = read_img(tif)
img_data[img_data<0] = np.nan
# SHP
CHN = cfeat.ShapelyFeature(Reader(CHNshp).geometries(),projPlc, edgecolor='k', facecolor='none')
NH = cfeat.ShapelyFeature(Reader(NHshp).geometries(),projPlc, edgecolor='k',facecolor='none')

# 栅格可视化，同时根据像元值设置图片颜色
# Rs = ax1.contourf(geoGrid[0], geoGrid[1], img_data, transform=Alberts_China, cmap=plt.cm.jet, levels=clevs, extend='both') #clevs,#transform=Alberts_China,cmap=plt.cm.jet,zorder=10)
tifExtent = geo.transform_points(projPlc,
                                np.array([img_geotrans[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1]]),
                                np.array([img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[3]]))
tifExtent = (tifExtent[0][0],tifExtent[1][0],tifExtent[0][1],tifExtent[1][1])
# Or use tifExtent = (img_geotrans[0], img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1], img_geotrans[3])

imgtif = ax1.imshow(img_data, extent=tifExtent, origin='upper',transform=projPlc,cmap=cbar,vmin=tifRange[0], vmax=tifRange[1]+tifRange[1]/colorBarSectionsNum,aspect=1.3)
ax1.add_feature(CHN, linewidth=2,alpha=0.9, zorder=1)
ax1.add_feature(NH, linewidth=2,alpha=0.9, zorder=1)
ax1.set_extent(extents)     # ax1.set_global()




#设置南海子图的坐标
left, bottom, width, height = 0.55, 0.077, 0.21, 0.23
axNH = fig.add_axes(
    [left, bottom, width, height],
    projection=projMct
)
#添加南海子图的详细内容
axNH.add_feature(CHN, linewidth=1.5,alpha=0.9, zorder=1)
axNH.add_feature(NH, linewidth=1.5,alpha=0.9, zorder=1)
axNH.add_feature(cfeature.OCEAN.with_scale(res))
axNH.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke', edgecolor='gray')
axNH.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
axNH.add_feature(cfeature.RIVERS.with_scale(res))

#设置南海子图的坐标范围，和边框linewidth
axNH.set_extent([105, 125, 0, 25])
axNH.outline_patch.set_linewidth(2)

# endregion

# region top Pic
moveWindow = 111
vx = np.linspace(start=tifExtent[0],stop=tifExtent[1], num=img_data.shape[1], endpoint=False)
vx = vx[int(moveWindow/2):img_data.shape[1]-int(moveWindow/2)-1]
# vm = np.nanmean(img_data,axis=1)
# vm[np.isnan(vm)]=0
# vx, vm = smooth_xy(vx, vm)
vm = np.array([])
mstd_up = np.array([])
mstd_bot = np.array([])
for i in tqdm(range(moveWindow-1,img_data.shape[1],1),desc='Processing'):
    vm = np.append(vm, np.nanmean(img_data[:,i-moveWindow:i]))
    mstd_up = np.append(mstd_up, np.nanmean(img_data[:,i-moveWindow:i]) + np.nanstd(img_data[:,i-moveWindow:i]) )
    mstd_bot = np.append(mstd_bot, np.nanmean(img_data[:,i-moveWindow:i]) - np.nanstd(img_data[:,i-moveWindow:i]) )
vm[np.isnan(vm)]=0
sid = max(np.where(vm != 0)[0][0],np.where(vx>73)[0][0])
eid = min(np.where(vm != 0)[0][-1],np.where(vx<138)[0][-1])
vm = vm[sid:eid+1-moveWindow]
mstd_up = mstd_up[sid:eid+1-moveWindow]
mstd_bot = mstd_bot[sid:eid+1-moveWindow]
vx = vx[sid:eid+1-moveWindow]
# vx, vm = smooth_xy(vx, vm)

ax2 = fig.add_subplot(gs[0, 0:col-1])
ax2.plot(vx,vm,linewidth=2,c=r'Black')
ax2.fill_between(vx,mstd_bot,mstd_up, color='skyblue')
# ax2.xaxis.set_label_position("top")
ax2.set_xlim(70,endLon)
ax2.set_ylim(0, 150)
ax2.tick_params("both", which='major', direction='in')
ax2.set_xlabel(r"Longitude ($^\circ$)",ha="center")
ax2.set_ylabel('Mean value of AGB')
# endregion

# region Right Pic
# moveWindow = 3
# vx = np.linspace(start=tifExtent[2],stop=tifExtent[3], num=img_data.shape[0], endpoint=False)
# # vm = np.nanmean(img_data,axis=1)
# # vm[np.isnan(vm)]=0
# # vx, vm = smooth_xy(vx, vm)
# vm = np.array([])
# for i in range(img_data.shape[0],moveWindow-1,-1):
#     vm = np.append(vm, np.nanmean(np.nanmean(img_data[i-moveWindow:i,:], axis=1)) )
# vm[np.isnan(vm)]=0
# sid = np.where(vx>20)[0][0]
# eid = np.where(vx<55)[0][-1]
# vm = vm[sid:eid+1-moveWindow]
# vx = vx[sid:eid+1-moveWindow]
# vx, vm = smooth_xy(vx, vm)
# ax3 = fig.add_subplot(gs[1:, col-1])
# ax3.plot(vm,vx,linewidth=2,c=r'Black')
# ax3.set_ylim(startLat,55)
# ax3.yaxis.set_label_position("right")
# ax3.tick_params("both", which='major', direction='in')
moveWindow = 111
vx = np.linspace(start=tifExtent[2],stop=tifExtent[3], num=img_data.shape[0], endpoint=False)
vx = vx[int(moveWindow/2):img_data.shape[0]-int(moveWindow/2)-1]
# vm = np.nanmean(img_data,axis=1)
# vm[np.isnan(vm)]=0
# vx, vm = smooth_xy(vx, vm)
vm = np.array([])
mstd_up = np.array([])
mstd_bot = np.array([])
for i in tqdm(range(img_data.shape[0],moveWindow,-1),desc='Process2'):
    vm = np.append(vm, np.nanmean(img_data[i - moveWindow:i, :]))
    mstd_up = np.append(mstd_up, np.nanmean(img_data[i - moveWindow:i, :]) + np.nanstd(img_data[i - moveWindow:i, :]) )
    mstd_bot = np.append(mstd_bot, np.nanmean(img_data[i - moveWindow:i, :]) - np.nanstd(img_data[i - moveWindow:i, :]) )

vm[np.isnan(vm)]=0
sid = max(np.where(vm != 0)[0][0],np.where(vx>=20)[0][0])
eid = min(np.where(vm != 0)[0][-1],np.where(vx<=55)[0][-1])
vm = vm[sid:eid+1-moveWindow]
mstd_up = mstd_up[sid:eid+1-moveWindow]
mstd_bot = mstd_bot[sid:eid+1-moveWindow]
vx = vx[sid:eid+1-moveWindow]
# vx, vm = smooth_xy(vx, vm)
ax3 = fig.add_subplot(gs[1:, col-1])
ax3.plot(vm,vx,linewidth=2,c=r'Black')
ax3.fill_betweenx(vx,mstd_bot,mstd_up, color='skyblue')

ax3.set_ylim(startLat,endLat)
# ax3.yaxis.set_label_position("right")
ax3.tick_params("both", which='major', direction='in')
ax3.set_xlim(0, 150)
ax3.set_ylim(startLat,55)
# ax3.set_title('Year: '+str(yr))
ax3.set_ylabel(r"Latitude ($^\circ$)",ha="center")
ax3.set_xlabel('Mean value of AGB')
# endregion



# region ColorBar
# 绘制色带1
# cbar = plt.colorbar(imgtif, orientation='horizontal',
#                     pad=0.08, aspect=20, shrink=0.65)
# cbar.set_ticks([int(i) for i in np.linspace(tifRange[0], tifRange[1], 11)])
# cbar.set_label('AGB Value (gDM/$^{-2}$)') #我的数据没有负值，所以快乐值都是正值，祝见者天天开心
# 绘制色带2
# axBar=fig.add_axes([0,0.17,0.5,0.05])
# fig.colorbar(imgtif,orientation='horizontal',cax=ax1)   #,cax=axBar
# 绘制色带3
# position = fig.add_axes([0.66,0.13,0.03,0.74])#添加子图用来存放色条
# cb = fig.colorbar(imgtif,shrink=0.4, fraction=0.05, ax = ax1)#,cax=ax3 )#绘制colorbar并省称为cb
# axBar = cb.ax#召唤出cb的ax属性并省称为ax2,这时ax2即视为一个子图
# axBar.yaxis.set_ticks_position('right')#将数值刻度移动到左侧
# axBar.tick_params(which='both',labelsize=10,left=True,direction='in')#修改刻度样式，并使左右都有刻度
# axBar.yaxis.set_minor_locator(mticker.MultipleLocator(20))
# 绘制色带4
from matplotlib import cm
clevs = np.linspace(tifRange[0], tifRange[1], colorBarSectionsNum+1)        # 对颜色进行分段
rotateBar = np.pi/2+np.pi/len(clevs)
axBar=fig.add_axes([0.09,0.1,0.15,0.15],projection='polar')#添加子图用来存放色条
cmap=cm.get_cmap(cbar,len(clevs)) #获得等值线填色图的色条对应分级
#划分极坐标系中的x坐标
angle=np.arange(0,2*np.pi,2*np.pi/(len(clevs)))
#划分极坐标系中的y坐标，由于我们要使cbar对其，所以高度都取2
radius=np.array([6]*(len(clevs)))
cmaps=cmap(range(len(clevs)))
#设置旋转偏移
axBar.set_theta_offset(rotateBar)
#绘制极坐标中的bar
axBar.bar(angle,radius,width=2*np.pi/(len(clevs)-1),color=cmaps,align='center')
# axBar.set_rmax(2)
# axBar.set_rmin(0)
axBar.set_rlim(0,2)
axBar.set_ylim(-2,2)

# axBar.set_rticks(np.arange(0, 2, 1))
# axBar.xaxis.set_major_locator(mticker.MultipleLocator(360/len(clevs)))
# axBar.gridlines(draw_labels=True,x_inline=clevs, y_inline=True) #添加网格线
# cb=fig.colorbar(imgtif,cax=position,shrink=0.4)#绘制colorbar并省称为cb
# axBar=cb.ax#召唤出cb的ax属性并省称为ax2,这时ax2即视为一个子图
# 标注文字
for i,x,y in zip(clevs,angle,radius):
    if i < 60:
        pad = 0
    else:
        pad = 0.3
    # axBar.text(x-np.pi/len(clevs)-rotateBar,2+0.8+pad,int(i),fontsize=10,horizontalalignment='center')       # axBar.xaxis.set_ticklabels([int(i) for i in clevs])   OR   # axBar.set_thetagrids(np.arange(0.0+180/len(clevs), 360.0+180/len(clevs), 360/len(clevs)), labels=[int(i) for i in np.append(clevs[1:],[0])])
    axBar.text(x-np.pi/len(clevs),2+0.5+pad,int(i),fontsize=10,horizontalalignment='center')       # axBar.xaxis.set_ticklabels([int(i) for i in clevs])   OR   # axBar.set_thetagrids(np.arange(0.0+180/len(clevs), 360.0+180/len(clevs), 360/len(clevs)), labels=[int(i) for i in np.append(clevs[1:],[0])])
axBar.text(np.pi-np.pi/len(clevs),-1,'\nAGB\nValue',fontsize=12,horizontalalignment='center')
axBar.yaxis.set_visible(False)
axBar.xaxis.grid(False)
axBar.set_xticklabels('')
# endregion


# region Adjust Pic Show
# plt.subplots_adjust(left=0.01, right=1, bottom=0.01, top=1, wspace=0.35, hspace=0.35)
plt.subplots_adjust(wspace=0.5, hspace=0.6)
plt.tight_layout()
# endregion

# plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PIC'+os.sep+'RF4_Distribution.png',dpi=500,bbox_inches='tight')#, transparent=True
plt.show()
print()

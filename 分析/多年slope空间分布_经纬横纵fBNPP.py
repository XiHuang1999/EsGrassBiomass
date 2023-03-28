#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time ： 2023/2/27 01:48:10
# @IDE ：PyCharm
# @Author : Xihuang Ouyang

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
from datetime import datetime
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mticker
import os
from tqdm import tqdm
import matplotlib as mpl

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


# -----------函数：添加比例尺--------------
def add_scalebar(lon0, lat0, length, ax):
    """
    绘制比例尺/添加x轴y轴标题
    :param lon0: float, 左下角经度
    :param lat0: float, 左下角纬度
    :param length: float,  比例尺长度
    :param ax: axis, 需要绘制的轴画布
    :return: None
    """
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)  # ax.transData
    ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length / 111, colors="black", ls="-", lw=1, label='%d km' % (length),transform=transform)
    ax.vlines(x=lon0, ymin=lat0 - 0.45, ymax=lat0 + 0.45, colors="black", ls="-", lw=1,transform=transform)
    ax.vlines(x=lon0 + length / 2 / 111, ymin=lat0 - 0.45, ymax=lat0 + 0.45, colors="black", ls="-", lw=1,transform=transform)
    ax.vlines(x=lon0 + length / 111, ymin=lat0 - 0.45, ymax=lat0 + 0.45, colors="black", ls="-", lw=1,transform=transform)
    # 坐标转化 # lon1, lat1 = ax.transData.inverted().transform((lon0, lat0)) 或使用transform参数
    ax.text(lon0 + length / 111, lat0 + 0.6, '    %d km' % (length), horizontalalignment='center',transform=transform)
    ax.text(lon0 + length / 2 / 111, lat0 + 0.6, '%d' % (length / 2), horizontalalignment='center',transform=transform)
    ax.text(lon0, lat0 + 0.6, r'0', horizontalalignment='center',transform=transform)
    # ax.text(104, 11, r'Longitude', horizontalalignment='center',transform=transform)
    # ax.text(69, 25, r'Latitude', rotation=90, horizontalalignment='center',transform=transform)


# region Parameters
# tif相关参数
tif = r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\fBNPP_2000to2018_slope_Proj.tif'#r'G:\1_BeiJingUP\CommonData\temp\111Pro.tif'
CHNshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\全国无子区域.shp'
NHshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\南沙群岛海上国境线.shp'
Alberts_China = ccrs.AlbersEqualArea(central_longitude=110, central_latitude=0, standard_parallels=(25, 47))

tifRange = [-0.2,0.2]
colorBarSectionsNum = 8
cbar = 'RdYlBu'#'jet'##'Spectral'

# fig相关参数
row,col = 4,4
startLon = 72
endLon = 138
startLat = 15
endLat = 50
res = '50m' # BaceMap Resolution
moveWindow = 3
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size'] = 10
plt.tick_params(pad=0.1)
extents = (startLon, endLon, startLat, endLat)  # [-180,180,-90,90]#
geo = ccrs.Geodetic()
projAlb = ccrs.AlbersEqualArea()
projPlc = ccrs.PlateCarree() #central_longitude=110
projMct = ccrs.Mercator()
# endregion

# region Drawing
fig = plt.figure(figsize=(18/2.54,14.8/2.54))  # 创建Figure对象
gs = fig.add_gridspec(row,col) # 添加子图
# endregion

# region Main Pic
ax1 = fig.add_subplot(gs[1:, 0:col-1], projection=projPlc)  # 通过添加projection参数 创建geoaxes对象
# ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='dashed', draw_labels=True,linewidth=0.3)
gl.top_labels = False
gl.right_labels = False
# ax1.stock_img()
ax1.add_feature(cfeature.OCEAN.with_scale(res))
ax1.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke')#, edgecolor='gray'
# ax1.add_feature(cfeature.LAKES.with_scale(res))
# ax1.add_feature(cfeature.RIVERS.with_scale(res))
# ax1.add_feature(cfeature.BORDERS.with_scale(res))

# 栅格
ds,img_proj,img_geotrans,img_data = read_img(tif)
img_data[img_data<-9999] = np.nan
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
bins = [-0.08,-0.01,-0.005,0,0.005,0.01,0.08]
nbin = len(bins) - 1
cmaps = mpl.cm.get_cmap(cbar, nbin)
norms = mpl.colors.BoundaryNorm(bins, nbin)
imgtif = ax1.imshow(img_data, extent=tifExtent, origin='upper',transform=projPlc,
                    vmin=tifRange[0], vmax=tifRange[1]+(tifRange[1]-tifRange[0])/colorBarSectionsNum,
                    cmap=cmaps,norm=norms,aspect=1.35) #

axbar=fig.add_axes([0.089,0.078,0.02,0.2])
cb = fig.colorbar(imgtif,cax=axbar,fraction=0.05,aspect=20,shrink=0.5,
                  orientation='vertical') #,pad=0.05label='fBNPP'
# cb.set_label(label='fBNPP',loc='top') #loc参数 (×$\mathregular{10^5}$)
cb.ax.set_title('fBNPP')

#设置basemap
ax1.add_feature(CHN, linewidth=1,alpha=0.9, zorder=1)
ax1.add_feature(NH, linewidth=1,alpha=0.9, zorder=1)
ax1.set_extent(extents)     # ax1.set_global()
#设置南海子图的坐标
left, bottom, width, height = 0.578, 0.063, 0.18, 0.2
axNH = fig.add_axes(
    [left, bottom, width, height],
    projection=projMct
)
#添加南海子图的详细内容
imgtif = axNH.imshow(img_data, extent=tifExtent, origin='upper',transform=projPlc,
                    vmin=tifRange[0], vmax=tifRange[1]+(tifRange[1]-tifRange[0])/colorBarSectionsNum,
                    cmap=cmaps,norm=norms) #
axNH.add_feature(CHN, linewidth=1,alpha=0.9, zorder=1)
axNH.add_feature(NH, linewidth=1,alpha=0.9, zorder=1)
axNH.add_feature(cfeature.OCEAN.with_scale(res))
axNH.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke', edgecolor='gray')
axNH.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
axNH.add_feature(cfeature.RIVERS.with_scale(res))

#设置南海子图的坐标范围，和边框linewidth
axNH.set_extent([105, 126, 2, 27])
axNH.outline_patch.set_linewidth(1)

# 比例尺
add_scalebar(86, 16, 1000, ax1)       # 设置比例尺和轴标题
# endregion


# ==================================================region top Pic========================================================
moveWindow = 111
vx2 = np.linspace(start=tifExtent[0],stop=tifExtent[1], num=img_data.shape[1], endpoint=False)
vx2 = vx2[int(moveWindow/2):img_data.shape[1]-int(moveWindow/2)-1]
##-----计算平均1------##
# vm = np.nanmean(img_data,axis=1)
# vm[np.isnan(vm)]=0
# vx, vm = smooth_xy(vx, vm)
vm2 = np.array([])
mstd_up2 = np.array([])
mstd_bot2 = np.array([])
for i in tqdm(range(moveWindow-1,img_data.shape[1],1),desc='Processing'):
    vm2 = np.append(vm2, np.nanmean(img_data[:,i-moveWindow:i]))
    mstd_up2 = np.append(mstd_up2, np.nanmean(img_data[:,i-moveWindow:i]) + np.nanstd(img_data[:,i-moveWindow:i]) )
    mstd_bot2 = np.append(mstd_bot2, np.nanmean(img_data[:,i-moveWindow:i]) - np.nanstd(img_data[:,i-moveWindow:i]) )
vm2[np.isnan(vm2)]=0
##-----计算平均2------##
# vm = np.nanmean(img_data,axis=0)
# mstd_up = vm + np.nanstd(img_data,axis=0) * 2
# mstd_bot = vm - np.nanstd(img_data,axis=0) * 2
##-----返回窗口------##
sid2 = max(np.where(vm2 != 0)[0][0],np.where(vx2>73)[0][0])
eid2 = min(np.where(vm2 != 0)[0][-1],np.where(vx2<138)[0][-1])
vm2 = vm2[sid2:eid2+1-moveWindow]
mstd_up2 = mstd_up2[sid2:eid2+1-moveWindow]
mstd_bot2 = mstd_bot2[sid2:eid2+1-moveWindow]
vx2 = vx2[sid2:eid2+1-moveWindow]
# vx, vm = smooth_xy(vx, vm)

ax2 = fig.add_subplot(gs[0, 0:col-1])
ax2.plot(vx2,vm2,linewidth=2,c=r'Black')
ax2.fill_between(vx2,mstd_bot2,mstd_up2, color='skyblue')
# ax2.xaxis.set_label_position("top")
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_xlim(startLon,endLon)
ax2.set_ylim(mstd_bot2.min(), mstd_up2.max())
ax2.tick_params("both", which='major', direction='in')
ax2.set_xlabel(r"Longitude ($^\circ$)",ha="center",labelpad=0)
ax2.set_ylabel('Mean value of fBNPP',labelpad=0)
# endregion

# ================================================== region Right Pic ==================================================
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
vx3 = np.linspace(start=tifExtent[2],stop=tifExtent[3], num=img_data.shape[0], endpoint=False)
vx3 = vx3[int(moveWindow/2):img_data.shape[0]-int(moveWindow/2)-1]
##-----计算平均1------##
# vm = np.nanmean(img_data,axis=1)
# vm[np.isnan(vm)]=0
# vx, vm = smooth_xy(vx, vm)
vm3 = np.array([])
mstd_up3 = np.array([])
mstd_bot3 = np.array([])
for i in tqdm(range(img_data.shape[0],moveWindow,-1),desc='Process2'):
    vm3 = np.append(vm3, np.nanmean(img_data[i - moveWindow:i, :]))
    mstd_up3 = np.append(mstd_up3, np.nanmean(img_data[i - moveWindow:i, :]) + np.nanstd(img_data[i - moveWindow:i, :]) )
    mstd_bot3 = np.append(mstd_bot3, np.nanmean(img_data[i - moveWindow:i, :]) - np.nanstd(img_data[i - moveWindow:i, :]) )
vm3[np.isnan(vm3)]=0
##-----计算平均2------##
# vm = np.nanmean(img_data,axis=1)
# mstd_up = vm + np.nanstd(img_data,axis=1) * 2
# mstd_bot = vm - np.nanstd(img_data,axis=1) * 2
##-----返回窗口------##
sid3 = max(np.where(vm3 != 0)[0][0],np.where(vx3>=20)[0][0])
eid3 = min(np.where(vm3 != 0)[0][-1],np.where(vx3<=55)[0][-1])
vm3 = vm3[sid3:eid3+1-moveWindow]
mstd_up3 = mstd_up3[sid3:eid3+1-moveWindow]
mstd_bot3 = mstd_bot3[sid3:eid3+1-moveWindow]
vx3 = vx3[sid3:eid3+1-moveWindow]
# vx, vm = smooth_xy(vx, vm)

ax3 = fig.add_subplot(gs[1:, col-1])
ax3.plot(vm3,vx3,linewidth=2,c=r'Black')
ax3.fill_betweenx(vx3,mstd_bot3,mstd_up3, color='skyblue')

# ax3.yaxis.set_label_position("right")
ax3.tick_params("both", which='major', direction='in')
ax3.set_xlim(mstd_bot3.min(), mstd_up3.max())
ax3.set_ylim(startLat,endLat+4.5)
# ax3.set_title('Year: '+str(yr))
ax3.set_ylabel(r"Latitude ($^\circ$)",ha="center",labelpad=0)
ax3.set_xlabel('Mean value of fBNPP',labelpad=0)
# endregion



# region Adjust Pic Show
plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.97, wspace=0.3, hspace=0.3)
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.subplots_adjust(left = 0, wspace=0.3, hspace=0.3)
# plt.tight_layout()
# endregion

# plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PIC'+os.sep+'fBNPP_Distribution_'+(datetime.now().strftime("%H-%M-%S"))+'.png',dpi=500,bbox_inches='tight')#, transparent=True
plt.show()
print()








































########===============================================#######
plt.rcParams['svg.fonttype'] = 'none'
# region Drawing
fig = plt.figure(figsize=(18/2.54,14.8/2.54))  # 创建Figure对象
gs = fig.add_gridspec(row,col) # 添加子图
# endregion

# region Main Pic
ax1 = fig.add_subplot(gs[1:, 0:col-1], projection=projPlc)  # 通过添加projection参数 创建geoaxes对象
# ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='dashed', draw_labels=True,linewidth=0.3)
gl.top_labels = False
gl.right_labels = False
# ax1.stock_img()
ax1.add_feature(cfeature.OCEAN.with_scale(res))
ax1.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke')#, edgecolor='gray'
# ax1.add_feature(cfeature.LAKES.with_scale(res))
# ax1.add_feature(cfeature.RIVERS.with_scale(res))
# ax1.add_feature(cfeature.BORDERS.with_scale(res))

# 栅格
ds,img_proj,img_geotrans,img_data = read_img(tif)
img_data[img_data<-9999] = np.nan
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
bins = [-0.08,-0.01,-0.005,0,0.005,0.01,0.08]
nbin = len(bins) - 1
cmaps = mpl.cm.get_cmap(cbar, nbin)
norms = mpl.colors.BoundaryNorm(bins, nbin)
imgtif = ax1.imshow(img_data, extent=tifExtent, origin='upper',transform=projPlc,
                    vmin=tifRange[0], vmax=tifRange[1]+(tifRange[1]-tifRange[0])/colorBarSectionsNum,
                    cmap=cmaps,norm=norms,aspect=1.35) #

axbar=fig.add_axes([0.089,0.078,0.02,0.2])
cb = fig.colorbar(imgtif,cax=axbar,fraction=0.05,aspect=20,shrink=0.5,
                  orientation='vertical') #,pad=0.05label='fBNPP'
# cb.set_label(label='fBNPP',loc='top') #loc参数 (×$\mathregular{10^5}$)
cb.ax.set_title('Slope')

#设置basemap
ax1.add_feature(CHN, linewidth=1,alpha=0.9, zorder=1)
ax1.add_feature(NH, linewidth=1,alpha=0.9, zorder=1)
ax1.set_extent(extents)     # ax1.set_global()
#设置南海子图的坐标
left, bottom, width, height = 0.578, 0.063, 0.18, 0.2
axNH = fig.add_axes(
    [left, bottom, width, height],
    projection=projMct
)
#添加南海子图的详细内容
imgtif = axNH.imshow(img_data, extent=tifExtent, origin='upper',transform=projPlc,
                     vmin=tifRange[0], vmax=tifRange[1]+(tifRange[1]-tifRange[0])/colorBarSectionsNum,
                     cmap=cmaps,norm=norms) #
axNH.add_feature(CHN, linewidth=1,alpha=0.9, zorder=1)
axNH.add_feature(NH, linewidth=1,alpha=0.9, zorder=1)
axNH.add_feature(cfeature.OCEAN.with_scale(res))
axNH.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke', edgecolor='gray')
axNH.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
axNH.add_feature(cfeature.RIVERS.with_scale(res))

#设置南海子图的坐标范围，和边框linewidth
axNH.set_extent([105, 126, 2, 27])
axNH.outline_patch.set_linewidth(1)

# 比例尺
add_scalebar(86, 16, 1000, ax1)       # 设置比例尺和轴标题
# endregion


ax2 = fig.add_subplot(gs[0, 0:col-1])
ax2.plot(vx2,vm2,linewidth=1,c=r'Black')
# ax2.fill_between(vx2,mstd_bot2,mstd_up2, color='skyblue')
# ax2.set_ylim(mstd_bot2.min(), mstd_up2.max())
ax2.set_ylim(-0.005, 0.005)
ax2.fill_between(vx2,vm2,where=(vm2<0), facecolor='#F46D43',alpha=0.5)
ax2.fill_between(vx2,vm2,where=(vm2>0), facecolor='#74ADD1',alpha=0.5)

# ax2.xaxis.set_label_position("top")
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_xlim(startLon,endLon)
ax2.tick_params("both", which='major', direction='in')
ax2.set_xlabel(r"Longitude ($^\circ$)",ha="center",labelpad=0)
ax2.set_ylabel('Mean value of Slope',labelpad=0)
ax2.set_yticks([-0.004,0,0.004])
# endregion



ax3 = fig.add_subplot(gs[1:, col-1])
ax3.plot(vm3,vx3,linewidth=1,c=r'Black')
# ax3.fill_betweenx(vx3,mstd_bot3,mstd_up3, color='skyblue')
# ax3.set_xlim(mstd_bot3.min(), mstd_up3.max())
ax3.set_xlim(-0.005, 0.005)
# ax3.fill_betweenx(vm3,vx3,0,color='skyblue')
ax3.fill_betweenx(vx3,vm3,where=(vm3<0), facecolor='#F46D43',alpha=0.5)
ax3.fill_betweenx(vx3,vm3,where=(vm3>0), facecolor='#74ADD1',alpha=0.5)

# ax3.yaxis.set_label_position("right")
ax3.tick_params("both", which='major', direction='in')
ax3.set_ylim(startLat,endLat+4.5)
# ax3.set_title('Year: '+str(yr))
ax3.set_ylabel(r"Latitude ($^\circ$)",ha="center",labelpad=0)
ax3.set_xlabel('Mean value of Slope',labelpad=0)
ax3.set_xticks([-0.004,0,0.004])
# endregion



# region Adjust Pic Show
plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.97, wspace=0.3, hspace=0.3)
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.subplots_adjust(left = 0, wspace=0.3, hspace=0.3)
# plt.tight_layout()
# endregion

plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PIC'+os.sep+'fBNPP_Distribution_'+(datetime.now().strftime("%H-%M-%S"))+'.svg'
            ,dpi=1000,bbox_inches='tight',format='svg')#, transparent=True
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PIC'+os.sep+'fBNPP_Distribution_'+(datetime.now().strftime("%H-%M-%S"))+'.svg'
#             ,dpi=1000,bbox_inches='tight',format='svg')#, transparent=True
plt.show()
print()

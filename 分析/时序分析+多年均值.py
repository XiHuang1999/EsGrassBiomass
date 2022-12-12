#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/11 11:34
# @Author : Xihuang O.Y.


import os, sys, time, math
from glob import glob
import pandas as pd
import numpy as np
import scipy.stats as st
from datetime import datetime
# here put the import lib
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
from adjustText import adjust_text
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader, natural_earth
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import matplotlib.ticker as mticker

# matplotlib.rcParams['backend'] = 'SVG'

sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

'''时序均值计算'''
keyName = [r'TNPP',r'BNPP',r'ANPP',r'fBNPP']
filePath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP'
stY = 2003
edY = 2017
yearList = [yr for yr in range(stY,edY+1)]
dataStc = []
ereStc = []
for vi in range(len(keyName)):
    tplist = []
    tplist_er = []
    for yr in range(stY,edY+1):
        # ANPP
        print(glob(filePath+os.sep+keyName[vi]+os.sep+r'*'+str(yr)+r'*.tif')[0])
        img_proj,img_geotrans,img_data = EsRaster.read_img(glob(filePath+os.sep+keyName[vi]+os.sep+r'*'+str(yr)+r'*.tif')[0])
        imgdata1 = img_data.reshape((img_data.shape[0]*img_data.shape[1],))
        # imgdata1[imgdata1<0] = np.nan
        tplist.append(np.nanmean(imgdata1))
        tplist_er.append(np.nanstd(imgdata1)) #/math.sqrt(len(imgdata1))
    dataStc.append(tplist)
    ereStc.append(tplist_er)
dataStc[3] = list(np.subtract([1]*len(dataStc[2]),dataStc[3]))

'''参数初始化设置'''
# region Parameters
# tif相关参数
tif = [r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP\out_Mean\ANPPMean_2003to2017_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP\out_Mean\BNPPMean_2003to2017_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP\out_Mean\TNPPMean_2003to2017_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP\out_Mean\fBNPPMean_2003to2017_Proj.tif']
CHNshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\全国无子区域.shp'
NHshp = r'G:\1_BeiJingUP\CommonData\标准-2020年中国行政区划边界-省、市-Shp\2020年中国行政区划边界-省、市-Shp\全国行政边界\南沙群岛海上国境线.shp'
QTPshp = r'G:\1_BeiJingUP\CommonData\QTP\QTP_WGS84.shp'
Alberts_China = ccrs.AlbersEqualArea(central_longitude=110, central_latitude=0, standard_parallels=(25, 47))
tifRange = [0,150]
colorBarSectionsNum = 10
cbar = 'Spectral'
# fig相关参数
row,col = 4,4
startLon = 72
endLon = 107
startLat = 24
endLat = 42
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



gs = gridspec.GridSpec(6,6,width_ratios=[0.4,2,2,2,2,0.4],height_ratios=[6,6,0.2,0.9,0.9,0.5],hspace=0.2)
fig = plt.figure(figsize=(15,10))  # 创建Figure对象
# gs = fig.add_gridspec(3,1) # 添加子图
ax01 = fig.add_subplot(gs[0,0:3], projection=projPlc) #sharex=ax1
ax02 = fig.add_subplot(gs[0,3:5], projection=projPlc) #sharex=ax1
ax03 = fig.add_subplot(gs[1,0:3], projection=projPlc) #sharex=ax1
ax04 = fig.add_subplot(gs[1,3:5], projection=projPlc) #sharex=ax1
ax1 = fig.add_subplot(gs[3:,1:4]) #sharex=ax1
ax2 = fig.add_subplot(gs[4,1:4],sharex=ax1)
ax3 = fig.add_subplot(gs[5,1:4],sharex=ax1)
ax2.set_facecolor('none')
ax2.set_alpha(0)
ax3.set_facecolor('none')
ax3.set_alpha(0)


'''绘图'''
axlist = [ax01,ax02,ax03,ax04]
vmins = [0,0,0,0]
vmaxs = [200,1000,1000,1]
for axi,tifi,ii in zip(axlist,tif,range(len(axlist))):
    print('GGG->')
    # region Main Pic
    gl = axi.gridlines(color='lightgrey', linestyle='-', draw_labels=True)  # , 'color': 'k', "font": Times}
    gl.top_labels = False   # 关闭上部经纬标签
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(startLon, endLon + 5, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(startLat, endLat + 4, 2))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}  # , "color": 'k', "font": Times}
    gl.ylabel_style = {'size': 12}

    # axi.stock_img()
    axi.add_feature(cfeature.OCEAN.with_scale(res))
    axi.add_feature(cfeature.LAND.with_scale(res),color='white', edgecolor='gray')
    axi.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
    axi.add_feature(cfeature.RIVERS.with_scale(res))
    # axi.add_feature(cfeature.BORDERS.with_scale(res))

    # 栅格
    img_proj,img_geotrans,img_data = EsRaster.read_img(tifi)
    img_data[img_data<0] = np.nan
    # SHP
    QTP = cfeat.ShapelyFeature(Reader(QTPshp).geometries(),projPlc, edgecolor='k', facecolor='none')
    # CHN = cfeat.ShapelyFeature(Reader(CHNshp).geometries(),projPlc, edgecolor='k', facecolor='none')
    # NH = cfeat.ShapelyFeature(Reader(NHshp).geometries(),projPlc, edgecolor='k',facecolor='none')

    # 栅格可视化，同时根据像元值设置图片颜色
    # Rs = axi.contourf(geoGrid[0], geoGrid[1], img_data, transform=Alberts_China, cmap=plt.cm.jet, levels=clevs, extend='both') #clevs,#transform=Alberts_China,cmap=plt.cm.jet,zorder=10)
    tifExtent = geo.transform_points(projPlc,
                                    np.array([img_geotrans[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1]]),
                                    np.array([img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[3]]))
    tifExtent = (tifExtent[0][0],tifExtent[1][0],tifExtent[0][1],tifExtent[1][1])
    # Or use tifExtent = (img_geotrans[0], img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1], img_geotrans[3])

    # 分级显示
    # norm = matplotlib.colors.Normalize(vmin=vmins[ii], vmax=vmaxs[ii])
    bins = np.arange(vmins[ii], vmaxs[ii]+(vmaxs[ii]-vmins[ii])/10, (vmaxs[ii]-vmins[ii])/10)
    nbin = len(bins) - 1
    cmaps = mpl.cm.get_cmap(cbar, nbin)
    norms = mpl.colors.BoundaryNorm(bins, nbin)
    # levels = np.arange(vmins[ii], vmaxs[ii], (vmaxs[ii]-vmins[ii])/10)

    # 现实主图tifmatplotlib
    imgtif = axi.imshow(img_data, extent=tifExtent, origin='upper',
                        transform=projPlc, cmap=cmaps, norm=norms,
                        aspect=1.3)#cmap = mpl.cm.RdBu_r,#vmin=vmins[ii], vmax=vmaxs[ii],#tifRange[1]+tifRange[1]/colorBarSectionsNum,
    # ax.contourf(lon, lat, temp, levels=np.arange(-40, 40), cmap='RdBu_r', transform=ccrs.PlateCarree())
    # 显示colorbar
    # cmap = mpl.cm.viridis
    # norm = mpl.colors.Normalize(vmin=vmins[ii], vmax=vmaxs[ii])
    # # position = plt.axes([0.1, 0.25, 0.7, 0.025])
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axi, extend='both', shrink=0.4, label='气温  ℃',
    #              orientation='horizontal')
    cb = fig.colorbar(imgtif,label=tifi.split(os.sep)[-1].split('Mean')[0],\
                      ,ax=axi) #ticks=bins[0:nbin]
    # cb.set_label(bins[0:nbin]) #'colorbar',fontdict=

    axi.add_feature(QTP, linewidth=2,alpha=0.9, zorder=1)
    # axi.add_feature(CHN, linewidth=2,alpha=0.9, zorder=1)
    # axi.add_feature(NH, linewidth=2,alpha=0.9, zorder=1)
    # axi.spines['right'].set_visible(False)
    axi.set_extent(extents)     # axi.set_global()
    # axi.set_title(tifi.split(os.sep)[-1].split('Mean')[0], fontsize='medium')
    # #设置南海子图的坐标
    # left, bottom, width, height = 0.55, 0.077, 0.21, 0.23
    # axNH = fig.add_axes(
    #     [left, bottom, width, height],
    #     projection=projMct
    # )
    # #添加南海子图的详细内容
    # axNH.add_feature(CHN, linewidth=1.5,alpha=0.9, zorder=1)
    # axNH.add_feature(NH, linewidth=1.5,alpha=0.9, zorder=1)
    # axNH.add_feature(cfeature.OCEAN.with_scale(res))
    # axNH.add_feature(cfeature.LAND.with_scale(res),color='whitesmoke', edgecolor='gray')
    # axNH.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
    # axNH.add_feature(cfeature.RIVERS.with_scale(res))
    #
    # #设置南海子图的坐标范围，和边框linewidth
    # axNH.set_extent([105, 125, 0, 25])
    # axNH.outline_patch.set_linewidth(2)








# 将构造的ax右侧的spine向右偏移
ax2.spines['right'].set_position(('outward',10))
ax3.spines['right'].set_position(('outward',10))

# 初始化颜色
key = ['TNPP and BNPP','TNPP and BNPP','ANPP','fBNPP']
lc = ['black','tab:brown','tab:green','tab:red']
# 绘制
img1, = ax1.plot(yearList,dataStc[0],'o-',c=lc[0], alpha=0.6, label="TNPP") #, c='tab:blue'
img2, = ax1.plot(yearList,dataStc[1],'o-',c=lc[1], alpha=0.6,label="BNPP")
img3, = ax2.plot(yearList,dataStc[2],'o-',c=lc[2], alpha=0.6,label="ANPP")
img4, = ax3.plot(yearList,dataStc[3],'o-',c=lc[3], alpha=0.6,label="fBNPP")
# 获取对应折线图颜色给到spine ylabel yticks yticklabels
axs = [ax1,ax1,ax2,ax3]
imgs = [img1,img2,img3,img4]
# 文字
for i in range(len(axs)):
    linreg = st.linregress(pd.Series(yearList, dtype=np.float64), pd.Series(dataStc[i], dtype=np.float64))
    x = np.linspace(min(yearList), max(yearList), 100)
    y = linreg[0]*x+linreg[1]
    axs[i].plot(x,y,'--',c=lc[i],lw=1)
    axs[i].text(max(x) -0.2, max(dataStc[i][-5:]), 'Slope=%.2f, R²=%.2f' % (linreg[0], linreg[2] ** 2), ha='right', va='bottom',
                    c=lc[i])
# 轴颜色
for i in range(len(axs)):
    if i != 1:
        axs[i].spines['right'].set_color(lc[i])
        axs[i].set_ylabel(key[i], c=lc[i])
        axs[i].tick_params(axis='y', color=lc[i], labelcolor = lc[i])
        # axs[i].spines['left'].set_color(lc[i])#注意ax1是left
        axs[i].spines['top'].set_visible(False)
# 设置其他细节
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 3
ax2.yaxis.tick_right()
ax3.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax3.yaxis.set_label_position("right")
ax1.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax2.get_xaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
# ax3.spines['bottom'].set_visible(False)
ax1.set_xlabel('Year')
x_major_locator=MultipleLocator(1)
ax1.xaxis.set_major_locator(x_major_locator)
ax1.set_ylim(50,350)
ax2.set_ylim(60,100)
ax3.set_ylim(0.6,0.7)
# plt.legend(loc='lower left')
# 图例
lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.8, 0.2))  # 图例的位置，bbox_to_anchor=(0.5, 0.92),
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.2,hspace=0.2)
# plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\QTP均值+年际变化_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.png',dpi = 800)
plt.show()



print()



# # 断轴
# d = .85  #设置倾斜度
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15,
#               linestyle='none', color='r', mec='r', mew=1, clip_on=False) #绘制断裂处的标记
# ax2.plot([0, 2020], [0, 150],transform=ax2.transAxes, **kwargs)
# plt.tight_layout()
# plt.savefig('n axis.png',dpi = 600)
# plt.show()
'''=========================================================================='''
# ###=================================='''图样式一'''================================================
# def cm2inch(x,y):
#     return(x/2.54,y/2.54)
#
# # 构造多个ax
# fig,ax1 = plt.subplots(figsize =(8,4))
# ax2 = ax1.twinx()
# ax3 = ax1.twinx()
# # 将构造的ax右侧的spine向右偏移
# ax2.spines['right'].set_position(('outward',10))
# ax3.spines['right'].set_position(('outward',60))
#
#
# # 绘制初始化
# axs = [ax1, ax1, ax2, ax3]
# key = ['TNPP and BNPP','TNPP and BNPP','ANPP','fBNPP']
# lc = ['black','tab:brown','tab:green','tab:red']
# img1, = ax1.plot(yearList,dataStc[0],'o-', alpha=0.6, c='black',label="TNPP")
# img2, = ax1.plot(yearList,dataStc[1],'o-', alpha=0.6, c='tab:brown',label="BNPP")
# img3, = ax2.plot(yearList,dataStc[2],'o-', alpha=0.6, c='tab:green',label="ANPP")
# img4, = ax3.plot(yearList,dataStc[3],'o-', alpha=0.6, c='tab:red',label="fBNPP")
# for i in range(len(axs)):
#     linreg = st.linregress(pd.Series(yearList, dtype=np.float64), pd.Series(dataStc[i], dtype=np.float64))
#     x = np.linspace(min(yearList), max(yearList), 100)
#     y = linreg[0]*x+linreg[1]
#     axs[i].plot(x,y,'--',c=lc[i],lw=1)
#     axs[i].text(max(x) + 0.5, max(dataStc[i][-4:]), 'Slope=%.2f, R²=%.2f' % (linreg[0], linreg[2] ** 2), ha='right', va='bottom',
#                     c=lc[i])
#
# #获取对应折线图颜色给到spine ylabel yticks yticklabels
# imgs = [img1,img2,img3,img4]
# for i in range(len(axs)):
#     if i != 1:
#         axs[i].spines['right'].set_color(lc[i])
#         axs[i].set_ylabel(key[i], c=lc[i])
#         axs[i].tick_params(axis='y', color=lc[i], labelcolor = lc[i])
#         # axs[i].spines['left'].set_color(lc[i])#注意ax1是left
#         axs[i].spines['top'].set_visible(False)
# # 设置其他细节
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['lines.markersize'] = 3
# ax1.spines['right'].set_visible(False)
# ax1.set_xlabel('Year')
# x_major_locator=MultipleLocator(1)
# ax1.xaxis.set_major_locator(x_major_locator)
# ax1.set_ylim(50,350)
# ax2.set_ylim(60,100)
# ax3.set_ylim(0.6,0.7)
# ax1.set_ylim(0,400)
# ax2.set_ylim(40,140)
# ax3.set_ylim(0.6,0.9)
# # 图例
# lines = []
# labels = []
# for ax in fig.axes:
#     axLine, axLabel = ax.get_legend_handles_labels()
#     lines.extend(axLine)
#     labels.extend(axLabel)
# fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 1))  # 图例的位置，bbox_to_anchor=(0.5, 0.92),
# plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\QTP均值年际变化_Style1_'+(datetime.now().strftime("%H-%M-%S"))+'.png',dpi = 600)
# plt.show()



# 构造多个ax



###=================================='''图样式二'''================================================


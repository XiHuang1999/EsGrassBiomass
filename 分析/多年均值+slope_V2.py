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

'''参数初始化设置'''
# region Parameters
# tif相关参数
tif = [r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\ANPPMean_2000to2018_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\BNPPMean_2000to2018_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\TNPPMean_2000to2018_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\fBNPPMean_2000to2018_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\ANPP_2000to2018_slope_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\BNPP_2000to2018_slope_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\TNPP_2000to2018_slope_Proj.tif',
       r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\fBNPP_2000to2018_slope_Proj.tif']
CHNshp = r'G:\1_BeiJingUP\CommonData\中国地图-审图号GS(2019)1822号-shp格式\中国地图-审图号GS(2019)1822号-shp格式\2019中国地图-审图号GS(2019)1822号\中国轮廓线_WGS84.shp'
NHshp = r'G:\1_BeiJingUP\CommonData\中国地图-审图号GS(2019)1822号-shp格式\中国地图-审图号GS(2019)1822号-shp格式\2019中国地图-审图号GS(2019)1822号\九段线_WGS84.shp'
QTPshp = r'G:\1_BeiJingUP\CommonData\QTP\QTP_WGS84.shp'
Alberts_China = ccrs.AlbersEqualArea(central_longitude=110, central_latitude=0, standard_parallels=(25, 47))
tifRange = [0,150]
cbNum = [5,6]
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
clist=[(120/255,38/255,38/255),(255/255,209/255,117/255),(185/255,178/255,191/255),(39/255,32/255,110/255)]   # clist=[[45,85,85,0],[0,0,25,0],[85,87,52,0]]
# clist=[(120/255,38/255,38/255),(255/255,255/255,145/255),(85/255,255/255,0/255),(39/255,32/255,110/255)]   # clist=[[45,85,85,0],[0,0,25,0],[85,87,52,0]]
#ewcmp = LinearSegmentedColormap.from_list('chaos', clist, N=)  #newcmp = ListedColormap(clist)
#newcmp = matplotlib.colors.ListedColormap(clist, 'indexed')
cbar = ['Spectral','RdYlGn']
# fig相关参数
row,col = 4,4
startLon = 72
endLon = 136
startLat = 17
endLat = 50
ticGapLon = 15
ticGapLat = 10
res = '50m' # BaceMap Resolution
moveWindow = 3
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rc('font',family='Times New Roman')
extents = (startLon, endLon, startLat, endLat)  # [-180,180,-90,90]#
projPlc = ccrs.PlateCarree() #central_longitude=110,projAlb = ccrs.AlbersEqualArea(),projMct = ccrs.Mercator()
geo = ccrs.Geodetic()
# 设置其他细节
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 2.3
plt.rcParams['font.size'] = 8
# endregion



gs = gridspec.GridSpec(4,6,width_ratios=[0.5,2,2,2,2,0.5],height_ratios=[25,25,25,25])#,hspace=0.2,hspace=0, wspace=0.2
fig = plt.figure(figsize=(19/2.54,23/2.54))  #(figsize=(12.7,9.5))  # 创建Figure对象单位为12.7inch宽+9.5inch高: inch=cm/f
plt.subplots_adjust(left=-0.01,bottom=0.03,right=1.04,top=0.99,hspace=0.15)#,wspace=0.19,hspace=0.21
# gs = fig.add_gridspec(3,1) # 添加子图
ax01 = fig.add_subplot(gs[0,0:3], projection=projPlc) #sharex=ax1
ax02 = fig.add_subplot(gs[0,3:5], projection=projPlc) #sharex=ax1
ax03 = fig.add_subplot(gs[1,0:3], projection=projPlc) #sharex=ax1
ax04 = fig.add_subplot(gs[1,3:5], projection=projPlc) #sharex=ax1
ax05 = fig.add_subplot(gs[2,0:3], projection=projPlc) #sharex=ax1
ax06 = fig.add_subplot(gs[2,3:5], projection=projPlc) #sharex=ax1
ax07 = fig.add_subplot(gs[3,0:3], projection=projPlc) #sharex=ax1
ax08 = fig.add_subplot(gs[3,3:5], projection=projPlc) #sharex=ax1


'''绘图'''
axlist = [ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax08]
vmins = [  0,    0,    0, 0, -60, -180, -260, -0.2]     # [  0, -60,    0, -180,    0, -260, 0, -0.2]
vmaxs = [500, 1500, 2500, 1,  60,  180,  260,  0.2]     # [500,  60, 1500,  180, 2500,  260, 1,  0.2]
HNlocx = [0.33,    0.33+0.475,  0.33, 0.33+0.475,
          0.33,    0.33+0.475,  0.33, 0.33+0.475]
HNlocy = [0.775,            0.775,              0.775-0.2488,  0.775-0.24833,
          0.775-2*0.24833,  0.775-2*0.24833,  0.775-3*0.24833,    0.775-3*0.24833]
textx = [25,166,25,166,25,166,25,166]
texty = [159.9,159.9,71.5,71.5,-17.0,-17.0,-105.5,-105.5]
txt = ['a','b','c','d','a','b','c','d'] #['a','e','b','f','c','g','d','h']
cbLabel = ['{x:.0f}','{x:.0f}','{x:.0f}','{x:.1f}','{x:.0f}','{x:.0f}','{x:.0f}','{x:.3f}']
lbtitle = ['','Slope of ']
for axi,tifi,ii in zip(axlist,tif,range(len(axlist))):
    print('GGG->')
    # region Main Pic
    gl = axi.gridlines(color='lightgrey', linestyle='--', draw_labels=True, alpha=0.5)  # , 'color': 'k', "font": Times}
    gl.top_labels = False   # 关闭上部经纬标签
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(ticGapLon*(startLon//ticGapLon), ticGapLon*(endLon//ticGapLon)+1, ticGapLon))
    gl.ylocator = mticker.FixedLocator(np.arange(ticGapLat*(startLat//ticGapLat), ticGapLat*(endLat//ticGapLat)+1, ticGapLat))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}  # , "color": 'k', "font": Times}
    gl.ylabel_style = {'size': 8}

    # axi.stock_img()
    axi.add_feature(cfeature.OCEAN.with_scale(res))
    axi.add_feature(cfeature.LAND.with_scale(res),color='white',lw=0.25, edgecolor='gray')
    # axi.add_feature(cfeature.LAKES.with_scale(res),lw=0.25, edgecolor='gray')
    # axi.add_feature(cfeature.RIVERS.with_scale(res))
    # axi.add_feature(cfeature.BORDERS.with_scale(res))

    # 栅格
    img_proj,img_geotrans,img_data = EsRaster.read_img(tifi)
    img_data[img_data<=-9998] = np.nan
    img_data[img_data>=vmaxs[ii]] = vmaxs[ii]-0.01
    img_data[img_data<=vmins[ii]] = vmins[ii]+0.01
    # SHP
    QTP = cfeat.ShapelyFeature(Reader(QTPshp).geometries(),projPlc, edgecolor='k', facecolor='none')
    CHN = cfeat.ShapelyFeature(Reader(CHNshp).geometries(),projPlc, edgecolor='k', facecolor='none')
    NH = cfeat.ShapelyFeature(Reader(NHshp).geometries(),projPlc, edgecolor='k',facecolor='none')

    # 栅格可视化，同时根据像元值设置图片颜色
    # Rs = axi.contourf(geoGrid[0], geoGrid[1], img_data, transform=Alberts_China, cmap=plt.cm.jet, levels=clevs, extend='both') #clevs,#transform=Alberts_China,cmap=plt.cm.jet,zorder=10)
    tifExtent = geo.transform_points(projPlc,
                                    np.array([img_geotrans[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1]]),
                                    np.array([img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[3]]))
    tifExtent = (tifExtent[0][0],tifExtent[1][0],tifExtent[0][1],tifExtent[1][1])
    # Or use tifExtent = (img_geotrans[0], img_geotrans[3] + img_geotrans[5] * img_data.shape[0], img_geotrans[0] + img_geotrans[1] * img_data.shape[1], img_geotrans[3])

    # 分级显示Label # norm = matplotlib.colors.Normalize(vmin=vmins[ii], vmax=vmaxs[ii]) # levels = np.arange(vmins[ii], vmaxs[ii], (vmaxs[ii]-vmins[ii])/10)
    if ii == 3:
        bins = [0,0.2,0.4,0.6,0.8,1]
    elif ii == 4:
        bins = [-260, -4, -2, 0, 2, 4, 260]
    elif ii == 5:
        bins = [-100,-20,-10,0,10,20,100]
    elif ii == 6:
        bins = [-100,-20,-10,0,10,20,100]
    elif ii == 7:
        bins = [-0.2, -0.01, -0.005, 0, 0.005, 0.01, 0.2]
    else:
        colorBarSectionsNum = cbNum[ii % 2]
        bins = np.arange(vmins[ii], vmaxs[ii]+(vmaxs[ii]-vmins[ii])/colorBarSectionsNum, (vmaxs[ii]-vmins[ii])/colorBarSectionsNum)
    nbin = len(bins) - 1
    # ColorBar 设置颜色
    if ii in [9]:#[4,5,6,7]:
        cmaps = LinearSegmentedColormap.from_list('chaos', clist, N=nbin)
    else:
        cmaps = mpl.cm.get_cmap(cbar[ii//4], nbin)
    norms = mpl.colors.BoundaryNorm(bins, nbin)
    formatter = mticker.StrMethodFormatter(cbLabel[ii])

    # 现实主图tifmatplotlib
    imgtif = axi.imshow(img_data, extent=tifExtent, origin='upper',
                        transform=projPlc, cmap=cmaps, norm=norms,
                        aspect=1.3,resample=False)#,binary_compression_level=0,cmap = mpl.cm.RdBu_r,#vmin=vmins[ii], vmax=vmaxs[ii],#tifRange[1]+tifRange[1]/colorBarSectionsNum,
    # ax.contourf(lon, lat, temp, levels=np.arange(-40, 40), cmap='RdBu_r', transform=ccrs.PlateCarree())
    # 显示colorbar
    # cmap = mpl.cm.viridis
    # norm = mpl.colors.Normalize(vmin=vmins[ii], vmax=vmaxs[ii])
    # # position = plt.axes([0.1, 0.25, 0.7, 0.025])
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axi, extend='both', shrink=0.4, label='气温  ℃',
    #              orientation='horizontal')
    cb = fig.colorbar(imgtif,label=lbtitle[ii//4]+tifi.split(os.sep)[-1].split(r'_')[0].split('Mean')[0],\
                      ax=axi) #ticks=bins[0:nbin]
    cb.set_ticks(bins)
    cb.set_ticklabels([formatter(bin) for bin in bins])
    # cb.set_label(bins[0:nbin]) #'colorbar',fontdict=

    # axi.add_feature(QTP, linewidth=0.5,alpha=0.9, zorder=1)
    axi.add_feature(CHN, linewidth=0.5,alpha=0.9, zorder=1)
    axi.add_feature(NH, linewidth=0.5,alpha=0.9, zorder=1)
    axi.set_extent(extents)     # axi.set_global()
    # axi.spines['right'].set_visible(False)
    # axi.set_title(tifi.split(os.sep)[-1].split('Mean')[0], fontsize='medium')

    # #设置南海子图的坐标
    left, bottom, width, height = HNlocx[ii], HNlocy[ii], 0.07, 0.07
    axNH = fig.add_axes(
        [left, bottom, width, height],
        projection=projPlc #projMct
    )
    #添加南海子图的详细内容
    axNH.add_feature(CHN, linewidth=0.5,alpha=0.9, zorder=1)
    axNH.add_feature(NH, linewidth=0.5,alpha=0.9, zorder=1)
    axNH.add_feature(cfeature.OCEAN.with_scale(res))
    axNH.add_feature(cfeature.LAND.with_scale(res),color='white', edgecolor='gray')
    axNH.add_feature(cfeature.LAKES.with_scale(res), edgecolor='gray')
    axNH.add_feature(cfeature.RIVERS.with_scale(res),lw=1)

    #设置南海子图的坐标范围，和边框linewidth
    axNH.set_extent([105, 125, 2, 27])
    # axNH.outline_patch.set_linewidth(2)


for ii in range(len(axlist)):
    plt.text(-113+140*(ii%2), 336.7-88.4*(ii//2),r'('+txt[ii]+r')',fontdict={'fontsize':8,'weight':'bold'})


# plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\QTP均值+年际变化_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 2500)
plt.show()




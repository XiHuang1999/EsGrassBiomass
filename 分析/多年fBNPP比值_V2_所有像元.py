#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/26 20:38
# @Author : Xihuang O.Y.

import os,sys,time
import pandas as pd
import numpy as np
from osgeo import gdal
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'black'
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font',family='Times New Roman')


fBNPP_Path = r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\BMA\fBNPP'
lucctif = r"G:\1_BeiJingUP\CommonData\China\CGrass_China1km_8.tif"
A,B,lucc = EsRaster.read_img(lucctif)
excFile = r"G:\1_BeiJingUP\AUGB\Table\CGrass草地类型_ReclassifyName.csv"
exc = pd.read_csv(excFile,header=None)
coll = list(exc.iloc[:,0])


fb = []
for lucci in range(len(coll)):
    print(r'==>'+str(lucci))
    tpfb = []
    for yr in range(2000,2018+1):
        tif = glob(fBNPP_Path+os.sep+r'*'+str(yr)+r'*.tif')[0]
        A,B,img = EsRaster.read_img(tif)
        pixv = img[lucc == lucci]
        pixv = list(pixv[~np.isnan(pixv)])
        tpfb += pixv #tpfb.append(pixv)
        # temp = img[lucc==lucci]
    fb.append(tpfb)
fb = pd.DataFrame(fb).T #
fb.columns=coll[0:fb.shape[1]]
cgrass = coll
fb1 = fb[cgrass]
fb = fb1

#计算均值
fbmean = []
for l in list(fb.values):
    fbmean+=(list(l))
fbmean = np.mean(fbmean)

print(coll)
#用matplotlib来画出箱型图
fig = plt.figure(figsize=(14/2.54,7/2.54))
plt.boxplot(x=fb,labels=coll[0:17],whis=1.5,widths=0.5,\
            vert=True,meanline=True,patch_artist = True,showmeans=True,\
            boxprops={'linewidth':1.4},\
            medianprops = {'linewidth': 1.5}, \
            #meanprops={'color': 'black', 'ls': 'dotted', 'linewidth': 1.5},\
            flierprops={'marker':'+','markersize':3.5,'markerfacecolor':'black'}
            )     # https://blog.csdn.net/weixin_44052055/article/details/121442449
# plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.2,hspace=0.2)
plt.yticks(rotation=0, fontproperties = 'Times New Roman',fontsize=8) # 旋转90度,fontsize=10
plt.xticks(rotation=90, fontproperties = 'SimHei',fontsize=8) # 旋转90度,fontsize=10
# plt.xlabel(cgrass)
plt.legend([Line2D([0], [0], color='orange',ls='-', lw=2),
            # Line2D([0], [0], color='black',ls='(0,(5,1))', lw=2),
            Line2D([0], [0], color='black',ls='dotted', lw=2)],\
           ['Median', 'Mean of fBNPP'], #, 'Mean'
           ) #bbox_to_anchor=(0.9, 0.05)
plt.axhline(0.62, linestyle='dotted', color='k') # 水平线，y需要传入一个具体数值，比如上图是0
# plt.text(fbmean+0.02,1,r'Mean=%.2f' % (fbmean),fontdict={'family': 'Times New Roman'}) # ,'fontsize':12
plt.text(11,fbmean-0.06,r'Mean=%.2f' % (0.62),fontdict={'family': 'Times New Roman'}) # ,'fontsize':12
# plt.axhline(fbmean, linestyle=(0,(5,1)), color='k') # 水平线，y需要传入一个具体数值，比如上图是0
# plt.text(15,fbmean+0.02,r'Mean=%.2f' % (fbmean),fontdict={'fontsize':12})
plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\QTP地下比_V_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.png',dpi = 800)
plt.show()

print()






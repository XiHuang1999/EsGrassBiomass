#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/11 11:34
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time, math
from glob import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# matplotlib.rcParams['backend'] = 'SVG'

sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

keyName = [r'ANPP',r'TNPP',r'BNPP',r'fBNPP']
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

# 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
plt.errorbar(yearList, dataStc[0], yerr=ereStc[0], fmt='g-o',color='green',capsize=4)#plt.errorbar(figdata.columns,figdata.loc['R1D1'],yerr=std_table.loc['R1D1'],fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms
plt.errorbar(yearList, dataStc[1], yerr=ereStc[1], fmt='b-o',color='black',capsize=4)
plt.errorbar(yearList, dataStc[2], yerr=ereStc[2], fmt='r-o',color='gray',capsize=4)
plt.show()



a = [i/math.sqrt(1166582) for i in ereStc[0]]
b = [i/math.sqrt(1166582) for i in ereStc[1]]
c = [i/math.sqrt(1166582) for i in ereStc[2]]
plt.errorbar(yearList, dataStc[0], yerr=a, fmt='g-o',color='green',capsize=4)#plt.errorbar(figdata.columns,figdata.loc['R1D1'],yerr=std_table.loc['R1D1'],fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms
plt.errorbar(yearList, dataStc[1], yerr=b, fmt='b-o',color='black',capsize=4)
plt.errorbar(yearList, dataStc[2], yerr=c, fmt='r-o',color='gray',capsize=4)
plt.show()
# imgdata4 = imgdata4.reshape((img_data.shape[0], img_data.shape[1]))
# EsRaster.write_img(outPath + os.sep + keyName[3] + os.sep + r'fBNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
#                    imgdata4)

print()
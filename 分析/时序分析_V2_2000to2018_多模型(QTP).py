#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/11 11:34
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time, math, random
from glob import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.stats as st
from datetime import datetime
from tqdm import tqdm
# # matplotlib.rcParams['backend'] = 'SVG'
# def average(data):
#     return sum(data) / len(data)
#
#
# def bootstrap(data, B, c):
#     """
#     计算bootstrap置信区间
#     :param data: array 保存样本数据
#     :param B: 抽样次数 通常B>=1000
#     :param c: 置信水平
#     :param func: 样本估计量
#     :return: bootstrap置信区间上下限
#     """
#     data = imgdata1[~np.isnan(imgdata1)]
#     array = np.array(data)
#     n = len(array)
#     # idx = [i for i in range(n)]
#     sample_result_arr = []
#     for i in tqdm(range(B)):
#         # index_arr = random.sample(idx, n, replace=True)
#         # data_sample = array[index_arr]
#         data_sample = np.random.choice(array, n, replace=True)
#         sample_result = np.nanmean(data_sample)
#         sample_result_arr.append(sample_result)
#     a = 1 - c
#     k1 = int(B * a / 2)
#     k2 = int(B * (1 - a / 2))
#     auc_sample_arr_sorted = sorted(sample_result_arr)
#     lower = auc_sample_arr_sorted[k1]
#     higher = auc_sample_arr_sorted[k2]
#     return lower, higher

sys.path.append(r'D:\Pycharm\PyCharmPythonFiles\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

'''时序均值计算'''
clname = r'TNPP'
textx = [2008,2003,2015,2006,2000.75]
texty = [225,200,250,100,425]
# clname = r'BNPP'
# textx = [2000,2003.75,2006,2006,2011]
# texty = [568, 408, 574, 463, 581]
# clname = r'fBNPP'
# textx = [2000,2003.75,2006,2006,2011]
# texty = [775,615,781,670,788]
keyName = [r'GLASS',
           r'GLOPEM-CEVSA',
           r'MODIS',
           r'MuSyQ',
           r'BMA']
filePath = r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format'
lucctif = r'G:\1_BeiJingUP\CommonData\China\CGrass_1_99_LUCCwbf_QTP.tif' #r"G:\1_BeiJingUP\CommonData\China\CGrassChina_OnlyGrass1_Greater0.tif"
A,B,lucc = EsRaster.read_img(lucctif)
lucc = lucc.reshape(lucc.shape[0]*lucc.shape[1],)
lucc = np.where((lucc>0)&(lucc<18),lucc,0)
stY = 2000
edY = 2018
yearList = [yr for yr in range(stY,edY+1)]
dataStc = []
ereStc = []
for vi in range(len(keyName)):
    tplist = []
    tplist_er = []
    for yr in range(stY,edY+1):
        # NPP
        # tp = glob(filePath + os.sep + keyName[vi] + os.sep + clname + os.sep + r'*' + str(yr) + r'*.tif')[0]
        tp = glob(filePath + os.sep + r'*' + keyName[vi] + r'*_chinese_Tiff' + os.sep + r'*' + str(yr) + r'*.tif')[0]
        print(tp)
        img_proj,img_geotrans,img_data = EsRaster.read_img(tp)
        imgdata1 = img_data.reshape((img_data.shape[0]*img_data.shape[1],))
        imgdata1[imgdata1<0] = np.nan
        # Mean
        dt = imgdata1[lucc!=0]
        tplist.append(np.nanmean(dt))

    dataStc.append(tplist)
    # ereStc.append(tplist_er)
# dataStc[3] = list(np.subtract([1]*len(dataStc[2]),dataStc[3]))

dataStc1 = dataStc.copy()
# dataStc[keyName.index('GLOPEM-CEVSA')] = [value/0.45 for value in dataStc[keyName.index('GLOPEM-CEVSA')]]

# Normal Units
for i in [0,1,2,3]:
    dataStc[i] = [numb/0.45 for numb in dataStc[i]]
# Calc SE and Mean
se = [np.std(data,ddof=0)/math.sqrt(len(data)) for data in zip(dataStc[0],dataStc[1],dataStc[2],dataStc[3])]
# dataStc.append([np.mean(data) for data in zip(dataStc[0],dataStc[1],dataStc[2],dataStc[3])])

'''=========================================================================='''
# here put the import lib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import MultipleLocator
# from adjustText import adjust_text
from scipy import interpolate

plt.rcParams['lines.linewidth'] = 1.6
plt.rcParams['lines.color'] = 'black'
plt.rcParams['font.size'] = 8
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['lines.markersize'] = 5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font',family='Times New Roman')
gs = gridspec.GridSpec(3,1,height_ratios=[2,2,1],hspace=0.2)
fig = plt.figure(figsize=(14/2.54,7/2.54))  # 创建Figure对象
# gs = fig.add_gridspec(3,1) # 添加子图
ax1 = fig.add_subplot(gs[:,:]) #sharex=ax1

# 初始化颜色
key = [s.split(r'_')[0] for s in keyName]
lc = ['#94B447','#264D59','#C2649A','#D46C4E','Black'] #lc = ['#86E3CE','#FFDD94','#FA897B','#CCABDB']  lc = ['#849f65','#ffa450','#C2649A','#ea6114']  lc = ['#264D59','#43978D','#C2649A','#F9AD6A']
# mk = ['*','<','s','d']
# 绘制
img1, = ax1.plot(yearList,dataStc[0],'*-',c=lc[0],ms=6, alpha=0.9, label=key[0]) #, c='tab:blue'
img2, = ax1.plot(yearList,dataStc[1],'<-',c=lc[1],ms=5, alpha=0.9,label=key[1])
img3, = ax1.plot(yearList,dataStc[2],'s-',c=lc[2],ms=2.5, alpha=0.9,label=key[2])
img4, = ax1.plot(yearList,dataStc[3],'d-',c=lc[3],ms=4, alpha=0.9,label=key[3])
imgMean, = ax1.plot(yearList,dataStc[4],'.-',c='black',ms=4, alpha=0.8,label=key[4])
plt.fill_between(yearList,np.subtract(dataStc[4],se),np.add(dataStc[4],se)
                 ,color='gray',alpha=0.3)

# 文字
for i in range(len(dataStc)):
    linreg = st.linregress(pd.Series(yearList, dtype=np.float64), pd.Series(dataStc[i], dtype=np.float64))
    x = np.linspace(min(yearList), max(yearList), 100)
    y = linreg[0]*x+linreg[1]
    ax1.plot(x,y,'--',c=lc[i],lw=1)
    print(linreg)
    if linreg[0]<0.01:
        ax1.text(textx[i],texty[i], 'Slope=%.3f, R²=%.2f' % (linreg[0], linreg[2] ** 2), ha='left', va='bottom',
                    c=lc[i]) # 位置：max(x) -0.2, max(dataStc[i][-5:])
    else:
        ax1.text(textx[i], texty[i], 'Slope=%.2f, R²=%.2f' % (linreg[0], linreg[2] ** 2), ha='left', va='bottom',
                    c=lc[i])  # 位置：max(x) -0.2, max(dataStc[i][-5:])

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xlabel('Year')
ax1.set_ylabel(clname)
ax1.xaxis.set_major_locator(MultipleLocator(1))
# Rotate the tick labels and set their alignment. X轴标签旋转
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax1.yaxis.set_major_locator(MultipleLocator(50))
ax1.set_xlim(1999.5,2018.5)
ax1.set_ylim(90,500)#TNPP
# ax1.set_ylim(380,625)#BNPP
# ax1.set_ylim(380,1300)#BMA-BNPP
# 图例
lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
fig.legend(lines, labels,ncol=5, loc='upper right',
           columnspacing=1,handletextpad=0.2, bbox_to_anchor=(0.87,0.98))
plt.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.85,hspace=0.1)  #,wspace=0.19,left=0,bottom=0,right=1,top=1,
# plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\多模型年际变化_'+clname+'_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 1000)
plt.savefig(r'G:\1_BeiJingUP\CRNPP\GYCL_20230518\代码\多模型年际变化_'+clname+'_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 1000)
plt.show()



print()
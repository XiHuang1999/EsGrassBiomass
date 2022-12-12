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
import scipy.stats as st
from datetime import datetime
# matplotlib.rcParams['backend'] = 'SVG'

sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

'''时序均值计算'''
keyName = [r'TNPP',r'BNPP',r'ANPP',r'fBNPP']
filePath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan'
stY = 2000
edY = 2018
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
        imgdata1[imgdata1<0] = np.nan
        tplist.append(np.nanmean(imgdata1))
        tplist_er.append(np.nanstd(imgdata1)) #/math.sqrt(len(imgdata1))
    dataStc.append(tplist)
    ereStc.append(tplist_er)
# dataStc[3] = list(np.subtract([1]*len(dataStc[2]),dataStc[3]))



'''=========================================================================='''
# here put the import lib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import MultipleLocator
from adjustText import adjust_text
from scipy import interpolate

###=================================='''图样式一'''================================================
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



###=================================='''图样式二'''================================================

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'black'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font',family='Times New Roman')
gs = gridspec.GridSpec(3,1,height_ratios=[2,2,1],hspace=0.2)
fig = plt.figure(figsize=(14/2.54,7/2.54))  # 创建Figure对象
# gs = fig.add_gridspec(3,1) # 添加子图
ax1 = fig.add_subplot(gs[:,:]) #sharex=ax1
ax2 = fig.add_subplot(gs[1,0:],sharex=ax1)
ax3 = fig.add_subplot(gs[2,0:],sharex=ax1)
ax2.set_facecolor('none')
ax2.set_alpha(0)
ax3.set_facecolor('none')
ax3.set_alpha(0)

# 将构造的ax右侧的spine向右偏移
# 将构造的ax右侧的spine向右偏移
ax2.spines['right'].set_position(('outward',10))
# ax2.spines['bottom'].set_position(('outward',500))
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
textx = [2009,2006,2006,2009]
texty = [785,560,210,0.7]
# 文字
for i in range(len(axs)):
    linreg = st.linregress(pd.Series(yearList, dtype=np.float64), pd.Series(dataStc[i], dtype=np.float64))
    x = np.linspace(min(yearList), max(yearList), 100)
    y = linreg[0]*x+linreg[1]
    axs[i].plot(x,y,'--',c=lc[i],lw=1)
    print(linreg)
    if linreg[0]<0.01:
        axs[i].text(textx[i],texty[i], 'Slope=%.3f, R²=%.2f' % (linreg[0], linreg[2] ** 2), ha='left', va='bottom',
                    c=lc[i]) # 位置：max(x) -0.2, max(dataStc[i][-5:])
    else:
        axs[i].text(textx[i], texty[i], 'Slope=%.2f, R²=%.2f' % (linreg[0], linreg[2] ** 2), ha='left', va='bottom',
                    c=lc[i])  # 位置：max(x) -0.2, max(dataStc[i][-5:])

# 轴颜色
for i in range(len(axs)):
    if i != 1:
        axs[i].spines['right'].set_color(lc[i])
        axs[i].set_ylabel(key[i], c=lc[i])
        axs[i].tick_params(axis='y', color=lc[i], labelcolor = lc[i])
        # axs[i].spines['left'].set_color(lc[i])#注意ax1是left
        axs[i].spines['top'].set_visible(False)
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
ax1.xaxis.set_major_locator(MultipleLocator(1))
# Rotate the tick labels and set their alignment. X轴标签旋转
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax1.yaxis.set_major_locator(MultipleLocator(200))
ax2.yaxis.set_major_locator(MultipleLocator(50))
ax3.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.set_xlim(1999.5,2018.5)
ax1.set_ylim(0,800)
ax2.set_ylim(150,300)
ax3.set_ylim(0.6,0.8)
# plt.legend(loc='lower left')
# 图例
lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
fig.legend(lines, labels,ncol=4, loc='upper right', bbox_to_anchor=(0.8,0.95))  # 图例的位置，bbox_to_anchor=(0.5, 0.92),
plt.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.85,hspace=0.1)  #,wspace=0.19,left=0,bottom=0,right=1,top=1,
# plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\QTP均值+年际变化_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 1000)
plt.show()



print()
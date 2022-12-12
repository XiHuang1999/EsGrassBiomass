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
outPath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\GrassType'
TPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\TAVG0'
PPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PRCP0'
lucctif = r"G:\1_BeiJingUP\CommonData\China\CGrassChina_OnlyGrass1_Greater0.tif"
A,B,lucc = EsRaster.read_img(lucctif)
excFile = r"G:\1_BeiJingUP\AUGB\Table\地下地上比.xls"
exc = pd.read_excel(excFile)
coll = list(exc.iloc[:,1])
stY = 2000
edY = 2018
yearList = [yr for yr in range(stY,edY+1)]
KeyName = ['T','P']
dataStc1 = []
dataStc2 = []
for yr in range(stY,edY+1):
    # print(str(yr)+'-')
    print(glob(TPath + os.sep + r'*' + str(yr) + r'*.tif')[0])
    # Reading Tiff and Reshaping itimg_proj, img_geotrans, img_T = EsRaster.read_img(glob(TPath + os.sep + r'*' + str(yr) + r'*.tif')[0])
    img_proj, img_geotrans, imgT = EsRaster.read_img(glob(TPath + os.sep + r'*' + str(yr) + r'*.tif')[0])
    img_proj, img_geotrans, imgP = EsRaster.read_img(glob(PPath + os.sep + r'*' + str(yr) + r'*.tif')[0])
    # imgT = img_T.reshape((imgT.shape[0] * imgT.shape[1],))
    # imgP = img_P.reshape((imgP.shape[0] * imgP.shape[1],))
    imgT[imgT < -9998] = np.nan
    imgP[imgP < 0] = np.nan

    tplist1_ = []
    tplist2_ = []
    for lucci in range(1,17+1):
        tplist1_.append(np.nanmean(imgT[lucc == lucci]))
        tplist2_.append(np.nanmean(imgP[lucc == lucci]))
    dataStc1.append(tplist1_)
    dataStc2.append(tplist2_)
# dataStc[3] = list(np.subtract([1]*len(dataStc[2]),dataStc[3]))

dflist = []
for fb,vi in zip([dataStc1.copy(),dataStc2.copy()],[0,1]):
    # fb = pd.DataFrame(fb)
    # fb.columns=coll[0:fb.shape[1]]
    # fb = fb.T
    # cgrass = ['高寒荒漠草原类', '高寒荒漠类', '高寒草甸草原类', '高寒草甸类', '高寒草原类',
    #           '热性草丛类', '热性灌草丛类',
    #           '温性荒漠类', '温性草原化荒漠类', '温性荒漠草原类', '温性草甸草原类', '温性草原类', '温性山地草甸类',
    #           '暖性草丛类', '暖性灌草丛类', '低地草甸类', '沼泽类']
    # fb1 = fb[cgrass]
    # fb = fb1
    # fb.to_excel(outPath+os.sep+keyName[vi]+r'.xlsx')
    fb = pd.read_excel(outPath+os.sep+keyName[vi]+r'.xlsx',index_col=0,header=0)
    dflist.append(fb)

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'black'
plt.rcParams['font.size'] = 8
# plt.rcParams['markersize'] = 4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.rc('font',family='Times New Roman')
plt.figure(figsize=(16/2.54,14/2.54))
for ci in range(0,17):
    plt.scatter(dflist[0].iloc[ci,:]*0.1,dflist[1].iloc[ci,:]*0.1,marker=r"${"+str(ci)+"}$",s =100)
plt.xlabel('Temperature', fontsize=8, color='k')  # x轴label的文本和字体大小
plt.ylabel('Precipitation', fontsize=8, color='k')  # y轴label的文本和字体大小
plt.legend(coll[0:fb.shape[0]], loc='upper right', bbox_to_anchor=(1.4,1))  # 图例的位置，bbox_to_anchor=(0.5, 0.92),
plt.tight_layout()
plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\Grass_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.png',dpi = 800)
plt.show()


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
plt.rcParams['lines.markersize'] = 4
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
# mk = ['.','v','^','d']
# 绘制
img1, = ax1.plot(yearList,dataStc[0],'.-',c=lc[0], alpha=0.6, label="TNPP") #, c='tab:blue'
img2, = ax1.plot(yearList,dataStc[1],'v-',c=lc[1], alpha=0.6,label="BNPP")
img3, = ax2.plot(yearList,dataStc[2],'^-',c=lc[2], alpha=0.6,label="ANPP")
img4, = ax3.plot(yearList,dataStc[3],'d-',c=lc[3], alpha=0.6,label="fBNPP")
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
ax3.set_ylim(0.5,0.7)
# plt.legend(loc='lower left')
# 图例
lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
fig.legend(lines, labels,ncol=4, loc='upper right', bbox_to_anchor=(0.8,0.98))  # 图例的位置，bbox_to_anchor=(0.5, 0.92),
plt.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.85,hspace=0.1)  #,wspace=0.19,left=0,bottom=0,right=1,top=1,
# plt.tight_layout()
# plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\QTP均值+年际变化_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 1000)
plt.show()



print()
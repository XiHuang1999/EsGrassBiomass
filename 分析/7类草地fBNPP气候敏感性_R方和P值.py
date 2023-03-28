#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time ： 2023/1/7 03:01:37
# @IDE ：PyCharm
# @Author : Xihuang Ouyang
import pandas as pd
import numpy as np
import os,matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

alldata = pd.read_excel(r"G:\1_BeiJingUP\AUGB\Data\SEM2_Mean\Table_Yearly\Yearly_GrassType_Change_NoTrim.xlsx")
var = ['ANPP','BNPP','TNPP','fBNPP']
cgName = ["热性草丛","暖性草丛","草甸草原","高寒草甸","高寒草甸","高寒草原","荒漠"] #此顺序从前到后为草地类型数据的1-7
outPath = r"G:\1_BeiJingUP\AUGB\Data\SEM2_Mean\各类草地气候相关性"

# matplotlib.get_cachedir()
plt.rc('font',family='Times New Roman')
plt.rcParams["font.weight"] = "bold"        # 将设置为粗体是相当容易的
plt.rcParams["axes.labelweight"] = "bold"

for vari in var[3:]:
    r, p, slp = [], [], []
    for ci in range(1,7+1):

        # 计算年际变化
        yrdt = alldata[alldata.iloc[:, 1] == ci]
        yrlinreg = st.linregress(pd.Series(yrdt.loc[:,'Year'], dtype=np.float64),
                                 pd.Series(yrdt.loc[:, vari], dtype=np.float64))
        print(cgName[ci-1]+'\n'+str(yrlinreg))

        # 计算年际差异
        dt = alldata[alldata.iloc[:, 1] == ci].iloc[:,2:]
        outdt = pd.DataFrame(index=list(dt.columns)) #dt.T #
        for rowi in range(0,dt.shape[0]-1):
            for rowj in range(rowi+1,dt.shape[0]):
                # print('Row'+str(rowj)+' Sub Row'+str(rowi))
                sub = dt.iloc[rowj,:]-dt.iloc[rowi,:]
                outdt = pd.concat([outdt, sub], axis=1, ignore_index=True)
        outdt = outdt.T
        # outdt.to_excel(outPath+os.sep+cgName[ci]+r'_SubData.xlsx')

        # 计算相关
        ci_r,ci_p,ci_slp = [],[],[]
        for metoi in list(outdt.columns[4:]):
            linreg = st.linregress(pd.Series(outdt.loc[:,vari], dtype=np.float64), pd.Series(outdt.loc[:,metoi], dtype=np.float64))
            ci_r.append(linreg.rvalue)
            ci_p.append(linreg.pvalue)
            ci_slp.append(linreg.slope)
        r.append(ci_r)
        p.append(ci_p)
        slp.append(ci_slp)
    r = pd.DataFrame(r,index=cgName,columns=outdt.columns[4:])
    p = pd.DataFrame(p,index=cgName,columns=outdt.columns[4:])
    slp = pd.DataFrame(slp,index=cgName,columns=outdt.columns[4:])

    # 绘图
    # fig , ax1 = plt.subplots(figsize=(14/2.54,12/2.54)) #设置图片大小，图的版式 ax1 = plt.gca(),nrows=1
    # s = sns.heatmap(r,                                          # 此处为相关系数矩阵
    #                 annot=True,                                 # annot设置为True，使得heatmap中每个方格显示相关系数
    #                 ax=ax1,                                     # 绘制的轴
    #                 fmt='.2f',                                  # 设置相关系数保留三位小数
    #                 vmax=1.0, vmin=-1.0,                        # 设置图例中最大值和最小值的显示值
    #                 cmap='RdBu'                                 # 此处为matplotlib的colormap名称或颜色对象，该参数颜色1为红，-1为蓝
    #                 # xticklabels=list(r.columns),                # 将列表中的内容显示在横坐标处
    #                 # yticklabels=list(r.index)                   # 将列表中的内容显示在纵坐标处
    #                 )
    # [x1_label_temp.set_fontname('simsun') for x1_label_temp in ax1.get_yticklabels()]
    # [x1_label_temp.set_fontweight("bold") for x1_label_temp in ax1.get_yticklabels()]
    # ax1.set_xticklabels(list(r.columns),rotation=45, ha='center',fontweight='bold')
    # ax1.set_yticklabels(list(r.index),rotation=0, ha='right',fontweight='bold')
    # # ax1.set_xlabel(r"${}$".format('Climate\ \ factors'), fontproperties = FontProperties( size = 12))  # 怎么说呢，看图就懂了
    # # ax1.set_ylabel(r"${}$".format('Grass\ \ type'), fontproperties = FontProperties( size = 12))       # 按自己喜欢的方式设置就好了
    # ax1.set_xlabel('Climate factors',fontweight='bold')  # 看图就懂了, fontproperties = FontProperties( size = 12)
    # ax1.set_ylabel('Grass type',fontweight='bold')       # 按自己喜欢的方式设置 , fontproperties = FontProperties( size = 12)
    # ax1.tick_params(axis='both', length=0)
    #
    # # 注释: 此处就是标注显著性了，调节好坐标就行了
    # for ri in range(1,p.shape[0]+1):
    #     for ci in range(1,p.shape[1]+1):
    #         if p.iloc[ri-1,ci-1]<0.001:
    #             plt.text(ci-0.5,ri-0.6,  "***", size=10, alpha=1, horizontalalignment='center', color="black")
    #         elif p.iloc[ri-1,ci-1]<=0.01:
    #             plt.text(ci-0.5,ri-0.6,  "**", size=10, alpha=1, horizontalalignment='center', color="black")
    #         elif p.iloc[ri-1,ci-1]<=0.05:
    #             plt.text(ci-0.5,ri-0.6,  "*", size=10, alpha=1, horizontalalignment='center', color="black")
    #
    #
    # plt.title(vari + ' ('+ str(chr(97+var.index(vari))) +')', fontsize=12)  # 图片标题文本和字体大小
    # # ax1.legend(title='vari', shadow=True, fontsize=12) #loc='upper center',
    # cbar = ax1.collections[0].colorbar
    # cbar.set_label(r'Correlation')
    # plt.tight_layout()
    # # plt.savefig(outPath + os.sep + vari + '_cor_results.png', dpi=600, bbox_inches='tight')  # , transparent=True
    # # plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic' + os.sep + vari + '_cor_results.png', dpi=600,
    # #             bbox_inches='tight')  # , transparent=True
    # plt.show()
    print()


# linreg = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
# print('Training score:%.4f' % linreg.rvalue**2,end=' / ')  # 得到预测结果区间[0,1]
# # region Drawing
# xy = np.vstack([Y, outResults])
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
# pltx = [int(x) for x in np.linspace(start=0, stop=max(Y), num=1000)]
# plty = [linreg.slope * x + linreg.intercept for x in pltx]
# f, ax = plt.subplots(figsize=(6, 6))
# plt.plot(pltx, plty, '-', color='red', alpha=0.8, linewidth=2, label='Fitting Line')    # color='#4169E1'
# plt.plot(pltx, pltx, '-', color='black', alpha=0.8, linewidth=2, label='1:1')
# plt.scatter(Y, outResults,c=z,s=1.3,cmap='Spectral')
# plt.text(400,1500,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
#          '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
#          '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
#          ,fontsize=8,color = "r",fontweight='bold')
# plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
# plt.xlabel('AGB Site Value')  # 添加x轴和y轴标签
# plt.ylabel('Model Value')
# # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'RF_results.png',dpi=500,bbox_inches='tight')#, transparent=True
# plt.show()
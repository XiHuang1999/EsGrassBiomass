#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/12/4 18:35
# @Author : Xihuang O.Y.
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/13 11:22
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time, math
from glob import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import scipy.ndimage
import scipy.stats as st
from datetime import datetime
import random
sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig


for stepT in [2,3,4]:

    '''初始化变量'''
    keyName = [r'ANPP',r'BNPP',r'TNPP',r'fBNPP']
    filePath = r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\BMA'
    TPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\TAVG0'
    PPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PRCP0'
    stY = 2000
    edY = 2018
    stT = -16   # Temperature is X axis
    edT = 26
    stP = 0     # Precipitation is Y axis
    edP = 3000
    stepP = 100
    # 填充等高线

    x = np.arange(stT, edT+1, stepT)
    y = np.arange(stP, edP+1, stepP)
    X, Y = np.meshgrid(x, y)    # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值


    # '''提取值'''
    # dflist = []
    # for vi in range(len(keyName)):
    #     # Tick time
    #     t1 = time.time()
    #     # set df data
    #     df = pd.DataFrame([[[] for a in range(stT, edT + 1, stepT)] for yr in range(stP, edP + 1, stepP)])
    #     df.set_index(y, inplace=True)
    #     df.columns = x
    #     for yr in range(stY,edY+1):
    #         # print(str(yr)+'-')
    #         print(glob(filePath+os.sep+keyName[vi]+os.sep+r'*'+str(yr)+r'*.tif')[0])
    #         # Reading Tiff and Reshaping it
    #         img_proj,img_geotrans,img_data = EsRaster.read_img(glob(filePath+os.sep+keyName[vi]+os.sep+r'*'+str(yr)+r'*.tif')[0])
    #         img_proj,img_geotrans,img_T = EsRaster.read_img(glob(TPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    #         img_proj,img_geotrans,img_P = EsRaster.read_img(glob(PPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    #         imgdata = img_data.reshape((img_data.shape[0]*img_data.shape[1],))
    #         imgT = img_T.reshape((img_T.shape[0]*img_T.shape[1],))
    #         imgP = img_P.reshape((img_P.shape[0]*img_P.shape[1],))
    #         imgdata[imgdata<0] = np.nan
    #         imgT[imgT<-9990] = np.nan
    #         imgP[imgP<0] = np.nan
    #         for T in range(stT,edT+1,stepT):
    #             print(T,end=',')
    #             for P in range(stP,edP+1,stepP):
    #                 value = np.where((imgT>=T*10) & (imgT<T*10+stepT*10) &
    #                                  (imgP>=P*10) & (imgP<P*10+stepP*10)
    #                                  ,imgdata,np.nan)
    #                 value = (value[~np.isnan(value)]).tolist()
    #                 df.iloc[int((P-stP)/stepP),int((T-stT)/stepT)] += value
    #         print('\b'*100,end='')
    #     dflist.append(df)
    #     print('Using Time:'+str(time.time()-t1)+'s')
    #
    # '''保存为Excel'''
    # for vi in range(len(keyName)):
    #     dflist[vi].to_csv(r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_'
    #                       +keyName[vi]+r'_Tstep'+str(stepT)
    #                       +r'_20221203'+r'.csv')
    #     time.sleep(60)

    '''读取Excel'''
    dflist = []
    for vi in range(len(keyName)):
        dfi = pd.read_csv(r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_'
                          +keyName[vi]+r'_Tstep'+str(stepT)
                          +r'_20221203'+r'.csv',
                            index_col=0)
        for T in range(stT, edT + 1, stepT):
            print(T,end=r',')
            for P in range(stP, edP + 1, stepP):
                exec('dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]=' + dfi.iloc[
                    int((P - stP) / stepP), int((T - stT) / stepT)])
                # if (dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)])[-1] != r']':
                #     exec('dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]='+dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]+r']')
                # else:
                #     exec('dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]='+dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)])
        dflist.append(dfi)

    '''计算均值'''
    dflist_ = []
    dflist_count_ = []
    for dfi in dflist:
        # set df data
        df = pd.DataFrame([[0.0 for a in range(stT, edT + 1, stepT)] for yr in range(stP, edP + 1, stepP)])
        df.set_index(y, inplace=True)
        df.columns = x
        df_count = df.copy()
        for T in range(stT, edT + 1, stepT):
            print(T,end=',')
            for P in range(stP, edP + 1, stepP):
                l = dfi.iloc[int((P - stP) / stepP), int((T - stT) / stepT)]
                lstd = np.std(l)
                lmean = np.mean(l)
                value = np.where((l > (lmean - 2 * lstd)) & (l < (lmean + 2 * lstd)), l, np.nan)
                df_count.iloc[int((P - stP) / stepP), int((T - stT) / stepT)] = len(value)
                value = np.nanmean(value)
                # if np.isnan(value):
                #     value = -9999
                # else:
                df.iloc[int((P - stP) / stepP), int((T - stT) / stepT)] = value
        dflist_.append(df)
        dflist_count_.append(df_count)

    '''保存/读取均值'''
    dflist_ = []
    dflist_count_ = []
    for vi in range(len(keyName)):
        # dflist_[vi].to_csv(
        #     r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_' + keyName[vi] + r'_Mean_Tstep' + str(
        #         stepT) +r'_20221203'+ r'.csv')
        # dflist_count_[vi].to_csv(
        #     r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_Count' + keyName[vi] + r'_Mean_Tstep' + str(
        #         stepT) +r'_20221203'+ r'.csv')

        dflist_.append(
            pd.read_csv(
            r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_' + keyName[vi] + r'_Mean_Tstep' + str(
                stepT) +r'_20221203'+ r'.csv',index_col=0) )
        dflist_count_.append(
            pd.read_csv(
            r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_Count' + keyName[vi] + r'_Mean_Tstep' + str(
                stepT) +r'_20221203'+ r'.csv',index_col=0) )

    '''绘图'''
    plt.rcParams['font.weight'] = 'bold'
    plt.rc('font',family='Times New Roman')
    # 设置其他细节
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 2.3
    plt.rcParams['font.size'] = 8

    # 绘图参数
    # Parameters: Colorbar分级。如下：
    bins = [-1, 0, 50, 100, 200, 300, 500, 1000]  # colorbar的分级
    nbin = len(bins) - 1
    cmaps = mpl.cm.get_cmap('Spectral', nbin)  # 获取Cmap参数
    norms = mpl.colors.BoundaryNorm(bins, nbin)  # 对Cmap的颜色进行归一化处理，用于分级

    # datadf[datadf.isna()]=-1
    vmins = [0,0,0,0]
    vmaxs = [500,1500,2500,1]
    txtl = ['(a)','(b)','(c)','(d)']
    fmt = ['%.0f','%.0f','%.0f','%.1f']
    segN = [6,6,6,6]
    # Mask Data
    # mask = pd.read_csv(
    #         r"G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_Mask.csv",index_col=0)
    # mask = mask.isna() # mask : 为True的元素对应位置不会画出来（mask面具的意义）
    mask = pd.read_csv(
        r"G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_CountfBNPP_Mean_Tstep2_20221203.csv", index_col=0)
    mask = mask[mask>20]  # mask : 为True的元素对应位置不会画出来（mask面具的意义）
    mask = mask.isna()  # mask : 为True的元素对应位置不会画出来（mask面具的意义）

    # ddata[mask] = np.nan

    for vi in range(len(keyName)):
        ddata = dflist_[vi].copy() # pd.DataFrame(dflist_[vi],dtype=np.float64)
        # ddata = np.array(dflist_[vi], dtype=np.float64)
        fig = plt.figure(figsize=(7/2.54,6/2.54))   # axi = fig.add_subplots(111)
        axi = plt.gca().axes

        # # Heatmap
        # h = sns.heatmap(ddata,cmap='Spectral',
        #                 vmin=vmins[vi],vmax=vmaxs[vi],
        #                 linewidths=0,cbar=True,edgecolors='face',
        #                 cbar_kws={'drawedges':False},mask = mask
        #                 ).invert_yaxis() #,square=True

        # # Contour
        # 平滑处理 # Resample your data grid by a factor of 3 using cubic spline interpolation.
        ddata_sample = ddata  # scipy.ndimage.zoom(ddata, 1) # x,y = np.meshgrid(list(ddata.index),[int(s) for s in list(ddata.columns)])
        ddata_sample_mask = np.ma.array(ddata_sample, mask=mask)
        cs = axi.contourf(X, Y, ddata_sample_mask, corner_mask=False,
                          levels=np.linspace(vmins[vi], vmaxs[vi], segN[vi]),
                          cmap='RdBu_r',vmin=vmins[vi],vmax=vmaxs[vi],
                          )
        # ddata_sample[ddata_sample.isna()] = 1
        contour = axi.contour(cs,
                              np.linspace(vmins[vi], vmaxs[vi], segN[vi]),
                              corner_mask=False,
                              algorithm='serial',
                              colors='k')
        plt.colorbar(cs, ax=axi)
        plt.clabel(contour, fontsize=8, colors='k', fmt=fmt[vi])

        plt.xlabel('Temperature',fontsize=8, color='k') #x轴label的文本和字体大小
        plt.ylabel('Precipitation',fontsize=8, color='k') #y轴label的文本和字体大小
        # axi.set_xticks(x)
        # axi.set_xticklabels(x, rotation='horizontal')
        plt.title(keyName[vi],fontsize=8) #图片标题文本和字体大小

        # Set Parameters
        axi.text(1,27,txtl[vi])
        sns.despine(ax=axi, top=False, right=False, left=False, bottom=False)
        # cbar = axi.collections[0].colorbar  # 显示colorbar
        # cbar = plt.colorbar(h)  # 显示colorbar
        # cbar.set_ticks(np.linspace(0,2000,5))
        # cbar.outline.set_linewidth(0.01)
        # cbar.dividers.set_linewidth(0)
        # cbar.outline.set_visible(False)
        plt.tight_layout()
        # plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\Meto\气温降水-Heatmap-'+keyName[vi]+'-'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 800)
        plt.show()

        # X, Y = np.meshgrid(x, y)
        # plt.figure(figsize=(7/2.54,6/2.54))
        # # 'Spectral','RdYlBu' #algorithm{'mpl2005', 'mpl2014', 'serial', 'threaded'},
        # plt.contourf(X, Y, ddata,cmap='Greens',algorithm='serial')
        # plt.colorbar()
        # plt.xlabel('Temperature',fontsize=8, color='k') #x轴label的文本和字体大小
        # plt.ylabel('Precipitation',fontsize=8, color='k') #y轴label的文本和字体大小
        # plt.title(keyName[vi],fontsize=8) #图片标题文本和字体大小
        # plt.text(-8,2600,txtl[vi])
        # plt.tight_layout()
        # plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\Meto\气温降水-Contourf-'+keyName[vi]+'-'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 800)
        # plt.show()
print()
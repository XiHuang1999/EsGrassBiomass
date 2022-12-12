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
import seaborn as sns
import scipy.ndimage
import scipy.stats as st
from datetime import datetime
import random
sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig
from matplotlib.pyplot import MultipleLocator
import colorsys
import random
from matplotlib.lines import Line2D


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

cc = list(map(lambda x: color(tuple(x)), ncolors(17)))
cc=['#F71414', '#F87833', '#FCBD28', '#F0FD28', '#A3F829', '#4BF617', '#38F54F',
 '#08FA7A', '#24FAD5', '#21D0F6', '#147DF2', '#0B27FA', '#5D2EF5', '#9E16FE',
 '#ED0EFB', '#F90DB4', '#FB1868']
excFile = r"G:\1_BeiJingUP\AUGB\Table\地下地上比.xls"
exc = pd.read_excel(excFile)
coll = list(exc.iloc[:,1])

for stepT in [1,2]:

    '''初始化变量'''
    keyName = [r'ANPP',r'BNPP',r'TNPP',r'fBNPP']
    lucctif = r"G:\1_BeiJingUP\CommonData\China\CGrassChina_OnlyGrass1_Greater0.tif"
    filePath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan'
    TPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\TAVG0'
    PPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\PRCP0'
    stY = 2000
    edY = 2018
    stT = -14   # Temperature is X axis
    edT = 28
    stP = 0     # Precipitation is Y axis
    edP = 2800
    stepP = 100
    # 填充等高线

    x = np.arange(stT, edT+1, stepT)
    y = np.arange(stP, edP+1, stepP)
    X, Y = np.meshgrid(x, y)    # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值



    '''提取值'''
    # Reading LULC Tiff and Reshaping it
    img_proj, img_geotrans, img_data = EsRaster.read_img(lucctif)
    imgdata = img_data.reshape((img_data.shape[0] * img_data.shape[1],))
    imgdata[imgdata <= 0] = 0 #np.nan
    imgdata[imgdata >= 18] = 0 #np.nan

    # Tick time
    t1 = time.time()
    gtp = []

    for ci in range(1,17+1):
        print(str(ci)+'-')
        # set df data
        gtpi = [[],[]] #[T,P]
        # gtpi = np.array([[],[]]) #[T,P]
        for yr in range(stY,edY+1):
            # print(glob(filePath+os.sep+keyName[vi]+os.sep+r'*'+str(yr)+r'*.tif')[0])
            img_proj,img_geotrans,img_T = EsRaster.read_img(glob(TPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
            img_proj,img_geotrans,img_P = EsRaster.read_img(glob(PPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
            imgT = img_T.reshape((img_T.shape[0]*img_T.shape[1],))
            imgP = img_P.reshape((img_P.shape[0]*img_P.shape[1],))
            imgT[imgT<-9000] = np.nan
            imgP[imgP<-9990] = np.nan
            imgT *= 0.1
            imgP *= 0.1
            gtpi[0] += list(imgT[imgdata==ci])
            gtpi[1] += list(imgP[imgdata==ci])
        gtp.append(gtpi)

    '''选出三倍STD值'''
    for gtpi,ci in zip(gtp,range(1,17+1)):
        mean0 = np.nanmean(gtpi[0])
        sd0 = np.nanstd(gtpi[0])
        mean1 = np.nanmean(gtpi[1])
        sd1 = np.nanstd(gtpi[1])
        jud = list(np.where((gtpi[0] > (mean0 - 3 * sd0)) & (gtpi[0] < (mean0 + 3 * sd0))
                      & (gtpi[1] > (mean1 - 3 * sd1)) & (gtpi[1] > (mean1 - 3 * sd1)),
                      True, False))
        gtpi[0]=np.array(gtpi[0])[jud]
        gtpi[1]=np.array(gtpi[1])[jud]


    '''只做每种草地类型的dfi'''
    dflist = []
    for gtpi,ci in zip(gtp,range(1,17+1)):
        df = pd.DataFrame([[-1 for a in range(stT, edT + 1, stepT)] for yr in range(stP, edP + 1, stepP)])
        df.set_index(y, inplace=True)
        df.columns = x
        for T in range(stT,edT+1,stepT):
            print(T,end=',')
            for P in range(stP,edP+1,stepP):
                jud = np.where((gtpi[0] >= T) & (gtpi[0] < T + stepT) &
                               (gtpi[1] >= P) & (gtpi[1] < P + stepP)
                               , True, False)

                value = int(len(jud[jud])>0)
                df.iloc[int((P-stP)/stepP),int((T-stT)/stepT)] = value
        print('\b'*200,end='')
        dflist.append(df)
        print('Using Time:'+str(time.time()-t1)+'s')



    '''保存为Excel'''
    # for vi in range(len(keyName)):
    #     dflist[vi].to_csv(r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_'+keyName[vi]+r'_Tstep'+str(stepT)+r'.csv')
    #     time.sleep(60)

    # '''读取Excel'''
    # dflist = []
    # for vi in range(len(keyName)):
    #     dfi = pd.read_csv(r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\MetoAnalysis\Meto_'+keyName[vi]+r'_Tstep'+str(stepT)+r'.csv',
    #                         index_col=0)
    #     for T in range(stT, edT + 1, stepT):
    #         for P in range(stP, edP + 1, stepP):
    #             if len(dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)])>32766:
    #                 exec('dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]='+dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]+']')
    #             else:
    #                 exec('dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)]='+dfi.iloc[int((P-stP)/stepP),int((T-stT)/stepT)])
    #     dflist.append(dfi)


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
    # ddata = np.array(dflist_[vi], dtype=np.float64)
    fig = plt.figure(figsize=(14 / 2.54, 8 / 2.54))
    axi = plt.gca().axes
    # axi = fig.add_subplots(121)

    # datadf[datadf.isna()]=-1
    vmins = [0,0,0,0]
    vmaxs = [500,1500,2500,1]
    txtl = ['(a)','(b)','(c)','(d)']
    fmt = ['%.0f','%.0f','%.0f','%.1f']
    segN = [6,4,6,4]

    for ci in range(len(dflist)):#17+1):
        ddata = dflist[ci] # pd.DataFrame(dflist_[vi],dtype=np.float64)


        # Heatmap
        h = sns.heatmap(ddata, cmap=None,
                        linewidths=0, cbar=False,  # edgecolors=None,
                        cbar_kws={'drawedges': False}, mask=np.where(ddata < 1, True, False),
                        alpha=0).invert_yaxis()  # ,square=True
        # h = sns.heatmap(ddata,cmap='Spectral',
        #                 linewidths=0,cbar=True,edgecolors='face',
        #                 cbar_kws={'drawedges':False}#,mask =
        #                 ).invert_yaxis() #,square=True
        plt.xlabel('Temperature',fontsize=8, color='k') #x轴label的文本和字体大小
        plt.ylabel('Precipitation',fontsize=8, color='k') #y轴label的文本和字体大小
        # axi.set_xticks(x)
        # axi.set_xticklabels(x, rotation='horizontal')
        # plt.title(keyName[ci],fontsize=8) #图片标题文本和字体大小
        # 平滑处理
        # Resample your data grid by a factor of 3 using cubic spline interpolation.
        # ddata_sample = ddata  # scipy.ndimage.zoom(ddata, 1)
        # ddata_sample[ddata_sample.isna()] = 1
        contour = plt.contour(ddata,
                              [0.5],
                              colors=cc[ci])
        plt.clabel(contour, fontsize=8, colors=cc[ci], fmt=str(ci)) #

        # Set Parameters
        # axi.text(1,27,txtl[vi])
        # axi.set_xicks(ddata.columns[::stepT])
        # axi.set_yticks(ddata.index[::stepT])
        # # axi.set_xticklabels(ddata.columns[::stepT])
        # # axi.set_yticklabels(ddata.index[::stepT])
        # # 把x轴的刻度间隔设置为1，并存在变量里
        # x_major_locator = MultipleLocator(2)
        # # 把y轴的刻度间隔设置为10，并存在变量里
        # y_major_locator = MultipleLocator(100)
        # # 把x轴的主刻度设置为1的倍数
        # axi.xaxis.set_major_locator(x_major_locator)
        # # 把y轴的主刻度设置为10的倍数
        # axi.yaxis.set_major_locator(y_major_locator)
    Leg = [Line2D([0], [0], color=cc[li], ls='-', lw=2) for li in range(len(dflist))]
    LegTxt = [str(li)+'-'+coll[li] for li in range(len(dflist))]
    plt.legend(Leg, LegTxt, prop={'family':'SimHei'},ncol=1,columnspacing=0.2,
               loc='center right',bbox_to_anchor=(1.7, 0.5))  # bbox_to_anchor=(0.9, 0.05)
    sns.despine(ax=axi, top=False, right=False, left=False, bottom=False)
    plt.subplots_adjust(top=0.9,bottom=0.15,
                        right=0.55, left=0.1,
                        wspace=0.5, hspace=0.6)
    # plt.tight_layout()
    plt.savefig(r'G:\1_BeiJingUP\AUGB\Pic\Meto\气温降水-Heatmap-Grass-StepT'+str(stepT)+'-'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.jpg',dpi = 800)
    plt.show()

print()
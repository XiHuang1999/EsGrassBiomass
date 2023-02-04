#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/18 19:56
# @Author : Xihuang O.Y.

import os, sys, time, math
from datetime import datetime
from glob import glob
import pandas as pd
import numpy as np
sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig
modelPath = r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format'
excelFile = r"G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\SAMPLE_Above- and belowground_global_grassland_NPP data.xlsx"
dynamicParasD = ['GLASS','GLOPEM-CEVSA','MODIS','MuSyQ']
allcsv = pd.read_excel(excelFile, header=0, sheet_name=r'only_single_year')


'''## Sample ##'''
print(r'Sample : ',end=r'')
allyr = pd.DataFrame([])
for yr in range(2002,2015+1):
    yrcsv = allcsv[allcsv['Sampling year']==yr]
    print(yr)
    if yrcsv.shape[0]==0:
        print(str(yr)+r'无数据跳过')
        continue
    # '''Static raster Sample'''
    # stcDf = EsRaster.SampleRaster_gdal3(staticPath, yrcsv, 'ID')
    # yrcsv = pd.concat([yrcsv,stcDf],axis=1,join='outer')

    '''Dynamic raster sample'''
    dymtif = [glob(modelPath+os.sep+dymKey+r'*'+os.sep+r'*'+str(yr)+r'*.tif') for dymKey in dynamicParasD]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    dymtif = sum(dymtif, [])    # 解决List嵌套
    dymDf = EsRaster.SampleRaster_gdal3(dymtif,yrcsv,'ID',
                                        LonFieldName=r'Longitude', LatFieldName=r'Latitude',colName=dynamicParasD,
                                        nirCellNum=7)
    yrcsv = pd.concat([yrcsv, dymDf], axis=1, join='outer')

    '''Merge'''
    # yrcsv = yrcsv[(yrcsv[r'FPAR0_FPAR'] > 0) | (yrcsv[r'FPAR1_FPAR'] > 0) | (yrcsv[r'FPAR2_FPAR'] > 0)]
    allyr = pd.concat([allyr, yrcsv], axis=0, join='outer')

print('')
allyr.to_excel(r'G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\Sample_OUT_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.xlsx')
aa = allyr.groupby(['Latitude','Longitude','Sampling year']).agg({'TNPP':'median','GLASS':'mean','GLOPEM-CEVSA':'mean','MODIS':'mean','MuSyQ':'mean'})
aa = aa.reset_index()
aa.to_excel(r"G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\Sample_OUT_221118(210534)_GroupBy.xlsx")

'''Select'''
import matplotlib.pyplot as plt
plt.scatter(aa.loc[0:,'TNPP'],aa.loc[0:,'GLOPEM-CEVSA'])
bb = aa[(aa.loc[0:,'TNPP']<1200) & ((aa.loc[0:,'GLOPEM-CEVSA']<780) | (aa.loc[0:,'TNPP']>591))]
bb = bb[((bb.loc[0:,'GLOPEM-CEVSA']>438) | (bb.loc[0:,'TNPP']<1000))]
plt.scatter(bb.loc[0:,'TNPP'],bb.loc[0:,'GLOPEM-CEVSA'])

import scipy.stats as st
for cc in dynamicParasD:
    print(cc,end='')
    linreg = st.linregress(pd.Series(bb.loc[0:,'TNPP'], dtype=np.float64),
                           pd.Series(bb.loc[0:,cc], dtype=np.float64))
    print(linreg)
    plt.scatter(bb.loc[0:, 'TNPP'], bb.loc[0:, cc])
bb.to_excel(r'G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\Sample_Select_NPP.xlsx')

''''''
# xy = np.vstack([y_train, outResults1])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # linreg = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
    # pltx = [x for x in np.linspace(start=0, stop=max(y_train), num=1000)]
    # plty = [linreg.slope * x + linreg.intercept for x in pltx]
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(pltx, plty, '-', color='red', alpha=0.8, linewidth=2, label='Fitting Line')    # color='#4169E1'
    # plt.plot(pltx, pltx, '-', color='black', alpha=0.8, linewidth=2, label='1:1')
    # plt.scatter(y_train, outResults1,c=z,s=1.3,cmap='Spectral')
    # plt.text(400,235,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
    #          '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
    #          '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
    #          ,fontsize=8,color = "r",fontweight='bold')
    # plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    # plt.xlabel('AGB Site Value')  # 添加x轴和y轴标签
    # plt.ylabel('Model Value')
    # # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'RF_results.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.show()
    # endregion
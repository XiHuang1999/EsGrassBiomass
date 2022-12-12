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
dynamicParasD = ['BMA']#['GLASS','GLOPEM-CEVSA','MODIS','MuSyQ']
allcsv = pd.read_excel(excelFile, header=0, sheet_name=r'only_single_year')
staticPath = [r"G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif"]
out = r'G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\自己选-第二次试验'

'''## Sample ##'''
print(r'Sample : ',end=r'')
allyr = pd.DataFrame([])
for yr in range(2002,2015+1):
    yrcsv = allcsv[allcsv['Sampling year']==yr]
    print(yr)
    if yrcsv.shape[0]==0:
        print(str(yr)+r'无数据跳过')
        continue
    '''Static raster Sample'''
    stcDf = EsRaster.SampleRaster_gdal3(staticPath, yrcsv, 'ID')
    yrcsv = pd.concat([yrcsv,stcDf],axis=1,join='outer')

    '''Dynamic raster sample'''
    dymtif = [glob(modelPath+os.sep+dymKey+r'*Tiff'+os.sep+r'*'+str(yr)+r'*.tif') for dymKey in dynamicParasD]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    dymtif = sum(dymtif, [])    # 解决List嵌套
    dymDf = EsRaster.SampleRaster_gdal3(dymtif,yrcsv,'ID',
                                        LonFieldName=r'Longitude', LatFieldName=r'Latitude',colName=dynamicParasD,
                                        nirCellNum=7)
    yrcsv = pd.concat([yrcsv, dymDf], axis=1, join='outer')

    '''Merge'''
    # yrcsv = yrcsv[(yrcsv[r'FPAR0_FPAR'] > 0) | (yrcsv[r'FPAR1_FPAR'] > 0) | (yrcsv[r'FPAR2_FPAR'] > 0)]
    allyr = pd.concat([allyr, yrcsv], axis=0, join='outer')

print('')
#allyr.to_excel(out + os.sep + 'Sample_OUT_'+(datetime.today().strftime("%y%m%d(%H%M%S)"))+'.xlsx')
aa = allyr.copy()
aa[["TNPP"]] = allyr.groupby(['Latitude','Longitude','Sampling year'])[["TNPP"]].transform("mean")
aa[["GLASS",'GLOPEM-CEVSA','MODIS','MuSyQ']] = allyr.groupby(['Latitude','Longitude','Sampling year'])[["GLASS",'GLOPEM-CEVSA','MODIS','MuSyQ']].transform("mean")
aa = aa.drop_duplicates(subset=['Latitude','Longitude','Sampling year'],keep='first')
aa = aa.reset_index()
# aa.to_excel(out+os.sep+r"Sample_OUT_221118(210534)_GroupBy.xlsx")

'''Select'''
import matplotlib.pyplot as plt
plt.scatter(aa.loc[0:,'TNPP'],aa.loc[0:,'GLOPEM-CEVSA'])
bb = aa.copy()
for cc in ["TNPP","GLASS",'GLOPEM-CEVSA','MODIS','MuSyQ']:
    npp = list(aa.loc[0:,'TNPP'])
    mean = np.mean(npp)
    std = np.std(npp)
    aa = aa[(aa.loc[0:,'TNPP']>(mean-2*std)) & (aa.loc[0:,'TNPP']<(mean+2*std))]
    # bb = bb[((bb.loc[0:,'GLOPEM-CEVSA']>438) | (bb.loc[0:,'TNPP']<1000))]
plt.scatter(aa.loc[0:,'TNPP'],aa.loc[0:,'GLOPEM-CEVSA'])


# bb = pd.read_excel(r"G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\自己选-第二次试验\【有规律】Sample_OUT_221118(210534)_GroupBy.xlsx",sheet_name=r'选中数据')
import scipy.stats as st
for cc in dynamicParasD:
    print(cc,end='')
    linreg = st.linregress(pd.Series(bb.loc[0:,'TNPP'], dtype=np.float64),
                           pd.Series(bb.loc[0:,cc], dtype=np.float64))
    print(linreg)
    plt.scatter(bb.loc[0:, 'TNPP'], bb.loc[0:, cc])
# bb.to_excel(r'G:\1_BeiJingUP\AUGB\RatioAB\Data_S1\Data_S1\Sample_Select_NPP.xlsx')



'''数据验证'''
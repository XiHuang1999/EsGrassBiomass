# -*- coding: utf-8 -*-

# @Time : 2022-06-29 15:41
# @Author : XiHuang O.Y.
# @Site : 
# @File : EsInitial.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os, sys
from glob import glob

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster

if __name__=="__main__":
    # ==== Static Paramerters====
    # SiteExcel
    siteFile = r'G:\1_BeiJingUP\AUGB\Table\ALL_SITES2010.csv'
    sitedf = pd.read_csv(siteFile,header=0)
    # Topography
    dem = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif'
    lat = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\LAT_China1km.tif'
    slp = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\slope_china1km.tif'
    asp = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\aspect_china1km.tif'
    # Vegetation
    cgc = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\cgrass_China1km.tif'
    staticData = [dem,lat,slp,asp,cgc]
    staticKey = [f.split(os.sep)[-1].split(r'.')[0] for f in staticData]

    # ==== Active Paramerters=====
    activePath = [r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG',
                   r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI',
                   r'G:\1_BeiJingUP\AUGB\Data\20220629\PRCP',
                   r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS']
    activeCalc = ['mean/8','max/1000','sum','sum']

    activeKey = [f.split(os.sep)[-1] for f in activePath]
    activeData = [] #dict(zip(activeKey,activePath))

    # Active Data
    for iactive in range(len(activePath)):
        filelist = glob(activePath[iactive]+os.sep+r'*'+activeKey[iactive]+r'*.tif')
        filelist.sort()
        if activeKey[iactive] == 'NDVI':
            startDate = 4  # 5月开始
            endDate = 7  # 7月结束
        else:
            startDate = 16  # 第16个8天开始
            endDate = 30  # 第30个8天结束
        filelist = filelist[startDate:endDate]
        value = EsRaster.SampleRaster(filelist, sitedf, 'SiteName')
        if activeCalc[iactive].lower() == 'mean/3':
            valueCalc = [sum(v)/(endDate-startDate) for v in value]
        elif activeCalc[iactive].lower() == 'mean/8':
            valueCalc = [sum(v)/(endDate-startDate) for v in value]
        elif activeCalc[iactive].lower() == 'max/1000':
            valueCalc = [max(v)/1000 for v in value]
        else:
            valueCalc = [sum(v) for v in value]
        sitedf[activeKey[iactive]] = valueCalc
        # activeData.append(EsRaster.RasterCalc(filelist,activeCalc[iactive]))

    # Static Data
    filelist = staticData
    value = EsRaster.SampleRaster(filelist, sitedf, 'SiteName')
    rfdf = pd.concat([sitedf, pd.DataFrame(value)], axis=1, join='outer')
    rfdf.columns = list(sitedf.columns) + staticKey
    # activeData.append(EsRaster.RasterCalc(filelist,activeCalc[iactive]))


    print('')
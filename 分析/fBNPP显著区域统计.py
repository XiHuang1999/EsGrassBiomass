#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/25 16:46
# @Author : Xihuang O.Y.

### 没有删除异常值 ####
import os, sys, time, math
from glob import glob
import pandas as pd
import numpy as np
import scipy.stats as st

sys.path.append(r'D:\Pycharm\PyCharmPythonFiles\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

fbnppf = r"G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\fBNPP_2000to2018_slope_显著区域True.tif"
img_proj,img_geotrans,fbnpp = EsRaster.read_img(fbnppf)
# fbnpp[fbnpp<-9990] = np.nan

tifp = [r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\TAVG0',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\PRCP0',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\SWRS0',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\CO2',
        r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\PostProcess_CGrass',
        r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\BMA_CGrass\fBNPP'
        ]

outdt = []
for ci in range(len(tifp)):
    cidt = []
    for yr in range(2000,2018+1):
        tiff = glob(tifp[ci]+os.sep+r"*"+str(yr)+r'*.tif')[0]
        img_proj, img_geotrans, tif = EsRaster.read_img(tiff)
        tif[tif<-9990] = np.nan
        tif = tif[fbnpp >= -9990]
        cidt.append(np.nanmean(tif))
    outdt.append(cidt)

outdt = pd.DataFrame(outdt)
outdt.to_excel(r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\CHN显著\CHN2.xlsx')




### 删除异常值 ####

import os, sys, time, math
from glob import glob
import pandas as pd
import numpy as np
import scipy.stats as st

sys.path.append(r'D:\Pycharm\PyCharmPythonFiles\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

fbnppf = r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\fBNPP_2000to2018_slope_显著区域True_只在Livestock区域.tif'#r"G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\AnalysisBMA_Mean_Slp\fBNPP_2000to2018_slope_显著区域True.tif"
img_proj,img_geotrans,fbnpp = EsRaster.read_img(fbnppf)
# fbnpp[fbnpp<-9990] = np.nan

tifp = [r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\TAVG0',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\PRCP0',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\SWRS0',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\CO2',
        r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\PostProcess_CGrass',
        r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan\BMA_CGrass\fBNPP',
        r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2013\Slope_2000_2013_2\Livestock\CGrassRegion'
        ]

outdt = []
for yr in range(2000,2018+1):
    cidt = []
    for ci in range(len(tifp)):
        tiff = glob(tifp[ci]+os.sep+r"*"+str(yr)+r'*.tif')[0]
        img_proj, img_geotrans, tif = EsRaster.read_img(tiff)
        tif[tif<-9990] = np.nan
        tif = tif[fbnpp >= -9990]
        cidt.append(np.nanmean(tif))
    outdt.append(cidt)

outdt = pd.DataFrame(outdt)
outdt.to_excel(r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\CHN显著\CHN_更新Livestock.xlsx')
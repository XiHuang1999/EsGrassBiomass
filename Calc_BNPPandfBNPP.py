#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/3 19:07
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time
from glob import glob

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig

# # 初始化参数
wks1 = r'G:\1_BeiJingUP\AUGB\Data\NPP\IntegResults\Mean_Year'
wks2 = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_4\ExtractByCGrass_Set-9999Null_QTP_QF'
bnppOutway = r'G:\1_BeiJingUP\AUGB\Data\BNPP_MedianNPP_RF4'
fbnppOutway = r'G:\1_BeiJingUP\AUGB\Data\fBNPP_MedianNPP_RF4'
# 替补TNPP
wks3 = [r'G:\1_BeiJingUP\AUGB\Data\NPP\IntegResults\Median_Year',
r'G:\1_BeiJingUP\AUGB\Data\NPP\IntegResults\Weight_Year',
r'G:\1_BeiJingUP\AUGB\Data\NPP\IntegResults\BMA_Year',
r'G:\1_BeiJingUP\AUGB\Data\NPP\1Y_\GLASS_2000_2017_1y',
r'G:\1_BeiJingUP\AUGB\Data\NPP\1Y_\Geodata_2000_2017_1y',
r'G:\1_BeiJingUP\AUGB\Data\NPP\1Y_\TPDC_2000_2017_1y',
r'G:\1_BeiJingUP\AUGB\Data\NPP\1Y_\W_2000_2017_1y',
r'G:\1_BeiJingUP\AUGB\Data\NPP\1Y_\MODIS_2000_2017_1y',
r'G:\1_BeiJingUP\AUGB\Data\NPP\IntegResults\Multiply_Regression_Year']
startY = 2003
endY = 2017
ftif = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_4\ExtractByCGrass_Set-9999Null_QTP_QF\RF_AGB_2000.tif'
R, cc, lucc = EsRaster.read_img(ftif)
# lucc_rs = np.reshape(lucc, [lucc.shape[0] * lucc.shape[1], 1])

# # Calc BNPP and fBNPP
for yr in range(startY,endY):
    nppFile = glob(wks1+'\*'+str(yr)+'*.tif')[0]
    anppFile = glob(wks2+'\*'+str(yr)+'*.tif')[0]

    # # Read tiff
    R, cc, npp = EsRaster.read_img(nppFile) # imread[ff]
    npp[lucc < -9000] = np.nan
    npp[npp < 0] = np.nan
    R, cc, anpp = EsRaster.read_img(anppFile) # imread[ff]
    anpp[lucc < -9000] = np.nan
    anpp[anpp < 0] = np.nan
    # # BNPP
    bnpp = npp - anpp
    # # Post BNPP
    bnpp = np.reshape(bnpp, [lucc.shape[0]*lucc.shape[1], 1])
    if len(bnpp[bnpp<0]) > 0:
        for nppwksi in range(1,len(wks3)):
            nppTif = glob(wks3[nppwksi] + '\*' + str(yr) + '*.tif')[0]
            R, cc, nppi = EsRaster.read_img(nppTif) # imread[ff]
            nppi[lucc < -9000] = np.nan
            nppi[nppi < 0] = np.nan
            bnppi = nppi - anpp

            print('小于0像元数:'+str(len(bnpp[bnpp<0])),end='->')
            bnppi = np.reshape(bnppi, [lucc.shape[0]*lucc.shape[1], 1])
            bnpp[bnpp < 0] = bnppi[bnpp < 0]
            if len(bnpp[bnpp<0]) == 0:
                print(r'|| Break => ' + nppwksi)
                break
            print()

    bnpp = np.reshape(bnpp, [lucc.shape[0],lucc.shape[1]])

    # bnpp[bnpp < 0] = np.nan
    # bnpp[bnpp < 0] = -9999
    # bnpp[isnan[bnpp]] = -9999
    fout = bnppOutway + r'\\' + 'BNPP_Mean-RF4_' + str(yr) + r'.tif'
    EsRaster.write_img(fout, R, cc, bnpp)

    fbnpp = bnpp / npp
    # fbnpp[fbnpp < 0] = -9999
    # fbnpp[isnan[fbnpp]] = -9999
    fout = fbnppOutway + r'\\' + 'fBNPP_Mean-RF4_' + str(yr) + r'.tif'
    EsRaster.write_img(fout, R, cc, fbnpp)

print('Done!')

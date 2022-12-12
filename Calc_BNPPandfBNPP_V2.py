#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/3 19:07
# @Author : Xihuang O.Y.
'''
使用GLCV-TNPP 和 RF5-ANPP 计算BNPP
'''
import pandas as pd
import numpy as np
import os, sys, time
from glob import glob

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig

# # 初始化参数
wks1 = r'G:\1_BeiJingUP\QTP_SST\Data\NPP\GLOPEM_CEVSA_Tiff_Chinese2'
wks2 = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_5\Set-9999toNull_Grass'
bnppOutway = r'G:\1_BeiJingUP\AUGB\Data\calcBNPP_2\BNPP_GLCVNPP_RF5'
fbnppOutway = r'G:\1_BeiJingUP\AUGB\Data\calcBNPP_2\fBNPP_GLCVNPP_RF5'
# 替补TNPP
wks3 = []
startY = 2000
endY = 2020
ftif = r'G:\1_BeiJingUP\CommonData\China\CGrassChina_OnlyGrass1_Greater0.tif'#'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_4\Set-9999toNull\RF_AGB_2020.tif'
R, cc, lucc = EsRaster.read_img(ftif)
# lucc_rs = np.reshape(lucc, [lucc.shape[0] * lucc.shape[1], 1])

# # Calc BNPP and fBNPP
for yr in range(startY,endY+1):
    print(yr)
    nppFile = glob(wks1+'\*'+str(yr)+'*.tif')[0]
    anppFile = glob(wks2+'\*'+str(yr)+'*.tif')[0]

    '''Read tiff'''
    # # ANPP
    R, cc, anpp = EsRaster.read_img(anppFile) # imread[ff]
    anpp[lucc > 18] = np.nan
    anpp[anpp < 0] = np.nan

    # # TNPP
    R, cc, npp = EsRaster.read_img(nppFile) # imread[ff]
    npp[lucc > 18] = np.nan
    npp[np.where(np.isnan(anpp))] = np.nan
    # npp /= 0.45


    # # BNPP
    bnpp = npp - anpp

    # # # Post BNPP
    # bnpp = np.reshape(bnpp, [lucc.shape[0]*lucc.shape[1], 1])
    # if len(bnpp[bnpp<0]) > 0:
    #     for nppwksi in range(1,len(wks3)):
    #         nppTif = glob(wks3[nppwksi] + '\*' + str(yr) + '*.tif')[0]
    #         R, cc, nppi = EsRaster.read_img(nppTif) # imread[ff]
    #         nppi[lucc < -9000] = np.nan
    #         nppi[nppi < 0] = np.nan
    #         bnppi = nppi - anpp
    #
    #         print('小于0像元数:'+str(len(bnpp[bnpp<0])),end='->')
    #         bnppi = np.reshape(bnppi, [lucc.shape[0]*lucc.shape[1], 1])
    #         bnpp[bnpp < 0] = bnppi[bnpp < 0]
    #         if len(bnpp[bnpp<0]) == 0:
    #             print(r'|| Break => ' + nppwksi)
    #             break
    #         print()
    #
    # bnpp = np.reshape(bnpp, [lucc.shape[0],lucc.shape[1]])

    bnpp[bnpp < 0] = np.nan
    # bnpp[bnpp < 0] = -9999
    # bnpp[isnan[bnpp]] = -9999
    fout = bnppOutway + r'\\' + 'BNPP_GLCV-RF5_' + str(yr) + r'.tif'
    EsRaster.write_img(fout, R, cc, bnpp)

    fbnpp = bnpp / npp
    # fbnpp[fbnpp < 0] = -9999
    # fbnpp[isnan[fbnpp]] = -9999
    fout = fbnppOutway + r'\\' + 'fBNPP_GLCV-RF5_' + str(yr) + r'.tif'
    EsRaster.write_img(fout, R, cc, fbnpp)

print('Done!')

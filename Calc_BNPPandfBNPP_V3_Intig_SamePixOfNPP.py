#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/3 19:07
# @Author : Xihuang O.Y.
'''
使用每个模型的TNPP 和 RF5-ANPP 计算BNPP
并且只输出BNPP大于0的像元
'''
import pandas as pd
import numpy as np
import os, sys, time
from glob import glob

EsInitialPath = os.getcwd() #r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass'
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig

# # 初始化参数
wkstnpp = glob(r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format\BMA*Tiff')
print(wkstnpp)
wks2 = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_5\Set-9999toNull_Grass'
out = r'G:\1_BeiJingUP\AUGB\Data\EveryModel_NPP_3_Nan'
# Grassland Class
# 替补TNPP
wks3 = [
r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format\GLOPEM-CEVSA_1980_2020_1y_chinese2_Tiff',
    r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format\GLASS_1982_2018_1y_chinese_Tiff',
r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format\MuSyQ_1981_2018_1y_chinese_Tiff',
r'G:\1_BeiJingUP\QTP_SST\Data\NPP\Tiff_Format\MODIS_2000_2017_1y_chinese_Tiff']
startY = 2000
endY = 2018
ftif = r'G:\1_BeiJingUP\CommonData\China\CGrassChina_OnlyGrass1_Greater0.tif'#'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_4\Set-9999toNull\RF_AGB_2020.tif'
R, cc, lucc = EsRaster.read_img(ftif)
# lucc_rs = np.reshape(lucc, [lucc.shape[0] * lucc.shape[1], 1])

for wks1 in wkstnpp:
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
        npp[npp < 0] = np.nan
        npp[np.where(np.isnan(anpp))] = np.nan
        if 'GLOPEM-CEVSA'.lower() not in wks1.lower():
            npp /= 0.45

        # # BNPP
        bnpp = npp - anpp
        fbnpp = bnpp/npp

        # # Post BNPP
        if len(bnpp[bnpp<0]) > 0:
            for nppwksi in range(0,len(wks3)):
                nppTif = glob(wks3[nppwksi] + '\*' + str(yr) + '*.tif')[0]
                R, cc, nppi = EsRaster.read_img(nppTif) # imread[ff]
                nppi[lucc < -9000] = np.nan
                nppi[nppi < 0] = np.nan
                if 'GLOPEM-CEVSA'.lower() not in wks1.lower():
                    nppi /= 0.45
                bnppi = nppi - anpp

                print('小于0像元数:'+str(len(bnpp[bnpp<0])),end='->')
                # bnppi = np.reshape(bnppi, [lucc.shape[0]*lucc.shape[1], 1])
                bnpp[bnpp < 0] = bnppi[bnpp < 0]
                if len(bnpp[bnpp<0]) == 0:
                    print(r'|| Break => ' + nppwksi)
                    break
                print()
        # bnpp = np.reshape(bnpp, [lucc.shape[0],lucc.shape[1]])

        '''Drop Nan'''
        npp[np.where(np.isnan(anpp))] = np.nan
        npp[np.where(bnpp<0)] = np.nan
        anpp[np.where(bnpp<0)] = np.nan
        fbnpp[np.where(bnpp<0)] = np.nan
        bnpp[bnpp < 0] = np.nan

        npp = anpp + bnpp
        fbnpp = bnpp/npp

        '''Save TIFF'''
        outWks = out + os.sep + wks1.split(os.sep)[-1].split(r'_')[0]
        if not os.path.exists(outWks):
            os.mkdir(outWks)
        #TNPP
        tnpp_outWks = out + os.sep + wks1.split(os.sep)[-1].split(r'_')[0] + os.sep + r'TNPP'
        if not os.path.exists(tnpp_outWks):
            os.mkdir(tnpp_outWks)
        fout = tnpp_outWks + os.sep + str(yr) + r'.tif'
        EsRaster.write_img(fout, R, cc, npp)
        #ANPP
        anpp_outWks = out + os.sep + wks1.split(os.sep)[-1].split(r'_')[0] + os.sep + r'ANPP'
        if not os.path.exists(anpp_outWks):
            os.mkdir(anpp_outWks)
        fout = anpp_outWks + os.sep + str(yr) + r'.tif'
        EsRaster.write_img(fout, R, cc, anpp)
        #BNPP
        bnpp_outWks = out + os.sep + wks1.split(os.sep)[-1].split(r'_')[0] + os.sep + r'BNPP'
        if not os.path.exists(bnpp_outWks):
            os.mkdir(bnpp_outWks)
        fout = bnpp_outWks + os.sep + str(yr) + r'.tif'
        EsRaster.write_img(fout, R, cc, bnpp)
        #fBNPP
        fbnpp_outWks = out + os.sep + wks1.split(os.sep)[-1].split(r'_')[0] + os.sep + r'fBNPP'
        if not os.path.exists(fbnpp_outWks):
            os.mkdir(fbnpp_outWks)
        fout = fbnpp_outWks + os.sep + str(yr) + r'.tif'
        EsRaster.write_img(fout, R, cc, fbnpp)

print('Done!')

# import win32api,win32con
# win32api.MessageBox(0,"Pycharm Process","Done!",win32con.MB_OK)
# from tkinter import messagebox
# print("这是一个弹出提示框")
# messagebox.showinfo("提示","Done!")

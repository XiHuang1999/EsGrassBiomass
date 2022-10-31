#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/2 11:03
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time
from glob import glob

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig

anppPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_5\Set-9999toNull_Grass'
tnppPath = r'G:\1_BeiJingUP\QTP_SST\Data\NPP\GLOPEM_CEVSA_Tiff_Chinese2'
bnppPath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\BNPP'
fnppPath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan\fBNPP'
keyName = [r'ANPP',r'TNPP',r'BNPP',r'fBNPP']
outPath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP_2_Nan'
stY = 2000
edY = 2020

# Read BNPP as STD
# data = pd.DataFrame([])   #columns=list(range(stY,edY,1))
# for yr in range(stY,edY+1):
#     img_proj,img_geotrans,img_data = EsRaster.read_img(glob(bnppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
#     data[str(yr)] = img_data.reshape((img_data.shape[0]*img_data.shape[1],))
#
# # 非空索引
# avind = data.index[data.notnull().T.all()]
# data_ = data
# data[data<0] = np.nan
# dvind = data.index[data.isnull().T.any()]


for yr in range(stY,edY+1):
    # ANPP
    img_proj,img_geotrans,img_data1 = EsRaster.read_img(glob(anppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    # imgdata1 = img_data.reshape((img_data.shape[0]*img_data.shape[1],))
    # imgdata1[dvind] = np.nan
    # imgdata1 = imgdata1.reshape((img_data.shape[0], img_data.shape[1]))
    EsRaster.write_img(outPath+os.sep+keyName[0]+os.sep+r'ANPP_'+str(yr)+r'.tif',img_proj,img_geotrans,img_data1)

    # BNPP
    img_proj,img_geotrans,img_data2 = EsRaster.read_img(glob(bnppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    # imgdata2 = img_data.reshape((img_data.shape[0] * img_data.shape[1],))
    # imgdata2[dvind] = np.nan
    # imgdata2 = imgdata2.reshape((img_data.shape[0], img_data.shape[1]))
    # EsRaster.write_img(outPath + os.sep + keyName[2] + os.sep + r'BNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
    #                    imgdata2)

    # TNPP
    imgdata3 = img_data1+img_data2
    EsRaster.write_img(outPath + os.sep + keyName[1] + os.sep + r'TNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
                       imgdata3)

    # fBNPP
    imgdata4 = img_data2 / imgdata3
    EsRaster.write_img(outPath + os.sep + keyName[3] + os.sep + r'fBNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
                       imgdata4)

print()
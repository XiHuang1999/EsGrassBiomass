#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/11 11:22
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time
from glob import glob


sys.path.append(r'E:\A_UCAS_Study\PythonWorkspace\EsGrassBiomass')      # 添加函数文件位置
import EsRaster,readConfig

keyName = [r'ANPP',r'TNPP',r'BNPP',r'fBNPP']
outPath = r'G:\1_BeiJingUP\AUGB\Data\Analysis_NPP'
stY = 2003
edY = 2017


for yr in range(stY,edY+1):
    # ANPP
    img_proj,img_geotrans,img_data = EsRaster.read_img(glob(anppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    imgdata1 = img_data.reshape((img_data.shape[0]*img_data.shape[1],))
    imgdata1[dvind] = np.nan
    imgdata1 = imgdata1.reshape((img_data.shape[0], img_data.shape[1]))
    EsRaster.write_img(outPath+os.sep+keyName[0]+os.sep+r'ANPP_'+str(yr)+r'.tif',img_proj,img_geotrans,imgdata1)

    # BNPP
    img_proj,img_geotrans,img_data = EsRaster.read_img(glob(bnppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    imgdata2 = img_data.reshape((img_data.shape[0] * img_data.shape[1],))
    imgdata2[dvind] = np.nan
    imgdata2 = imgdata2.reshape((img_data.shape[0], img_data.shape[1]))
    EsRaster.write_img(outPath + os.sep + keyName[2] + os.sep + r'BNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
                       imgdata2)

    # TNPP
    img_proj,img_geotrans,img_data = EsRaster.read_img(glob(tnppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    # imgdata3 = img_data.reshape((img_data.shape[0] * img_data.shape[1],))
    # imgdata3[dvind] = np.nan
    # imgdata3 = imgdata3.reshape((img_data.shape[0], img_data.shape[1]))
    imgdata3 = imgdata1+imgdata2
    EsRaster.write_img(outPath + os.sep + keyName[1] + os.sep + r'TNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
                       img_data)

    # fBNPP
    img_proj,img_geotrans,img_data = EsRaster.read_img(glob(fnppPath+os.sep+r'*'+str(yr)+r'*.tif')[0])
    imgdata4 = img_data.reshape((img_data.shape[0] * img_data.shape[1],))
    imgdata4[dvind] = np.nan
    imgdata4 = imgdata4.reshape((img_data.shape[0], img_data.shape[1]))
    EsRaster.write_img(outPath + os.sep + keyName[3] + os.sep + r'fBNPP_' + str(yr) + r'.tif', img_proj, img_geotrans,
                       imgdata4)

print()
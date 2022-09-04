#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/3 19:07
# @Author : Xihuang O.Y.

import pandas as pd
import numpy as np
import os, sys, time
from glob import glob

EsInitialPath = os.getcwd[]
sys.path.append[EsInitialPath]      # 添加函数文件位置
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

# # Judge
ANPP and TNPP
first
Tiff
file
name
# if nppTifs[tifi].name[isstrprop[nppTifs[tifi].name, 'digit']] != anppTifs[tifi].name[
        isstrprop[nppTifs[tifi].name, 'digit']]

# # Calc
BNPP and fBNPP
for yr=startY:endY
nppTifs = dir[[wks1, '\*', num2str[yr], '*.tif']]
anppTifs = dir[[wks2, '\*', num2str[yr], '*.tif']]
nppFile = [wks1, filesep, nppTifs.name]
anppFile = [wks2, filesep, anppTifs.name]

# # Read
tiff
[npp, R, cc] = geotiffread[nppFile] # imread[ff]
npp[lucc < -9000] = nan
npp[npp < -9000] = nan
[anpp, R, cc] = geotiffread[anppFile] # imread[ff]
anpp[lucc < -9000] = nan
anpp[anpp < -9000] = nan
# # BNPP
bnpp = npp - anpp
# # Post
BNPP
bnpp = reshape[bnpp, [info.Height * info.Width, 1]]
if sum[sum[bnpp < 0]] > 100
    for nppwksi = 1:length[wks3]
    nppTifs = dir[cell2mat[[wks3[nppwksi], '\*', num2str[yr], '*.tif']]]
    [nppi, R, cc] = geotiffread[cell2mat[[wks3[nppwksi], filesep, nppTifs.name]]] # imread[ff]
    nppi[lucc < -9000] = nan
    nppi[nppi < -9000] = nan
    bnppi = nppi - anpp

    fprintf['小于0像元数: #d  /  ', sum[bnpp < 0]]
    bnppi = reshape[bnppi, [info.Height * info.Width, 1]]
    ind_bnpp = find[bnpp < 0]
    ind_bnppi = find[bnppi > bnpp]
    [ind, inda, indb] = intersect[ind_bnpp, ind_bnppi]
    bnpp[ind] = bnppi[ind]
    if sum[sum[bnpp < 0]] < 100
        disp[[num2str[nppwksi], '==>Break']]
        break
    end
end
end
bnpp = reshape[bnpp, [info.Height, info.Width]]

bnpp[bnpp < 0] = nan
# bnpp[bnpp < 0] = -9999
# bnpp[isnan[bnpp]] = -9999
# fout = [bnppOutway, '\','BNPP_Mean - RF4_',num2str[yr],'.tif']
          # geotiffwrite[fout, bnpp, R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag]
o_bnpp[yr - startY + 1,:] = reshape[bnpp, [1, info.Height * info.Width]]
o_anpp[yr - startY + 1,:] = reshape[anpp, [1, info.Height * info.Width]]
o_npp[yr - startY + 1,:] = reshape[npp, [1, info.Height * info.Width]]

# fbnpp = bnpp / npp
# fbnpp[fbnpp < 0] = nan
# fbnpp[fbnpp < 0] = -9999
# fbnpp[isnan[fbnpp]] = -9999
# fout = [fbnppOutway, '\','fBNPP_Mean - RF4_',num2str[yr],'.tif']
          # geotiffwrite[fout, bnpp, R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag]
# fprintf['#s \n', fout]
#
# x = npp[bnpp < 0]
# y = anpp[bnpp < 0]
#
#
# bnd = [-100 1100]
# img = bnpp
# img[isnan[img]] = 9999
# figure
imagesc[img, bnd]
colorbar
horz
colormap[[[1 0 0]
parula[12 - 2]
white[1]]]
#
# img = bnpp
# # img[isnan[img]] = -100
# sepB = -100:50: 900
# figure[1]
# hist[img, sepB]
#
#
# bnd = [-100 1100]
# img[isnan[img]] = 9999
# figure
imagesc[img, bnd]
colorbar
horz
colormap[[[1 0 0]
parula[12 - 2]
white[1]]]
#
# img = reshape[bnpp, [info.Height * info.Width, 1]]
# sepB = -100:50: 900
# figure[1]
# hist[[x, y], sepB]
# xlim[[-150, 150]]
# legend['npp', 'anpp']
#
# img1 = reshape[npp, [info.Height * info.Width, 1]]
# sepB = -100:50: 900
# figure[1]
# hist[img, sepB]
#
# img2 = reshape[anpp, [info.Height * info.Width, 1]]
# sepB = -100:50: 900
# figure[1]
# hist[[img1, img2], sepB]
#
# legend['npp', 'anpp']
end

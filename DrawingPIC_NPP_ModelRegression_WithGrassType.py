# -*- coding: utf-8 -*-

# @Time : 2022-08-15 21:38
# @Author : XiHuang O.Y.
# @Site : 
# @File : DrawingPIC_NPP_ModelRegression_WithGrassType.py
# @Software: PyCharm
# -*- coding: utf-8 -*-

# @Time : 2022-08-10 20:30
# @Author : XiHuang O.Y.
# @Site :
# @File : DrawingPIC_NPP_ModelRegression.py
# @Software: PyCharm
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pandas as pd
import sys, os, math

from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

## Parallel Processing of Raster
from rasterio.enums import Resampling
from rasterio.windows import Window
import time,multiprocessing
from multiprocessing import Manager
from glob import glob
import numpy as np
from ctypes import c_char_p
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling

def read_img(filename):
    """
    filename: str,要读取的栅格文件
    return: 函数输入图像所在的路径，输出图像的投影，仿射矩阵以及图像数组\n
    """
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    del dataset
    return im_proj,im_geotrans,im_data


# LAI Data
luccPath = r""
laiPath = r"J:\Integrated_analysis_data\Data\1Y\LAI_2003_2017_1y"
# data after normal Path
datNormalPath = [r"G:\1_BeiJingUP\AUGB\Data\NPP\Normal_Geodata",
                 r"G:\1_BeiJingUP\AUGB\Data\NPP\Normal_GLASS",
                 r"G:\1_BeiJingUP\AUGB\Data\NPP\Normal_MODIS",
                 r"G:\1_BeiJingUP\AUGB\Data\NPP\Normal_TPDC",
                 r"G:\1_BeiJingUP\AUGB\Data\NPP\Normal_W"]
# data Path
datPath = [r"J:\Integrated_analysis_data\Data\1Y\Geodata_2000_2017_1y",
           r"J:\Integrated_analysis_data\Data\1Y\GLASS_2000_2017_1y",
           r"J:\Integrated_analysis_data\Data\1Y\MODIS_2000_2017_1y",
           r"J:\Integrated_analysis_data\Data\1Y\TPDC_2000_2017_1y",
           r"J:\Integrated_analysis_data\Data\1Y\W_2000_2017_1y",
           r"G:\1_BeiJingUP\AUGB\Data\20220629\TAVG_Yearly_QTP",
           r"G:\1_BeiJingUP\AUGB\Data\20220629\PRCP_Yearly_QTP"]
# data file key word
dataKey = ['Mask_Mul_',
           'Mask_Mul_',
           'Mask_Mul_',
           'Mask_Reproject_Mul_',
           'Resample_Mask_',
           'Mask_Reproject_Mul_',
           r'TAVG_',
           r'PRCP_']
# The R square of the data and LAI
r2Path = [r"J:\Integrated_analysis_data\Data\Out\R2_Geodata\R2_geodata_setnodata.tif",
          r"J:\Integrated_analysis_data\Data\Out\R2_GLASS\R2_Glass_setnodata.tif",
          r"J:\Integrated_analysis_data\Data\Out\R2_MODIS\R2_Modis_setnodata.tif",
          r"J:\Integrated_analysis_data\Data\Out\R2_TPDC\R2_TPDC_setnodata.tif",
          r"J:\Integrated_analysis_data\Data\Out\R2_W\R2_W_setnodata.tif"]
lucc_proj,lucc_geotrans,lucc = read_img(luccPath)
mean = []
startY = 2003
endY = 2017

# LAI
mdlmean = []
for yr in range(startY,endY+1):
    tif = glob(laiPath + os.sep + str(yr) + os.sep + r"Mul*.tif")[0]
    img_proj, img_geotrans, laiyr = read_img(tif)
    laiyr[laiyr < 0] = np.nan
    if yr // 4 == 0:
        nday = 366
    else:
        nday = 365
    mdlmean.append(np.nanmean(laiyr)/nday)
mean.append(mdlmean)
# Model
for mdi in range(len(datPath)):
    mdlmean = []
    for yr in range(startY,endY+1):
        tif = glob(datPath[mdi] + os.sep + str(yr) + os.sep + r"*" + dataKey[mdi] + r"*.tif")[0]
        print(tif)
        img_proj,img_geotrans,img_data = read_img(tif)
        img_data[img_data<0] = np.nan
        mdlmean.append(np.nanmean(img_data))
    mean.append(mdlmean)

dfmean = pd.DataFrame(mean).T
dfmean.columns = ["MOD-LAI","Geodata","GLASS",r"MODIS",r"TPDC",r"UCAS"]
corr_mat = dfmean.corr()
f, ax = plt.subplots(figsize=(4, 4))
mask = np.zeros_like(corr_mat)
for i in range(1, len(mask)):
    for j in range(0, i):
        mask[j][i] = True
#注释3：此处就是标注显著性了，调节好坐标就行了
for col in range(dfmean.shape[1]):
    for row in range(col,dfmean.shape[1]):
        linreg = st.linregress(pd.Series(dfmean.iloc[:,col], dtype=np.float64), pd.Series(dfmean.iloc[:,row], dtype=np.float64))
        if linreg.pvalue < 0.01:
            print(col, ' and ', row, str(linreg.rvalue**2))
            plt.text(col+0.6, row+0.45, "**", size=12, alpha=2, color="Black")
        elif linreg.pvalue < 0.05:
            print(col, ' and ', row, str(linreg.rvalue**2))
            plt.text(col+0.6, row+0.45, "*", size = 12, alpha = 2, color = "Black")
        else:
            None
sns.heatmap(corr_mat, cmap='PiYG', annot=True, mask=mask, linewidths=.05, square=True, annot_kws={'size': 6.5, 'weight':'bold'}, fmt=".2f")
# print(corr_mat)
plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
plt.title(r'NPP Corr-Picture')      # 添加标题
# plt.savefig(r"G:\1_BeiJingUP\AUGB\Data\NPP\PIC"+os.sep+'Corr_Heatmap_5ModelAndLAI.png',dpi=500,bbox_inches='tight')#, transparent=True
plt.show()
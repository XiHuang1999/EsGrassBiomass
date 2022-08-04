# -*- coding: utf-8 -*-

# @Time : 2022-08-04 16:33
# @Author : XiHuang O.Y.
# @Site : 
# @File : PreProcessNDVI.py
# @Software: PyCharm
from arcpy import env
import arcpy
import os

arcpy.env.overwriteOutput=True
arcpy.CheckOutExtension(r"Spatial")
tifPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI'
#inShp = r'G:\1_BeiJingUP\CommonData\SJY\sjy_lucc_grass2015.shp'
masktif = r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG\TAVG_2000001.tif'
outPath1 = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\AfterPreProcess'

env.workspace = tifPath
arcpy.env.snapRaster = masktif   #捕捉栅格
arcpy.env.mask = masktif           #环境掩膜
arcpy.env.cellSize = masktif   #像元大小
arcpy.env.outputcoordinatesystem = masktif

tiffiles = arcpy.ListFiles(r"*.tif")
for tif in tiffiles:
    print(tif)
    #arcpy.gp.RasterCalculator_sa(r'SetNull("' + tif + r'"<=0,"' + tif + r'")', outPath1 + os.sep + tif)
    arcpy.gp.ExtractByMask_sa(tif, masktif, outPath1 + os.sep + tif)
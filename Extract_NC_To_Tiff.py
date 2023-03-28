#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time ： 2023/3/14 014 08:58:14
# @IDE ：PyCharm
# @Author : Xihuang Ouyang


import os
import netCDF4 as nc
import numpy as np
from osgeo import gdal, osr, ogr
import glob
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PROJ_LIB'] = r"D:\Pycharm\PyCharmPythonFiles\venv\Lib\site-packages\pyproj\proj_dir\share\proj"
os.environ['GDAL_DATA'] = r'D:\Pycharm\PyCharmPythonFiles\venv\Lib\site-packages\pyproj\proj_dir\share'

def nc2tif(data, Output_folder, varName='CO2'):
    tmp_data = nc.Dataset(data)  # 利用.Dataset()方法读取nc数据
    Lat_data = tmp_data.variables['lat'][:]
    Lon_data = tmp_data.variables['lon'][:]
    pre_data = tmp_data.variables[varName][:]
    tmp_arr = np.asarray(tmp_data.variables[varName])
    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [Lon_data.min(), Lat_data.max(), Lon_data.max(), Lat_data.min()]
    print(Lonmin, Latmax, Lonmax, Latmin)
    # 分辨率计算
    Num_lat = len(Lat_data)  # 5146
    Num_lon = len(Lon_data)  # 7849
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # print(Num_lat, Num_lon)
    # print(Lat_res, Lon_res)
    for i in range(len(tmp_arr[:])):
        # i=0,1,2,3,4,5,6,7,8,9,...
        # 创建tif文件
        driver = gdal.GetDriverByName('GTiff')
        out_tif_name = Output_folder + '\\' + data.split('\\')[-1].split('.')[0] + '_' + str(i + 1) + '.tif'
        out_tif = driver.Create(out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Int16)
        # 设置影像的显示范围
        # Lat_re前需要添加负号
        geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
        out_tif.SetGeoTransform(geotransform)
        # 定义投影
        prj = osr.SpatialReference()
        prj.ImportFromEPSG(4326)  # WGS84
        out_tif.SetProjection(prj.ExportToWkt())
        # 数据导出
        out_tif.GetRasterBand(1).WriteArray(tmp_arr[i])  # 将数据写入内存，此时没有写入到硬盘
        out_tif.FlushCache()  # 将数据写入到硬盘
        out_tif = None  # 关闭tif文件


def nc2tif_NoLonLat(data, Output_folder, varName='CO2'):
    tmp_data = nc.Dataset(data)  # 利用.Dataset()方法读取nc数据
    tmp_arr = np.asarray(tmp_data.variables[varName])

    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [-180,88,180,-60]
    print(Lonmin, Latmax, Lonmax, Latmin)
    # 分辨率计算
    Num_lat = tmp_data.dimensions['lat'].size  # (Latmax-Latmin)
    Num_lon = tmp_data.dimensions['lon'].size  # 7849
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon))
    # print(Num_lat, Num_lon)
    # print(Lat_res, Lon_res)
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = Output_folder + '\\' + data.split('\\')[-1].split('.')[0] + '.tif' #+ '_' + str(i + 1)
    out_tif = driver.Create(out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Float64)
    # 设置影像的显示范围
    # Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
    out_tif.SetGeoTransform(geotransform)
    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)  # WGS84
    out_tif.SetProjection(prj.ExportToWkt())
    # 数据导出
    out_tif.GetRasterBand(1).WriteArray(tmp_arr.T)  # 将数据写入内存，此时没有写入到硬盘
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


Input_folder = r"G:\1_BeiJingUP\GHG\DATA\GlobalSimulatedCO2_1992-2020\CO2_mean_nc"
Output_folder = r"G:\1_BeiJingUP\GHG\DATA\GlobalSimulatedCO2_1992-2020\CO2_mean_Tiff"
# 读取所有数据
data_list = glob.glob(os.path.join(Input_folder +os.sep+ r'*_20*.nc'))
print(data_list)

# 了解数据
data = data_list[0] # 输入文件
NC_DS = nc.Dataset(data)
print(NC_DS.variables) # 了解变量的基本信息

for i in range(len(data_list)):
    data = data_list[i]
    nc2tif_NoLonLat(data, Output_folder)
    print('转tif成功')


# -*- coding: utf-8 -*-

# @Time : 2022-08-16 16:59
# @Author : XiHuang O.Y.
# @Site : 
# @File : DrawingPIC_NPP_ModelIntegValidation.py
# @Software: PyCharm
## Raster Process
from osgeo import gdal
from osgeo import ogr
import pandas as pd
import numpy as np
import sys, os
from glob import glob

EsRasterPath = os.getcwd()
sys.path.append(EsRasterPath)      # 添加函数文件位置
import CoordSys
os.environ['PROJ_LIB'] = r"C:\Anaconda\Anaconda\Lib\site-packages\osgeo\data\proj"

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
def SampleRaster(tifList, ptShp, siteName=r'FID', nirCellNum=1, Scope=''):
    '''
    Using PtShp File to Sample Raster
    :param tifList: 栅格列表
    :param ptShp: str or DataFrame,点shp文件 or 第一、五列是站点代码、年份，第1列是Lon第2列是Lat的DataFrame
    :param siteName: 点字段名称
    :param nirCellNum: 方格大小，取像元附近均值（只能奇数）；默认值为1，为像元本身
    :param Scope: 筛选范围，eg：“>0” or ""
    :return: pd.DataFrame
    '''
    # 存放每个点的xy坐标的数组
    xValues = []
    yValues = []
    # 存放点“区站号”
    station_list = []
    ptNum = 0
    if isinstance(ptShp, str):
        # 数据驱动driver的open()方法返回一个数据源对象 0是只读，1是可写
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(ptShp, 0)
        if ds is None:
            print('打开矢量文件失败')
            sys.exit(1)
        else:
            print('打开矢量文件成功')
        # 读取数据层
        # 一般ESRI的shapefile都是填0的，如果不填的话默认也是0.
        layer = ds.GetLayer(0)
        # 要素个数(属性表行)
        ptNum = layer.GetFeatureCount()

        # 重置遍历对象（所有要素）的索引位置，以便读取
        layer.ResetReading()
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            geometry = feature.GetGeometryRef()
            x = geometry.GetX()
            y = geometry.GetY()
            # 将要分类的点的坐标和点的类型添加
            xValues.append(x)
            yValues.append(y)
            # 读取矢量对应”区站号“字段的值
            station = feature.GetField(siteName)
            station_list.append(station)

    elif isinstance(ptShp, pd.DataFrame):
        station_list = list(ptShp.iloc[:, 0])
        xValues = list(ptShp.iloc[:, 1])
        yValues = list(ptShp.iloc[:, 2])
        ptNum = len(xValues)

    # 创建二维空列表
    values = [[0 for col in range(len(tifList))] for row in range(len(station_list))]
    for tif, tif_i in zip(tifList, range(len(tifList))):
        # Execute ExtractByMask
        print(os.path.basename(tif), end='/')
        # # 打开文件
        dr = gdal.Open(tif)
        # 存储着栅格数据集的地理坐标信息
        transform = dr.GetGeoTransform()
        # Sample
        if 'NDVI' in os.path.basename(tif).upper():
            for i in range(len(station_list)):
                # LonLat to ColRow
                [x, y] = CoordSys.geo2imagexy(dr, xValues[i], yValues[i])
                # if math.isnan(x):
                #     print('sdf')
                # 判断方框滤波
                if nirCellNum == 1:
                    dt = dr.ReadAsArray(int(x), int(y), 1, 1)
                elif nirCellNum % 2 == 0:
                    print('nirCellNum：The box image element is set incorrectly and must be an odd number')
                    break
                else:
                    dt = dr.ReadAsArray(int(x - (nirCellNum - 1) / 2), int(y - (nirCellNum - 1) / 2), nirCellNum,
                                        nirCellNum)
                if dt is None:
                    print('EsRaster Error！ Cheack ReadAsArray at: ' + tif)
                valueList = dt.flatten()
                # 判断取值范围
                if Scope == '':
                    values[i][tif_i] = valueList.mean()
                elif Scope == '>0':
                    valueList = valueList[valueList > 0]
                    values[i][tif_i] = valueList.mean()
                else:
                    print('EsRaster Error！ Shp ID in ' + str(i))
                del dt
        else:
            for i in range(len(station_list)):
                # LonLat to ColRow
                [x, y] = CoordSys.lonlat2imagexy(dr, xValues[i], yValues[i])
                # if math.isnan(x):
                #     print('sdf')
                # 判断方框滤波
                if nirCellNum == 1:
                    dt = dr.ReadAsArray(int(x), int(y), 1, 1)
                elif nirCellNum % 2 == 0:
                    print('nirCellNum：The box image element is set incorrectly and must be an odd number')
                    break
                else:
                    dt = dr.ReadAsArray(int(x - (nirCellNum - 1) / 2), int(y - (nirCellNum - 1) / 2), nirCellNum,
                                        nirCellNum)
                if dt is None:
                    print()
                valueList = dt.flatten()
                # 判断取值范围
                if Scope == '':
                    values[i][tif_i] = valueList.mean()
                elif Scope == '>0':
                    valueList = valueList[valueList > 0]
                    values[i][tif_i] = valueList.mean()
                else:
                    print('EsRaster Error！ Shp ID in ' + str(i))
                del dt
        del dr
    colmns = [tif.split(os.sep)[-2] + r'_' + os.path.basename(tif).split(r'_')[0] for tif in tifList]
    values = pd.DataFrame(values, columns=colmns)
    print()
    return values
def attibute_table(shape_path):
    shp_ds= ogr.Open(shape_path)#打开矢量文件
    ''':type:osgeo.ogr.DataSource'''
    lyr = shp_ds.GetLayer(0)#获取图层
    ''':type:osgeo.ogr.Layer'''
    n = lyr.GetFeatureCount()  # 该图层中有多少个要素
    columnName = [feat.GetName() for feat in lyr.schema]#获取字段名称
    ptdf = []
    for ni in range(n):
        feat = lyr.GetFeature(ni)
        fid = [feat.GetField(ci) for ci in columnName]
        ptdf.append(fid)
    ptdf = pd.DataFrame(ptdf,columns=columnName)
    return ptdf


# data Path
datPath = r"G:\1_BeiJingUP\AUGB\Data\NPP\IntegResults\多年均值"
ptShp = r"G:\1_BeiJingUP\AUGB\Data\NPP\数据分析Table\Dataset_TNPP_QTP.shp"
startY = 2003
endY = 2017

# Model
tiflist = glob(datPath + os.sep + r"*_Mean.tif")
tifName = [os.path.basename(tifi).split(r'_')[0] for tifi in tiflist]
spdf = SampleRaster(tiflist, ptShp, 'PTID')
ptdf = attibute_table(ptShp)
spdf['TNPP'] = ptdf['TNPP']
spdf.to_excel(r'G:\1_BeiJingUP\AUGB\Data\NPP\数据分析Table\集成结果和TNPP.xlsx')
print()


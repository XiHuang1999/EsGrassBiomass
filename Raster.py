# -*- coding: utf-8 -*-

# @Time : 2022-06-29 14:50
# @Author : XiHuang O.Y.
# @Site : 
# @File : Raster.py
# @Software: PyCharm
from osgeo import gdal
import sys, os

EsGBPath = os.getcwd()
sys.path.append(EsGBPath)      # 添加函数文件位置
import CoordSys

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

def write_img(filename,im_proj,im_geotrans,im_data):
    """
    filename: 要保存的路径
    im_proj:投影
    im_geotrans:仿射矩阵
    im_data:图像数组
    return: None
    """
    if "int8" in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    if "int16" in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def SampleRaster(tifList, ptShp, siteName, nirCellNum=1, Scope=''):
    '''
    Using PtShp File to Sample Raster
    :param tifList: 栅格列表
    :param ptShp: 点shp文件
    :param siteName: 点字段名称
    :param nirCellNum: 方格大小，取像元附近均值（只能奇数）；默认值为1，为像元本身
    :param Scope: 筛选范围，eg：“>0” or ""
    :return:
    '''
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

    # 存放每个点的xy坐标的数组
    xValues = []
    yValues = []
    # 存放点“区站号”
    station_list = []

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

    # 创建二维空列表
    values = [[0 for col in range(len(tifList))] for row in range(len(station_list))]
    for tif, tif_i in zip(tifList, range(len(tifList))):
        # Execute ExtractByMask
        print(tif)
        # # 打开文件
        dr = gdal.Open(tif)
        # 存储着栅格数据集的地理坐标信息
        transform = dr.GetGeoTransform()
        # Sample
        for i in range(len(station_list)):
            # LonLat to ColRow
            [x,y] = CoordSys.lonlat2imagexy(dr,xValues[i],yValues[i])
            # 判断方框滤波
            if nirCellNum==1:
                dt = dr.ReadAsArray(int(x), int(y), 1, 1)
            elif nirCellNum%2==0:
                print('nirCellNum：The box image element is set incorrectly and must be an odd number')
                break
            else:
                dt = dr.ReadAsArray(int(x-(nirCellNum-1)/2), int(y-(nirCellNum-1)/2), nirCellNum, nirCellNum)
            valueList = dt.flatten()
            # 判断取值范围
            if Scope == '':
                values[i][tif_i * ptNum + i] = valueList.mean()
            elif Scope == '>0':
                valueList = valueList[valueList>0]
                values[i][tif_i * ptNum + i] = valueList.mean()

    return values


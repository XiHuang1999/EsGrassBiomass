# -*- coding: utf-8 -*-

# @Time : 2022-06-29 14:50
# @Author : XiHuang O.Y.
# @Site : 
# @File : EsRaster.py
# @Software: PyCharm
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pandas as pd
import sys, os, math

EsRasterPath = os.getcwd()
sys.path.append(EsRasterPath)      # 添加函数文件位置
import CoordSys
os.environ['PROJ_LIB'] = r'C:\Anaconda\Anaconda\Lib\site-packages\osgeo\data\proj'

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
    :param ptShp: str or DataFrame,点shp文件 or 第一、五列是站点代码、年份，第1列是Lon第2列是Lat的DataFrame
    :param siteName: 点字段名称
    :param nirCellNum: 方格大小，取像元附近均值（只能奇数）；默认值为1，为像元本身
    :param Scope: 筛选范围，eg：“>0” or ""
    :return:
    '''
    # 存放每个点的xy坐标的数组
    xValues = []
    yValues = []
    # 存放点“区站号”
    station_list = []
    ptNum = 0
    if isinstance(ptShp,str):
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
        print('Sample Raster :',tif)
        # # 打开文件
        dr = gdal.Open(tif)
        # 存储着栅格数据集的地理坐标信息
        transform = dr.GetGeoTransform()
        # Sample
        for i in range(len(station_list)):
            # LonLat to ColRow
            [x,y] = CoordSys.lonlat2imagexy(dr,xValues[i],yValues[i])
            # if math.isnan(x):
            #     print('sdf')
            # 判断方框滤波
            if nirCellNum==1:
                dt = dr.ReadAsArray(int(x), int(y), 1, 1)
            elif nirCellNum%2==0:
                print('nirCellNum：The box image element is set incorrectly and must be an odd number')
                break
            else:
                dt = dr.ReadAsArray(int(x-(nirCellNum-1)/2), int(y-(nirCellNum-1)/2), nirCellNum, nirCellNum)
            if dt is None:
                print()
            valueList = dt.flatten()
            # 判断取值范围
            if Scope == '':
                values[i][tif_i] = valueList.mean()
            elif Scope == '>0':
                valueList = valueList[valueList>0]
                values[i][tif_i] = valueList.mean()
            else:
                print('EsRaster Error！ Shp ID in '+str(i))
    return values

def RasterMean_8Days(DayFileList,beginDays,endDays):
    '''
    将气象数据的8天数据计算均值
    :param DayFileList: List,带路径的栅格数据文件名数组
    :param beginDays: int,第一个8天的天数
    :param endDays: int,最后的一个8天的天数
    :param countDays: int,一共有多少个8天
    :return: Dataframe,均值数组
    '''
    if len(DayFileList)<1:
        print(r'Raster List is empty！')
        return 0
    elif len(DayFileList)>=1:
        # First 8 day data
        proj,geotrans,Mean = read_img(DayFileList[0])
        Mean *= beginDays
        # Data in the middle of the range
        for i in range(1,len(DayFileList)-1):
            proj,geotrans,iMean = read_img(DayFileList[i])
            Mean += iMean*8
        # Last one of 8 day data
        proj, geotrans, iMean = read_img(DayFileList[-1])
        Mean = Mean + iMean * endDays
        # Mean of range time
        Mean = Mean / ((len(DayFileList)-2)*8+beginDays+endDays)
        return Mean,proj,geotrans

def RasterSum_8Days(DayFileList,beginDays,endDays):
    '''
    将气象数据的8天数据计算求和
    :param DayFilePath: List,带路径的栅格数据文件名数组
    :param beginDays: int,第一个8天的天数
    :param endDays: int,最后的一个8天的天数
    :return: Dataframe,求和数组
    '''
    if len(DayFileList)<1:
        print(r'Raster List is empty！')
        return 0
    elif len(DayFileList)>=1:
        # First 8 day data
        proj,geotrans,SumRaster = read_img(DayFileList[0])
        SumRaster /= 8
        SumRaster *= beginDays
        # Data in the middle of the range
        for i in range(1,len(DayFileList)-1):
            proj,geotrans,iSumRaster = read_img(DayFileList[i])
            SumRaster += iSumRaster
        # Last one of 8 day data
        proj, geotrans, iSumRaster = read_img(DayFileList[-1])
        iSumRaster /= 8
        SumRaster = SumRaster + iSumRaster * endDays
        # Sum of range time, that is SumRaster
        return SumRaster,proj,geotrans

def RasterMax(DayFileList):
    '''
    将气象数据的8天数据计算最大值
    :param DayFilePath: List,带路径的栅格数据文件名数组
    :return: Dataframe,栅格最大值
    '''
    if len(DayFileList)<1:
        print(r'Raster List is empty！')
        return 0
    elif len(DayFileList)>=1:
        proj,geotrans,MaxRaster = read_img(DayFileList[0])
        for i in range(1,len(DayFileList)):
            proj,geotrans,iMaxRaster = read_img(DayFileList[i])
            MaxRaster = iMaxRaster[MaxRaster<iMaxRaster]
        return MaxRaster

def RasterMin(DayFileList):
    '''
    将气象数据的8天数据计算最小值
    :param DayFilePath: List,带路径的栅格数据文件名数组
    :return: Dataframe,最小值数组
    '''
    if len(DayFileList) < 1:
        print(r'Raster List is empty！')
        return 0
    elif len(DayFileList) >= 1:
        proj, geotrans, MinRaster = read_img(DayFileList[0])
        for i in range(1, len(DayFileList)):
            proj, geotrans, iMinRaster = read_img(DayFileList[i])
            MinRaster = iMinRaster[MinRaster > iMinRaster]
        return MinRaster

# def RasterCalc(DayFileList, calcKey):
#     '''
#     栅格计算
#     :param DayFileList: List,带路径的栅格数据文件名数组
#     :param calcKey: str,计算类型（包括Mean,Sum,Max,Min）
#     :return:计算后的值
#     '''
#     calcKey = calcKey.capitalize()
#     if calcKey == 'Mean':
#         calcResult = RasterMean(DayFileList)
#     elif calcKey == 'Sum':
#         calcResult = RasterSum(DayFileList)
#     elif calcKey == 'Max':
#         calcResult = RasterMax(DayFileList)
#     elif calcKey == 'Min':
#         calcResult = RasterMin(DayFileList)
#     return calcResult
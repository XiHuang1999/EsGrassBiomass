# -*- coding: utf-8 -*-

# @Time : 2022-06-29 14:50
# @Author : XiHuang O.Y.
# @Site : 
# @File : EsRaster.py
# @Software: PyCharm

## Raster Process
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pandas as pd
import sys, os, math

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
    :return: pd.DataFrame
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
        print(os.path.basename(tif),end='/')
        # # 打开文件
        dr = gdal.Open(tif)
        # 存储着栅格数据集的地理坐标信息
        transform = dr.GetGeoTransform()
        # Sample
        if 'NDVI' in os.path.basename(tif).upper():
            for i in range(len(station_list)):
                # LonLat to ColRow
                [x,y] = CoordSys.geo2imagexy(dr,xValues[i],yValues[i])
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
                    print('EsRaster Error！ Cheack ReadAsArray at: '+tif)
                valueList = dt.flatten()
                # 判断取值范围
                if Scope == '':
                    values[i][tif_i] = valueList.mean()
                elif Scope == '>0':
                    valueList = valueList[valueList>0]
                    values[i][tif_i] = valueList.mean()
                else:
                    print('EsRaster Error！ Shp ID in '+str(i))
                del dt
        else:
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
                del dt
        del dr
    colmns = [tif.split(os.sep)[-2] + r'_' + os.path.basename(tif).split(r'_')[0] for tif in tifList]
    values = pd.DataFrame(values, columns=colmns,index=list(ptShp.index))
    print()
    return values

def SampleRaster_Para(tifList, ptShp, siteName, nirCellNum=1, Scope=''):
    '''
    Using PtShp File to Sample Raster by Paralleled
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
    colmns = [tif.split(os.sep)[-2] + r'_' + os.path.basename(tif).split(r'_')[0] for tif in tifList]
    values = pd.DataFrame(values, columns=colmns,index=list(ptShp.index))

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
        SumRaster[SumRaster<=0] = 0
        SumRaster /= 8
        SumRaster *= beginDays
        # Data in the middle of the range
        for i in range(1,len(DayFileList)-1):
            proj,geotrans,iSumRaster = read_img(DayFileList[i])
            iSumRaster[iSumRaster <= 0] = 0
            SumRaster += iSumRaster
        # Last one of 8 day data
        proj, geotrans, iSumRaster = read_img(DayFileList[-1])
        iSumRaster[iSumRaster <= 0] = 0
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

def RasterCalc(DayFileList, calcKey):
    '''
    栅格计算
    :param DayFileList: List,带路径的栅格数据文件名数组
    :param calcKey: str,计算类型（包括Mean,Sum,Max,Min）
    :return:计算后的值
    '''
    calcKey = calcKey.capitalize()
    if calcKey == 'Mean':
        calcResult = RasterMean(DayFileList)
    elif calcKey == 'Sum':
        calcResult = RasterSum(DayFileList)
    elif calcKey == 'Max':
        calcResult = RasterMax(DayFileList)
    elif calcKey == 'Min':
        calcResult = RasterMin(DayFileList)
    return calcResult

def ResampleLargeRaster(tif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',blockRow=10000,blockCol=10000,poolCount=20):
    '''
    重采样大型栅格数据
    :param tif: str,栅格
    :param blockRow: int,按块读取时每块行数
    :param blockCol: int,按块读取时每块列数
    :param poolCount: int,并行核心数
    :return: ndarray,并行返回数;object,属性参数;
    '''
    print('并行读取Tif：'+tif)
    # 数据属性读取
    scr = rasterio.open(tif)
    row, col = scr.shape[0], scr.shape[1]
    # 按窗口参数设置窗口列表
    # windows = scr.block_windows()
    # windows = [win for ji, win in scr.block_windows()] # 获取数据读取窗口
    windows = []
    for r in range(-(-row // blockRow)):
        for c in range(-(-col // blockCol)):
            if (r*blockRow+blockRow)<=row and (c*blockCol+blockCol)<=col:
                wins = Window(c*blockCol, r*blockRow, blockCol, blockRow)
            elif (r*blockRow+blockRow)>row and (c*blockCol+blockCol)<=col:
                wins = Window(c * blockCol, r * blockRow, blockCol, row-r*blockRow)
            elif (r*blockRow+blockRow)<=row and (c*blockCol+blockCol)>col:
                wins = Window(c * blockCol, r * blockRow, col-c*blockCol, blockRow)
            elif (r*blockRow+blockRow)>row and (c*blockCol+blockCol)>col:
                wins = Window(c * blockCol, r * blockRow, col-c*blockCol, row-r*blockRow)
            windows.append(wins)
    # 并行
    p = multiprocessing.Pool(processes=poolCount)
    t1 = time.time()
    # res = p.map(generate_mulcpu_vars, list(zip(windows, [tif] * len(windows))))
    paraRes = run_imap_mp(generate_mulcpu_vars, list(zip(windows, [tif] * len(windows))),poolCount)
    p.close()
    p.join()
    print('Time: %.2fs' % (time.time() - t1))
    del windows,p

    # 合并数据（方法一） 效率比方法二高
    # t1 = time.time()
    results = paraRes[0]
    for c in range(1, -(-col // blockCol)):
        results = np.append(results, paraRes[c], axis=1)
    for r in range(1,-(-row // blockRow)):
        colData = paraRes [r*(-(-row // blockRow))]
        for c in range(1,-(-col // blockCol)):
            colData = np.append(colData,paraRes[r*(-(-row // blockRow))+c],axis=1)
        results = np.append(results,colData,axis=0)
    # t2 = time.time()
    # print('Time: %.2fs' % (t2 - t1))

    # # 检查合并结果 出图
    # from matplotlib import pyplot
    # pyplot.imshow(results, cmap='pink') #r = np.append(np.append(paraRes[0], paraRes[1], axis=1), np.append(paraRes[2], paraRes[3], axis=1), axis=0)
    # pyplot.show()

    # # 合并数据（方法二） 效率比方法二高
    # t1 = time.time()
    # results = ''
    # for r in range(-(-row // blockRow)):
    #     colData = paraRes[r*(-(-row // blockRow))]
    #     for c in range(1,-(-col // blockCol)):
    #         colData = np.append(colData,paraRes[r*(-(-row // blockRow))+c],axis=1)
    #     if r == 0:
    #         results = colData
    #     else:
    #         results = np.append(results,colData,axis=0)
    # t2 = time.time()
    # print('Time: %.2fs' % (t2 - t1))
    del paraRes, colData

    # 更新元数据
    scr_Temp = rasterio.open(r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG\TAVG_2000001.tif')  # 模板数据属性读取
    transform, width, height = calculate_default_transform(
        scr.crs,  # 输入坐标系
        scr_Temp.crs,  # 输出坐标系
        results.shape[1],  # 输入图像宽
        results.shape[0],  # 输入图像高
        *scr.bounds)  # 输入数据源的图像范围
    # 更新数据集的元数据信息
    kwargs = scr.meta.copy()
    kwargs.update({
        'crs': scr_Temp.crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    return results,kwargs

def generate_mulcpu_vars(args):
    '''
    并行多参数调用
    :param args: 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    :return:
    '''
    return Parallel_BlockReadAndSample(args[0],args[1])


def Parallel_BlockReadAndSample(wins,tif):
    '''

    :param blk:
    :param windows:
    :return:
    '''
    scr = rasterio.open(tif)      # 数据属性读取
    data = scr.read(
            window=wins,
            out_shape=(1, int(wins.height/4), int(wins.width/4)), # out_shape=(2044, 2499),
            resampling=Resampling.average
        )[0]
    # data = scr.read(window=wins)[0]
    return data

def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    '''
    并行计算启动器,形象化并行计算并合理分配内存。
    :param func: function,函数
    :param argument_list: list,参数列表
    :param num_processes: int,进程数，不填默认为总核心3
    :param is_tqdm: bool,是否展示进度条，默认True
    :return: 并行返回值
    '''
    result_list_tqdm = []
    try:
        if num_processes == '':
            num_processes = multiprocessing.cpu_count()-3
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func,argument_list))
    return result_list_tqdm

def GdalReprojectImage(srcFile, outFile, refFile=r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG\TAVG_2000001.tif', resampleFactor=1, sampleMethod='average'):
    """
    栅格重采样,最近邻(gdal.gdalconst.GRA_NearestNeighbour)，采样方法自选。
    :param srcFile: str,待处理源文件
    :param outFile: str,输出文件
    :param refFile: str,参考投影的影像
    :param resampleFactor: float or int,重采样因子
    :param sampleMethod: str,采样方法
    :return: None
    """
    # 参数初始化
    smpM = {'near':gdal.gdalconst.GRA_NearestNeighbour,
            'bilinear': gdal.gdalconst.GRA_Bilinear,
            'cubic': gdal.gdalconst.GRA_Cubic,
            'cubicspline': gdal.gdalconst.GRA_CubicSpline,
            'lanczos': gdal.gdalconst.GRA_Lanczos,
            'average': gdal.gdalconst.GRA_Average,
            'mode':gdal.gdalconst.GRA_Mode}

    # 获取参考影像投影信息
    referencefile = gdal.Open(refFile, gdal.GA_ReadOnly)
    referencefileProj = referencefile.GetProjection()
    referencefileTrans = referencefile.GetGeoTransform()

    # 载入原始栅格
    dataset = gdal.Open(srcFile, gdal.GA_ReadOnly)
    srcProjection = dataset.GetProjection()
    srcGeoTransform = dataset.GetGeoTransform()
    srcWidth = dataset.RasterXSize
    srcHeight = dataset.RasterYSize
    srcBandCount = dataset.RasterCount
    srcNoDatas = [
        dataset.GetRasterBand(bandIndex).GetNoDataValue()
        for bandIndex in range(1, srcBandCount+1)
    ]
    srcBandDataType = dataset.GetRasterBand(1).DataType

    # 创建重采样后的栅格
    # srcFileName = os.path.basename(srcFile)   #获取数据属性
    # name = os.path.splitext(srcFileName)[0]   #获取文件名、后缀
    # outFileName = name + ".tif"
    # outFilePath = os.path.join(saveFolderPath, outFileName)
    driver = gdal.GetDriverByName('GTiff')
    outWidth = int(srcWidth * resampleFactor)
    outHeight = int(srcHeight * resampleFactor)
    outDataset = driver.Create(
        outFile,
        outWidth,
        outHeight,
        srcBandCount,
        srcBandDataType
    )
    geoTransforms = list(srcGeoTransform)
    geoTransforms[1] = geoTransforms[1]/resampleFactor
    geoTransforms[5] = geoTransforms[5]/resampleFactor
    outGeoTransform = tuple(geoTransforms)
    outDataset.SetGeoTransform(referencefileTrans)
    outDataset.SetProjection(referencefileProj)
    for bandIndex in range(1, srcBandCount+1):
        band = outDataset.GetRasterBand(bandIndex)
        band.SetNoDataValue(srcNoDatas[bandIndex-1])
    gdal.ReprojectImage(
        dataset,            # 输入数据集
        outDataset,         # 输出文件
        srcProjection,      # 参考投影
        srcProjection,      # 参考投影
        smpM[sampleMethod], # 重采样方法
        0.0, 0.0,           # 容差参数,容差参数,回调函数
    )
    return 0

def read_tifList(tifs):
    '''
    Concat Tif List
    :param tifs: list,Tiff列表
    :return: pd.Dataframe,一个tiff一列
    '''
    alldata = pd.DataFrame([])
    for tif in tifs:
        img_proj,img_geotrans,data = read_img(tif)
        data = data.reshape([-1,1])
        alldata = pd.concat([alldata,pd.DataFrame(data)],axis=1,join='outer')
        del data
    return alldata,img_proj,img_geotrans
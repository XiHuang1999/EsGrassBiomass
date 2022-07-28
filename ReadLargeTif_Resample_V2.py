# -*- coding: utf-8 -*-

# @Time : 2022-07-26 9:39
# @Author : XiHuang O.Y.
# @Site : 
# @File : ReadLargeTif_Resample_V2.py
# @Software: PyCharm
import rasterio
from glob import glob
import numpy as np
from scipy import stats
from rasterio.enums import Resampling
from rasterio.windows import Window
import time,multiprocessing,os
from multiprocessing import Manager
import rasterio
from ctypes import c_char_p
from tqdm import tqdm
from rasterio.warp import calculate_default_transform, reproject, Resampling

def ResampleRaster(tif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',outTif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\参数有错_NDVI.tif',blockRow=10000,blockCol=10000,poolCount=10):
    '''
    重采样栅格
    :param tif: str,栅格
    :param outTif: str,输出栅格
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
    print('Reading Time: %.2fs' % (time.time() - t1))
    # del windows,p

    # 合并数据（方法一） 效率比方法二高
    t1 = time.time()
    results = paraRes[0]
    for c in range(1, -(-col // blockCol)):
        results = np.append(results, paraRes[c], axis=1)
    for r in range(1,-(-row // blockRow)):
        colData = paraRes[r*(-(-col // blockCol))]
        for c in range(1,-(-col // blockCol)):
            colData = np.append(colData,paraRes[r*(-(-col // blockCol))+c],axis=1)
        results = np.append(results,colData,axis=0)
    t2 = time.time()
    print('Merge Time: %.2fs' % (t2 - t1))

    # # 检查合并结果 出图
    from matplotlib import pyplot
    pyplot.imshow(results, cmap='pink') #r = np.append(np.append(paraRes[0], paraRes[1], axis=1), np.append(paraRes[2], paraRes[3], axis=1), axis=0)
    pyplot.show()

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
    import numpy as np
    results = np.load(r"G:\1_BeiJingUP\AUGB\Data\20220629\Pycharm_result.npy", allow_pickle=True)
    windows = np.load(r"G:\1_BeiJingUP\AUGB\Data\20220629\Pycharm_windows.npy", allow_pickle=True)

    results = (results * 1.2 / 255 - 0.2)
    # results = list(map(lambda x : ((x * 12000 / 255 - 2000) / 10000), results))

    scr_Temp = rasterio.open(r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG\TAVG_2000001.tif')  # 模板数据属性读取
    transform, width, height = calculate_default_transform(
        scr.crs,  # 输入坐标系
        scr_Temp.crs,  # 输出坐标系
        results.shape[1], #int(scr.shape[1]/4),  # 输入图像宽
        results.shape[0], #int(scr.shape[0]/4),  # 输入图像高
        *scr.bounds)  # 输入数据源的图像范围
    # 更新数据集的元数据信息
    kwargs = scr.meta.copy()
    kwargs.update({
        'crs': scr_Temp.crs,
        'transform': transform,
        'width': width,
        'height': height,
        'dtype': 'float64'
    })

    # 写栅格
    # t1 = time.time()
    # outTif = r"G:\1_BeiJingUP\AUGB\Data\NDVI2015_3.tif"
    for bandi in range(1, scr.count + 1):
        with rasterio.open(outTif, 'w', **kwargs) as dst:
            # src_scr = scr.read(bandi)
            #des_scr = np.empty((height, width), dtype='float64')  # 初始化输出图像数据 str(results.dtype)
            des_scr = np.zeros(results.shape, np.float64)
            # reproject(
            #     # 源文件参数
            #     source=results,  # rasterio.band(scr2, i),
            #     src_crs=scr.crs,
            #     src_transform=scr.transform,
            #     # 目标文件参数
            #     destination=des_scr,  # rasterio.band(dst, i),des_scr
            #     dst_transform=transform,
            #     dst_crs=scr_Temp.crs,
            #     # dst_nodata=np.nan,
            #     resampling=Resampling.nearest  # 还有：Resampling.average
            #     # num_threads=2
            # )
            reproject(
                source=results,  # rasterio.band(scr2, i),
                destination=des_scr,  # rasterio.band(dst, i),des_scr
                src_transform=scr.transform,
                src_crs=scr.crs,
                dst_transform=transform,
                dst_crs=scr_Temp.crs
                # dst_nodata=np.nan,
                # resampling=Resampling.nearest  # 还有：Resampling.average
                # num_threads=2
            )
            for i,win in zip(range(len(windows)),windows):
                dst.write(results[windows[i].row_off:windows[i].row_off+windows[i].height,
                          windows[i].col_off:windows[i].col_off+windows[i].width],
                          bandi,
                          window=windows[i])
            # for i,win in zip(range(len(windows)),windows):
            #     dst.write(results[windows[i].row_off:windows[i].row_off+windows[i].height,
            #               windows[i].col_off:windows[i].col_off+windows[i].width],
            #               bandi,
            #               window=windows[i])
        dst.close()  # 写入后关闭文件
    # t2 = time.time()
    # print('Write Time: %.2fs' % (t2 - t1))
    # return results,kwargs

def generate_mulcpu_vars_GdalReprojectImage(args):
    '''
    并行多参数调用
    :param args: 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    :return:
    '''
    return GdalReprojectImage(args[0],args[1],args[2],args[3],args[4])


def Parallel_BlockRead(wins,tif):
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
    并行计算启动器
    :param func: function,函数
    :param argument_list: list,参数列表
    :param num_processes: int,进程数，不填默认为总核心3
    :param is_tqdm: bool,是否展示进度条，默认True
    :return: 并行返回值
    '''
    result_list_tqdm = []
    try:
        import multiprocessing
        if num_processes == '':
            num_processes = multiprocessing.cpu_count()-3
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            from tqdm import tqdm
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func,argument_list))
    return result_list_tqdm

import gdal,os

def GdalReprojectImage_Method1(srcFilePath, resampleFactor, saveFolderPath):
    """
    栅格重采样,最近邻(gdal.gdalconst.GRA_NearestNeighbour)，
    采样方法自选。
    :param srcFilePath:
    :param saveFolderPath:
    :return:
    """
    # 载入原始栅格
    dataset = gdal.Open(srcFilePath, gdal.GA_ReadOnly)
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
    srcFileName = os.path.basename(srcFilePath)
    name = os.path.splitext(srcFileName)[0]
    # 创建重采样后的栅格
    outFileName = name + ".tif"
    outFilePath = os.path.join(saveFolderPath, outFileName)
    driver = gdal.GetDriverByName('GTiff')
    outWidth = int(srcWidth * resampleFactor)
    outHeight = int(srcHeight * resampleFactor)
    outDataset = driver.Create(
        outFilePath,
        outWidth,
        outHeight,
        srcBandCount,
        srcBandDataType
    )
    geoTransforms = list(srcGeoTransform)
    geoTransforms[1] = geoTransforms[1]/resampleFactor
    geoTransforms[5] = geoTransforms[5]/resampleFactor
    outGeoTransform = tuple(geoTransforms)
    outDataset.SetGeoTransform(outGeoTransform)
    outDataset.SetProjection(srcProjection)
    for bandIndex in range(1, srcBandCount+1):
        band = outDataset.GetRasterBand(bandIndex)
        band.SetNoDataValue(srcNoDatas[bandIndex-1])
    gdal.ReprojectImage(
        dataset,
        outDataset,
        srcProjection,
        srcProjection,
        gdal.gdalconst.GRA_NearestNeighbour,
        0.0, 0.0,
    )
    return 0

def GdalReprojectImage(srcFile, outFile, refFile=r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG\TAVG_2000001.tif', resampleFactor=1, sampleMethod='near'):
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

    # # 获取参考影像投影信息
    # referencefile = gdal.Open(refFile, gdal.GA_ReadOnly)
    # referencefileProj = referencefile.GetProjection()
    # referencefileTrans = referencefile.GetGeoTransform()

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
    outDataset.SetGeoTransform(outGeoTransform)
    outDataset.SetProjection(srcProjection)
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

if __name__=="__main__":

    # GdalReprojectImage(r"G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2001.tif", 0.1,
    #                    r"G:\1_BeiJingUP\AUGB\Data")
    outPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI'
    tiflist = glob(r'K:\OuyangXiHuang\AGB\NDVI_董金玮\data\*NDVI*.tif')[0:2]
    # tiflist.sort()
    outtiflist = [outPath+os.sep+os.path.basename(tif).replace(r'max', r'_') for tif in tiflist]
    reftiflist = [r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG\TAVG_2000001.tif' for i in range(len(tiflist))]
    rsplist = [0.3 for i in range(len(tiflist))]
    mtdlist = ['near' for i in range(len(tiflist))]

    print('Start')
    kwgs = list(zip(tiflist,outtiflist,tiflist,rsplist,mtdlist))
    run_imap_mp(generate_mulcpu_vars_GdalReprojectImage, kwgs, num_processes=1, is_tqdm=True)
    print('Done')
    print()

    # tif = r'K:\OuyangXiHuang\AGB\NDVI_董金玮\data\NDVImax2015.tif'
    # tif1 = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
    # ResampleRaster(tif1,r'G:\1_BeiJingUP\AUGB\Data\temp.tif',2044,2499,2)
    # ResampleRaster(tif,r'G:\1_BeiJingUP\AUGB\Data\temp.tif',12000,12000,24)
    # inPath = r'K:\OuyangXiHuang\AGB\NDVI_董金玮\data'
    # outPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI'
    # inTif = glob(inPath+os.sep+r'NDVI*.tif')
    # inTif.sort()
    #
    # for tif in inTif:
    #     outtif = outPath + os.sep + tif.split(os.sep)[-1]
    #     ResampleRaster(tif,outtif,12000,12000,24)
    # print()


#
# ##裁剪测试
# from osgeo import gdal
#
# input_shape = r"G:\1_BeiJingUP\AUGB\Data\NDVImax2001_Temp.tif"
# input_shape = gdal.Open(input_shape)
# output_raster = r"G:\1_BeiJingUP\AUGB\Data\NDVImax2001_Wrap.tif"
# # tif输入路径，打开文件
# input_raster = r"G:\1_BeiJingUP\AUGB\Data\NDVImax2001.tif"
# # 矢量文件路径，打开矢量文件
# input_raster = gdal.Open(input_raster)
# # 开始裁剪，一行代码，爽的飞起
# ds = gdal.Warp(output_raster,
#                input_raster,
#                format='GTiff',
#                cutlineDSName=input_shape,
#                cutlineWhere="FIELD = 'whatever'",
#                dstNodata=0)
# ds = None




# # 投影测试
# from rasterio.windows import Window
# import numpy as np
# import rasterio
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# tif1 = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
# tif2 = r'G:\1_BeiJingUP\AUGB\Data\SWRS_2000001_NoProj.tif'
# tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif'
# scr1 = rasterio.open(tif1)      # 数据属性读取-模板
# scr2 = rasterio.open(tif2)      # 数据属性读取
# transform, width, height = calculate_default_transform(
#     scr2.crs,    # 输入坐标系
#     scr1.crs,    # 输出坐标系
#     scr2.width,  # 输入图像宽
#     scr2.height, # 输入图像高
#     *scr2.bounds)# 输入数据源的图像范围
# # 更新数据集的元数据信息
# kwargs = scr2.meta.copy()
# kwargs.update({
#     'crs': scr1.crs,
#     'transform': transform,
#     'width': width,
#     'height': height
# })
# with rasterio.open(r'G:\1_BeiJingUP\AUGB\Data\tempProj6.tif', 'w', **kwargs) as dst:
#     for i in range(1, scr2.count + 1):
#         src_scr = scr2.read(i)
#         des_scr = np.empty((height, width), dtype=kwargs['dtype'])  # 初始化输出图像数据
#         reproject(
#             # 源文件参数
#             source=src_scr,                 #rasterio.band(scr2, i),
#             src_crs=scr2.crs,
#             src_transform=scr2.transform,
#             # 目标文件参数
#             destination=des_scr,            #rasterio.band(dst, i),
#             dst_transform=transform,
#             dst_crs=scr1.crs,
#             # dst_nodata=np.nan,
#             resampling=Resampling.nearest   #还有：Resampling.average
#             #num_threads=2
#         )
#         dst.write(des_scr, i)


# ## 窗口写
# outTif = r"G:\1_BeiJingUP\AUGB\Data\NDVI2015_1.tif"
# with rasterio.open(outTif, 'w', **kwargs) as dst:
#     des_scr = np.zeros(results.shape, np.float64)
#     reproject(
#         # 源文件参数
#         source=results,  # rasterio.band(scr2, i),
#         src_crs=scr.crs,
#         src_transform=scr.transform,
#         # 目标文件参数
#         destination=des_scr,  # rasterio.band(dst, i),des_scr
#         dst_transform=transform,
#         dst_crs=scr_Temp.crs,
#         # dst_nodata=np.nan,
#         # resampling=Resampling.nearest  # 还有：Resampling.average
#         # num_threads=2
#     )
#     for win in windows:
#         dst.write(results, bandi)
#     dst.close()

# with rasterio.open('rasterio/tests/data/RGB.byte.tif') as src:
#     transform, width, height = calculate_default_transform(
#         src.crs, dst_crs, src.width, src.height, *src.bounds)
#     kwargs = src.meta.copy()
#     kwargs.update({
#         'crs': dst_crs,
#         'transform': transform,
#         'width': width,
#         'height': height
#     })
#     with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
#         for i in range(1, src.count + 1):
#             reproject(
#                 source=rasterio.band(src, i),
#                 destination=rasterio.band(dst, i),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.nearest)

##窗口读取
# windows1 = [win1 for ij1, win1 in scr1.block_windows()]
# row, col = scr.shape[0],scr.shape[1]
#
# wins = Window(0, 0, 1000, 1000)
# data1 = scr.read()[0]
# data2 = scr.read(1)
#
# import rasterio
# from rasterio.enums import Resampling
# dataa = scr.read(
#         window=wins,
#         out_shape=(1, int(250),int(250)), # out_shape=(4088, 4998),
#         resampling=Resampling.average
#     )[0]
#
# from matplotlib import pyplot
# pyplot.imshow(data2, cmap='pink')
# pyplot.show()
#
# from matplotlib import pyplot
# pyplot.imshow(data1, cmap='pink')
# pyplot.show()
# #
#
# win2 = windows[23*3]
# scr = rasterio.open(tif)      # 数据属性读取
# data = scr.read(
#         window=win2,
#         out_shape=(1, int(wins.height/4), int(wins.width/4)), # out_shape=(2044, 2499),
#         resampling=Resampling.average
#     )[0]
# pyplot.imshow(data, cmap='pink')
# pyplot.show()
#
# from matplotlib import pyplot
# r = np.append(np.append(paraRes[0],paraRes[1],axis=1),np.append(paraRes[2],paraRes[3],axis=1),axis=0)
# pyplot.imshow(r, cmap='pink')
# pyplot.show()

# import dill
# dill.dump_session('file_name.pkl')

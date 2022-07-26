# -*- coding: utf-8 -*-

# @Time : 2022-07-26 9:39
# @Author : XiHuang O.Y.
# @Site : 
# @File : ReadLargeTif_Resample_V2.py
# @Software: PyCharm
import rasterio
import glob
import numpy as np
from scipy import stats
from rasterio.enums import Resampling
from rasterio.windows import Window
import time,multiprocessing
from multiprocessing import Manager
import rasterio
from ctypes import c_char_p
from tqdm import tqdm
from rasterio.enums import Resampling


def ResampleRaster(tif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',blockRow=10000,blockCol=10000,poolCount=10):
    '''
    重采样栅格
    :param tif: str,栅格
    :param blockRow: int,按块读取时每块行数
    :param blockCol: int,按块读取时每块列数
    :param poolCount: int,并行核心数
    :return: 并行返回值
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
    t1 = time.time()
    results = paraRes[0]
    for c in range(1, -(-col // blockCol)):
        results = np.append(results, paraRes[c], axis=1)
    for r in range(1,-(-row // blockRow)):
        colData = paraRes [r*(-(-row // blockRow))]
        for c in range(1,-(-col // blockCol)):
            colData = np.append(colData,paraRes[r*(-(-row // blockRow))+c],axis=1)
        results = np.append(results,colData,axis=0)
    t2 = time.time()
    print('Time: %.2fs' % (t2 - t1))

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
    return results

def generate_mulcpu_vars(args):
    '''
    并行多参数调用
    :param args: 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    :return:
    '''
    return Parallel_BlockRead(args[0],args[1])


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

def Parallel_BlockWarp(wins)

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



if __name__=="__main__":
    tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif'
    tif1 = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
    # d = ResampleRaster(tif1,2044,2499,2)
    d = ResampleRaster(tif,10000,10000,20)
    print()

#
# from rasterio.windows import Window
# tif1 = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
# tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif'
# scr1 = rasterio.open(tif1)      # 数据属性读取
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

with rasterio.open('rasterio/tests/data/RGB.byte.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
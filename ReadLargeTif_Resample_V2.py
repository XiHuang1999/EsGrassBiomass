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
    res = run_imap_mp(generate_mulcpu_vars, list(zip(windows, [tif] * len(windows))),poolCount)
    p.close()
    p.join()
    print('Time: %.2fs' % (time.time() - t1))
    return res

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
            out_shape=(1, wins.height, wins.width), # out_shape=(2044, 2499),
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



if __name__=="__main__":
    tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif'
    d = ResampleRaster(tif,10000,10000,20)
    print()

#
# from rasterio.windows import Window
# tif1 = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
# tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif'
# scr = rasterio.open(tif1)      # 数据属性读取
# windows = [win for ij, win in scr.block_windows()]
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
#
# from matplotlib import pyplot
# pyplot.imshow(dataa, cmap='pink')
# pyplot.show()
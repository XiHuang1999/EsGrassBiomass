# -*- coding: utf-8 -*-

# @Time : 2022-07-25 19:32
# @Author : XiHuang O.Y.
# @Site : 
# @File : ReadLargeTif_Resample.py
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

# #
# tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
# upscale_factor = 0.5
# scr = rasterio.open(tif)
# print(scr.shape)
# with rasterio.open(tif) as dataset:
#     data = dataset.read(
#         out_shape=(dataset.count, int(dataset.height * upscale_factor), int(dataset.width * upscale_factor)),
#         resampling=Resampling.average
#     )
# print(data.shape)
#
# from rasterio.windows import Window
# import rasterio
# tif = r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS\SWRS_2000001.tif'
# with rasterio.open(tif) as src:
#     w = src.read(dataset.count, window=Window(0, 0, 512, 256))

# def ResampleRaster1(tif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',pCount=5):
#     '''
#     重采样栅格
#     :param tif: str,Tiff文件路径名称
#     :param cellsize: float,像素大小
#     :return:
#     '''
#     # manager = Manager()  # 多进程可以共享的命名空间
#     # partif = manager.Value(c_char_p, tif)  # 这里c_char_p是个类
#
#     scr= rasterio.open(tif)      # 数据属性读取
#     windows = scr.block_windows() # 获取数据读取窗口
#     p = multiprocessing.Pool(processes=pCount)
#     t1 = time.time()
#     res = p.map(Parallel_Block, tqdm(list(zip(windows, [scr] * len(plan_able)))) )
#
#     # for i in range(pCount):
#     #     parResults = p.apply_async(Parallel_Block, (windows, partif))
#     #     get_result(res,lst) # 不使用回调函数而是单独 get结果
#     p.close()
#     p.join()
#     print('Time: %.2fs' % (time.time() - t1))
#     return parResults.get()
#     # src.read(1, window=Window(0, 0, 512, 256))

def ResampleRaster(tif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',pCount=5):
    '''
    重采样栅格
    :param tif: str,Tiff文件路径名称
    :param cellsize: float,像素大小
    :return:
    '''
    scr = rasterio.open(tif)  # 数据属性读取
    # windows = scr.block_windows()
    windows = [win for ji, win in scr.block_windows()] # 获取数据读取窗口
    p = multiprocessing.Pool(processes=pCount)
    t1 = time.time()
    # res = p.map(generate_mulcpu_vars, list(zip(windows, [tif] * len(windows))))
    res = run_imap_mp(generate_mulcpu_vars, list(zip(windows, [tif] * len(windows))),20)
    p.close()
    p.join()
    print('Time: %.2fs' % (time.time() - t1))
    return res

# def ResampleRaster2(tif=r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',pCount=5):
#     '''
#     重采样栅格
#     :param tif: str,Tiff文件路径名称
#     :param cellsize: float,像素大小
#     :return:
#     '''
#     scr = rasterio.open(tif)  # 数据属性读取
#     # windows = scr.block_windows()
#     windows = [win for ji, win in scr.block_windows()] # 获取数据读取窗口
#     p = multiprocessing.Pool(processes=pCount)
#     t1 = time.time()
#     res = p.map(Parallel_Block, tqdm(list(zip(windows, [scr] * len(windows)))))
#     p.close()
#     p.join()
#     print('Time: %.2fs' % (time.time() - t1))
#     return res

# def get_result(result,resultL):
#     # 不使用回调函数而是单独 get结果
#     print('get_result')
#     [index,y]=result.get()
#     resultL[index]=y

def generate_mulcpu_vars(args):
    return Parallel_Block1(args[0],args[1])


def Parallel_Block(wins,scr):
    '''

    :param blk:
    :param windows:
    :return:
    '''
    # scr = rasterio.open(tif)      # 数据属性读取
    # wins,scr = args  # 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    data = scr.read(window=wins)[0]
    return data
def Parallel_Block(wins,tif):
    '''

    :param blk:
    :param windows:
    :return:
    '''
    scr = rasterio.open(tif)      # 数据属性读取
    # wins,scr = args  # 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    data = scr.read(window=wins)[0]
    return data

# def Parallel_Block3(wins,tif):
#     '''
#
#     :param blk:
#     :param windows:
#     :return:
#     '''
#     global scr
#     # scr = rasterio.open(tif)      # 数据属性读取
#     data = scr.read(window=wins)[0]
#     return data

def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    '''
    并行计算启动器
    :param func: function,函数
    :param argument_list: list,参数列表
    :param num_processes: int,进程数，不填默认为总核心3
    :param is_tqdm: bool,是否展示进度条，默认展示
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

    d = ResampleRaster(r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\NDVImax2015.tif',10)
    print()




















# #练习：共享string类型变量
# from multiprocessing import Process,Manager,Value
# from ctypes import c_char_p
#
# def greet(str):
#     str.value=str.value+",wangjing"
#
# if __name__=="__main__":
#     manager=Manager()  #多进程可以共享的命名空间
#     shareStr=manager.Value(c_char_p,"hello")   #这里c_char_p是个类
#     p=Process(target=greet,args=(shareStr,))
#     p.start()
#     p.join()
#     print(shareStr.value)


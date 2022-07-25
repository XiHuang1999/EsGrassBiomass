# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:05:17 2021

@author: lwq
"""

import rasterio
import glob
import numpy as np
from scipy import stats
import time
data_2 = np.arange(2000, 2016, 1)     #生成数组
start = time.time()                   #返回当前时间的时间戳
file1 = r'D:\test\python\趋势和显著性\MAT/'
File1 = r't2000.tif'
with rasterio.open(file1 + File1) as src:    # 获取数据读取窗口
    windows = [window for ij, window in src.block_windows()]   #此处一个栅格文件默认读取窗口为30个
    profile = src.profile       #源数据的元信息集合（使用字典结构存储了数据格式，数据类型，数据尺寸，投影定义，仿射变换参数等信息）
    nodata = src.nodata         #用于定义应屏蔽哪些像素
write_file = ['slope.tif', 'p_value.tif']    #输出文件
Data = []
for i in write_file:
    """
    路径结尾用 / 或者 \\ ，因为\有转义的含义，\' 会把 ' 转义从而与 ‘’ 不匹配
    W 写入模式
    **profile 输出数据的元信息集合
    """
    src1 = rasterio.open(r'D:\test\python\趋势和显著性\输出/' + i, 'w', **profile)
    Data.append(src1)


def fun2(data_1):
    """
    data_1: 需要计算趋势和显著性水平的数组，为三维数组
    return: 返回趋势r2score和显著性水平pvalue
    """
    J = data_1.shape[1]      #在三维数组中，shape[1]表示二维数组的行数
    K = data_1.shape[2]      #在三维数组中，shape[2]表示二维数组的列数
    """
    np.empty创建空的多维数组，
    np.empty((2, J, K)为三维数组，
    表示2个二维数组,每个二维数组有J个一维数组,一维数组长度为K
    dtype 指定输出数组的数值类型
    """
    k = np.empty((2, J, K), dtype='float32')
    #循环三维数组的每个通道的列表
    for x in range(0, J):
        for y in range(0, K):
            y_data = data_1[:, x, y]     # data_1[:, x, y] 三维数组中的所有二维数组中的 x行 y列
            if len(y_data[y_data == nodata]) > 0:   #y_data[y_data == nodata]，y_data为一维数组，此处提取y_data中值为nodata的部分
                k[:, x, y] = np.nan      #通道里的每个值都为nan
            else:
                x_data = data_2          #自变量为年份
                OLS = stats.linregress(x_data, y_data)   #线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
                slope = OLS[0]
                pvalue = OLS[3]
                k[1, x, y] = pvalue      # 显著性水平
                k[0, x, y] = slope       # 趋势
    return k


for win in windows:
    file = r'D:\test\python\趋势和显著性\MAT\*'
    dat = []
    for year in range(2000, 2016):
        file2 = glob.glob(file + str(year) + '*.tif')
        with rasterio.open(file2[0]) as src1:
            block_array = src1.read(window=win)[0]     #产生二维数组
        dat.append(block_array)             #复合列表，里面是多个二维数组
    data1 = np.array(dat)                   #把复合列表转化为三维数组
    data2 = fun2(data1)                     #引用 fun2 函数，输出趋势和显著性水平
    Data[0].write(data2[0], 1, window=win)  #栅格文件为只有一个波段的三维数组，所以写入的也必须是一个三维数组，此处指定波段即可
    Data[1].write(data2[1], 1, window=win)
Data[0].close()       #写入后关闭文件
Data[1].close()
end = time.time()
print(end - start)    #计算程序运行时间
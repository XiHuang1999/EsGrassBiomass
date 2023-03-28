#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time ： 2023/3/21 021 10:58:26
# @IDE ：PyCharm
# @Author : Xihuang Ouyang

import numpy as np
import pandas as pd
import os
import scipy.stats as st
##求P
exce=r"G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult2_一阶差分\2000_2018_三个IP+碳指标合并最终结果_更新了NDVI.xlsx"
a=pd.read_excel(exce)
cl=['Livestock','FBNPP','TAVG0','PRCP0','SWRS0','NDVI','CO2']
for ci in cl:
    r = []
    x = pd.Series(np.linspace(2000,2018,19), dtype = np.int32)
    if ci == '':
        coln = [i for i in range(2000,2019)]
    else:
        coln = [ci+r'_'+str(i) for i in range(0,19)]
    for i in range(a.shape[0]):
        y = pd.Series(a.loc[i,coln], dtype = np.float64)
        linreg = st.linregress(x, y)
        r.append(linreg.pvalue)
    if ci == '':
        ci='Livestock'
    a[ci+"_00_18_P"] = r
b = a.copy()
for ci in ['FBNPP']: #,'TAVG0','PRCP0','CO2','SWRS0','NDVI'
    if ci == '':
        ci='Livestock'
    b= b[b[ci+"_00_18_P"]<0.05]
    print(ci+"_00_18_P:")
    print(b.shape[0])
b.to_excel(r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult2_一阶差分\2000_2018_全部7个指标显著变化（NDVI不影响）.xlsx')



##求slope
startY = 2000
endY = 2018
exce=r"G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult\2000_2018_三个IP+碳指标合并最终结果1.xlsx"
a=pd.read_excel(exce)
cl=['','FBNPP','TAVG0','PRCP0','SWRS0','NDVI','CO2']
for ci in cl:
    r = []
    x = pd.Series(np.linspace(startY,endY,endY-startY+1), dtype = np.int32)
    if ci == '':
        coln = [i for i in range(startY,endY+1)]
    else:
        coln = [ci+r'_'+str(i) for i in range(startY-2000,endY-2000+1)]
    for i in range(a.shape[0]):
        y = pd.Series(a.loc[i,coln], dtype = np.float64)
        linreg = st.linregress(x, y)
        r.append(linreg.slope)
    if ci == '':
        ci='Livestock'
    a[ci+"_13_18_slp"] = r
b = a.copy()
for ci in ['','FBNPP','TAVG0','PRCP0','SWRS0','CO2']:
    if ci == '':
        ci='Livestock'
    b= b[b[ci+"_00_18_P"]<0.05]
    print(ci+"_00_18_P:")
    print(b.shape[0])
a.to_excel(r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult\2000_2018_fBNPP显著变化.xlsx')

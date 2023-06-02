#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time ： 2023/3/21 021 10:58:26
# @IDE ：PyCharm
# @Author : Xihuang Ouyang

import numpy as np
import pandas as pd
import os
import scipy.stats as st

##求slope
startY = 2000
endY = 2018
exce=r"F:\ScienceResearchProject\AUGB\SEM_2000_2018\全国年均值变化_加Livestock.xlsx"
a=pd.read_excel(exce)
dt = a.copy()
hd = a.iloc[:,[32,34,36,37,2]]
cl=[r'Livestock',r'FBNP',r'TAVG',r'PRCP',r'SWRS',r'NDVI',r'CO2']

s1 = pd.DataFrame()
for i in range(0,endY-startY):
    s2 = []
    for ci in cl:
        sub = list(a.loc[:,ci+r'_'+str(i+1)] - a.loc[:,ci+r'_'+str(i)])
        s2.append(sub)
    if i == 0:
        s2 = pd.DataFrame(s2).T
        s1 = pd.concat([hd,s2],axis=1)
    else:
        s2 = pd.DataFrame(s2).T
        s2 = pd.concat([hd,s2],axis=1)
        s1 = s1.append(s2) #pd.concat([s1,pd.DataFrame(s2).T],axis=0)
s1.columns = list(s1.columns[:5])+cl
print()
s1 = s1.dropna(subset=cl)
s1.to_excel(r'F:\ScienceResearchProject\AUGB\SEM_2000_2018\2000_2018_CHN_New_一阶差分结果.xlsx')


#####

startY = 2000
endY = 2018
exce=r"G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult2_一阶差分\ZonalCHN\全国年均值变化_加Livestock.xlsx"
a=pd.read_excel(exce)
dt = a.copy()
hd = a.iloc[:,[0,2,4]]
cl=['Livestock','fBNPP','TAVG','PRCP','SWRS','NDVI','CO2']
s = pd.DataFrame()
for j in range(0,endY-startY):
    s1 = pd.DataFrame()
    for i in range(j,endY-startY):
        print(i+1,end='-')
        print(j)
        s2 = []
        for ci in cl:
            print(ci + r'_' + str(i + 1) + r'  -  ' + ci + r'_' + str(j))
            sub = list(a.loc[:,ci+r'_'+str(i+1)] - a.loc[:,ci+r'_'+str(j)])
            s2.append(sub)
        # s2.append(str(i + 1) + r' - ' + str(j))
        if i == 0:
            s2 = pd.DataFrame(s2).T
            s1 = pd.concat([hd,s2],axis=1)
        else:
            s2 = pd.DataFrame(s2).T
            s2 = pd.concat([hd,s2],axis=1)
            s1 = s1.append(s2) #pd.concat([s1,pd.DataFrame(s2).T],axis=0)
    s = s.append(s1)  # pd.concat([s1,pd.DataFrame(s2).T],axis=0)
s.columns = list(s.columns[:3])+cl
# s.columns = list(s.columns[:5])+cl
print()
s[s.iloc[:,6]==0] = np.nan
s = s.dropna(subset=cl)
s.to_excel(r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult2_一阶差分\更新了NDVI的Excel\2000_2018_除了SWRS都变化显著_一阶所有差分结果.xlsx')





#### 不管显著不显著，全做差分

##求slope
startY = 2000
endY = 2018
exce=r"G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult2_一阶差分\2000_2018_三个IP+碳指标合并最终结果_更新了NDVI.xlsx"
a=pd.read_excel(exce)
dt = a.copy()
hd = a.iloc[:,[32,34,36,37,2]]
cl=[r'Livestock',r'FBNPP',r'TAVG0',r'PRCP0',r'SWRS0',r'NDVI',r'CO2']

s1 = pd.DataFrame()
for i in range(0,endY-startY):
    s2 = []
    for ci in cl:
        print(ci+r'_'+str(i+1)+r'  -  '+ci+r'_'+str(i))
        sub = list(a.loc[:,ci+r'_'+str(i+1)] - a.loc[:,ci+r'_'+str(i)])
        s2.append(sub)
    if i == 0:
        s2 = pd.DataFrame(s2).T
        s1 = pd.concat([hd,s2],axis=1)
    else:
        s2 = pd.DataFrame(s2).T
        s2 = pd.concat([hd,s2],axis=1)
        s1 = s1.append(s2) #pd.concat([s1,pd.DataFrame(s2).T],axis=0)
s1.columns = list(s1.columns[:5])+cl
print()
s1 = s1.dropna(subset=cl)
s1.to_excel(r'G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\ZonalResult2_一阶差分\更新了NDVI后_不管县显著不显著全部做\2000_2018_不管是否显著变化_一阶差分结果.xlsx')











##求slope
startY = 2000
endY = 2018
path = r"G:\1_BeiJingUP\AUGB\Data\SEM_2000_2018\全区域最新"
name = r'全区草地_Mean_汇总'
exce = path + os.sep + name +r'.csv'
a = pd.read_csv(exce,header=0,index_col=0)
dt = a.copy()
cl=list(a.index)

s = pd.DataFrame()
for j in range(0,endY-startY):
    s1 = pd.DataFrame()
    for i in range(j,endY-startY):
        print(i+1,end='-')
        print(j)
        s2 = list(a.iloc[:,(i+1)] - a.iloc[:,(j)])
        # s2.append(str(i + 1) + r' - ' + str(j))
        if i == 0:
            s2 = pd.DataFrame(s2).T
            s1 = s2
        else:
            s2 = pd.DataFrame(s2).T
            s1 = s1.append(s2) #pd.concat([s1,pd.DataFrame(s2).T],axis=0)
    s = s.append(s1)  # pd.concat([s1,pd.DataFrame(s2).T],axis=0)
print()
# s[s.iloc[:,6]==0] = np.nan
# s = s.dropna(subset=cl)
s.columns = cl
s.to_csv(path + os.sep + name + r'_一阶差分.csv')





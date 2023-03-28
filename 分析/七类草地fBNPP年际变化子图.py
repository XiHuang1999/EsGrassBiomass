#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time ： 2023/3/12 012 09:25:38
# @IDE ：PyCharm
# @Author : Xihuang Ouyang

import pandas as pd
import numpy as np
import os, sys, time, math, random
from glob import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.stats as st
from datetime import datetime
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
# matplotlib.rcParams['backend'] = 'SVG'
def average(data):
    return sum(data) / len(data)


###=================================='''图样式三'''================================================

def startP(p_):
    if linreg.pvalue<=0.001:
        pstart = '$^{***}$'
    elif linreg.pvalue<=0.01:
        pstart = '$^{**}$'
    elif linreg.pvalue <= 0.05:
        pstart = '$^{*}$'
    else:
        pstart = ''
    print(linreg)
    return pstart

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'black'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rc('font',family='Times New Roman')

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=8)
# xlsx = r"G:\1_BeiJingUP\AUGB\Table\时间序列验证\NDVI\Mean.xlsx"
xlsx = r"F:\ScienceResearchProject\AUGB\Table\七个草地\Mean.xlsx"

dataStc = pd.read_excel(xlsx,sheet_name='fBNPP',index_col=1)
dataStc = dataStc.iloc[0:8,1:1+19]
fig = plt.figure(figsize=(18/2.54,15/2.54))  # 创建Figure对象
yrs = np.arange(2000,2018+1,1)


var = list(dataStc.index)
for i in range(0,len(var)):
    figi = 430+i+1
    ax = plt.subplot(figi)

    linreg = st.linregress(pd.Series(yrs, dtype=np.float64), pd.Series(dataStc.iloc[i,:], dtype=np.float64))
    pltx = [x for x in np.linspace(start=min(yrs), stop=max(yrs), num=100)]
    plty = [linreg.slope * x + linreg.intercept for x in pltx]
    ax.plot(yrs, dataStc.iloc[i,:], 'o-')
    ax.plot(pltx, plty, '--', color='black', alpha=0.8, linewidth=1, label=r'Slope = ' + "%.2f" % linreg.slope  + "g/m\u00b2/yr \n P = " + "%.2f" % linreg.pvalue)    # color='#4169E1'
    ax.set_ylabel(var[i],labelpad=0,fontproperties=font) #X轴标签
    ax.set_xlabel("year",labelpad=0) #X轴标签

    ax.text(2000,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*7.8/10,
            r'('+chr(97+i)+')  Slope = ' + "%.2f" % linreg.slope + startP(linreg.pvalue)+'\n'+
            '       R$^{2}$ = %.2f' % (linreg.rvalue**2)
            ,linespacing=1.1) #" g/m\u00b2/yr"

plt.subplots_adjust(left=0.06, right=0.99, bottom=0.09, top=0.97, wspace=0.25, hspace=0.3)
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.subplots_adjust(left = 0, wspace=0.3, hspace=0.3)
# plt.tight_layout()
# endregion
# plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(r'F:\ScienceResearchProject\AUGB\Pic'+os.sep+'七个fBNPP年际动态_'+(datetime.now().strftime("%H-%M-%S"))+'.png'
            ,dpi=1000,bbox_inches='tight',format='png')#, transparent=True
plt.show()
print()
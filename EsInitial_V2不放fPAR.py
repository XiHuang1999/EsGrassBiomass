# -*- coding: utf-8 -*-

# @Time : 2022-06-29 15:41
# @Author : XiHuang O.Y.
# @Site : 
# @File : EsInitial.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os, sys, time
from glob import glob
import openpyxl,os

from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig,RF_Algorithm_V2_RF5_NofPAR
#SVM_Algorithm,Cubist_Algorithm


if __name__=="__main__":
    '''读取参数文件，并初始化参数'''
    if os.path.exists(EsInitialPath+os.sep+"Parameters.ini"):
        inoutParas,excelParas,staticParasD,dynamicParasD,dymDayselect = readConfig.get_stcAnddym()
        outPath = inoutParas['out'] #输出路径
        exlFile = excelParas['exlfile'] #excel文件
        staticKey = list(staticParasD.keys()) #静态数据关键字
        staticPath = list(staticParasD.values()) #静态数据文件路径
        # dynamicKey = list(dynamicParasD.keys())  #动态数据关键字数
        # dynamicPath = list(dynamicParasD.values())   #动态数据文件路径
        exec('dynamicPreduceKey='+dymDayselect['8daysdataname'])    #动态数据中8天数据
        exec('dynamicPreduceDays='+dymDayselect['daysscope'])       #动态数据中选择范围
    else:
        # ==== Static Paramerters====
        # OutFile Path
        outPath = r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'

        # SiteExcel
        siteFile = r'G:\1_BeiJingUP\AUGB\Table\ALL_SITES2010.csv'
        exlFile = r"G:\1_BeiJingUP\AUGB\Table\ALL_SITES_Select(gDMm-2).xlsx"

        # Topography
        dem = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif'
        lat = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\LAT_China1km.tif'
        slp = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\slope_china1km.tif'
        asp = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\aspect_china1km.tif'
        # Vegetation
        cgc = r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\cgrass_China1km.tif'
        staticPath = [dem,lat,slp,asp,cgc]
        staticKey = [f.split(os.sep)[-1].split(r'.')[0] for f in staticPath]

        # ==== Active Paramerters=====
        dynamicPath = [r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\PRCP',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS']

        dynamicKey = [f.split(os.sep)[-1] for f in dynamicPath]
        activeData = [] #dict(zip(activeKey,activePath))

    '''Pre-Process of Dynamic raster'''
    # 过程文件夹创建
    for vari in range(len(dynamicPreduceKey)):
        for seasoni in range(int(len(dynamicPreduceDays)/2)+1):
            if not os.path.exists(outPath + os.sep + dynamicPreduceKey[vari].upper() + str(seasoni)):
                os.makedirs(outPath + os.sep + dynamicPreduceKey[vari].upper() + str(seasoni))

    # 预处理数据 产生过程文件
    # for yr in range(2000,2021):         # 按年遍历
    #     print(yr,end=' ==>> ')
    #     for vari in dynamicPreduceKey:  # 每年的每个变量
    #         # 列举tif文件
    #         filelist = glob(dynamicParasD[vari] + os.sep + r'*' + vari.upper()+ r'*_' + str(yr) + r'*.tif')
    #         filelist.sort()
    #
    #         # 计算8天数据的季节值
    #         for seasoni in range(int(len(dynamicPreduceDays)/2)):
    #             # 输出判断
    #             print(str(seasoni), end=' ')
    #             outRasterFileName = outPath + os.sep + vari.upper() + str(seasoni+1) + os.sep + vari.upper()+r'_'+ str(yr) +r'.tif'
    #             if os.path.exists(outRasterFileName):
    #                 continue
    #             # 转换闰年天数
    #             if yr % 4 != 0:
    #                 startDay = dynamicPreduceDays[seasoni * 2]
    #                 endDay = dynamicPreduceDays[seasoni * 2 + 1]
    #             else:
    #                 startDay = dynamicPreduceDays[seasoni * 2] +1
    #                 endDay = dynamicPreduceDays[seasoni * 2 + 1] +1
    #
    #             # 将第i天转换为第j个8天 [upper integer]
    #             start8Day = -(-startDay // 8)    # 第16个8天开始 此处的16开始为1 而非0开始
    #             end8Day = -(-endDay // 8)  # 第31个8天结束
    #             bdays = 8 - startDay % 8 + 1  # 在要处理的8天数据中，第一个8天的天数
    #             edays = endDay % 8  # 在要处理的8天数据中，最后一个8天的天数
    #
    #             # 根据时间范围处理 [8天数据处理]
    #             eightDaysFilelist = filelist[start8Day-1:end8Day]
    #             if vari.upper() == r'TAVG':
    #                 rasterR, proj, geotrans = EsRaster.RasterMean_8Days(eightDaysFilelist,bdays,edays)
    #             else:
    #                 rasterR, proj, geotrans = EsRaster.RasterSum_8Days(eightDaysFilelist, bdays, edays)
    #
    #             # 输出过程栅格
    #             EsRaster.write_img(outRasterFileName, proj, geotrans, rasterR)
    #             # del rasterR
    #             # # 下一次循环变量赋值
    #             # start8Day = dynamicPreduceDays[seasoni+1 * 2] // 8   # 第16个8天开始 此处的16开始为1
    #             # end8Day = dynamicPreduceDays[seasoni+1 * 2 + 1] // 8  # 第30个8天结束
    #
    #         # 计算8天数据的年值
    #         print(vari,end=' ')
    #         outRasterFileName = outPath + os.sep + vari.upper() + r'0' + os.sep + \
    #                             vari.upper() + r'_' + str(yr) + r'.tif'
    #         if os.path.exists(outRasterFileName):
    #             continue
    #         # 数据处理
    #         bdays = 8
    #         if yr % 4 != 0:
    #             bdays = 5
    #         else:
    #             bdays = 6
    #         if vari.upper() == r'TAVG':
    #             rasterR, proj, geotrans = EsRaster.RasterMean_8Days(filelist, bdays, edays)
    #         else:
    #             rasterR, proj, geotrans = EsRaster.RasterSum_8Days(filelist, bdays, edays)
    #         # 输出过程栅格
    #         EsRaster.write_img(outRasterFileName, proj, geotrans, rasterR)
    #     print('')

    # 并更新数据路径
    try:
        for vari in dynamicPreduceKey:
                dynamicParasD.pop(vari.lower())
                for seasoni in range(int(len(dynamicPreduceDays)/2)+1):
                    dynamicParasD.update({vari+str(seasoni): outPath + os.sep + vari.upper() + str(seasoni)})
    except Exception as E_results:
        print('预处理有异常：', E_results)
        print('\n\n\n我不管，预处理错误,你看着办吧!\nPreProcessing File Direction is ERROR！Please Check it！')
        time.sleep(100000)
    finally:  # finally的代码是肯定执行的，不管是否有异常,但是finally语块是可选的。
        print('',end='')


    # # '''读取站点文件'''
    # # G:\1_BeiJingUP\CommonData\temp\Export_Output_2.shp
    # # sitedf = pd.read_csv(exlFile, header=0)
    # # Load CSV files and Delete unstable Data
    # wb = openpyxl.load_workbook(exlFile)
    # sheetList = wb.sheetnames
    # allcsv = pd.DataFrame([])
    # for sheet in sheetList[:-2]:
    #     print(sheet)
    #     csv = pd.read_excel(exlFile, sheet_name=sheet)
    #     # print(type(csv.LON[0]))
    #     csv[r'Year'] = [int(sheet) for y in range(csv.shape[0])]
    #     allcsv = pd.concat([allcsv, csv])
    #     if not pd.api.types.is_numeric_dtype(csv['LON']):
    #         print(sheet,end='有错')
    #
    # allcsv1 = allcsv
    # # allcsv = allcsv[allcsv.AGB < allcsv.AGB.std() * 3 + allcsv.AGB.mean()]
    # # allcsv = allcsv[allcsv.AGB <= 600]
    # # allcsv = allcsv[allcsv.AGB >= 2]
    # allcsv = allcsv[(allcsv['LON'] >= 73) & (allcsv['LON'] <= 135)]
    # allcsv = allcsv[(allcsv['LAT'] > 0) & (allcsv['LAT'] <= 53)]
    # # allcsv.to_excel(outPath + os.sep + r'Table\All_yrs_Sites.xlsx', index=False)
    # # allcsv.to_csv(outPath + os.sep + r'Table\All_yrs_Sites1.csv', index=False)
    # print()
    #
    # '''## Sample ##'''
    # print(r'Sample : ',end=r'')
    # allyr = pd.DataFrame([])
    # for yr in range(2000,2021):
    #     yrcsv = allcsv[allcsv['Year']==yr]
    #     print(yr)
    #     if yrcsv.shape[0]==0:
    #         print(str(yr)+r'无数据跳过')
    #         continue
    #     '''Static raster Sample'''
    #     stcDf = EsRaster.SampleRaster_gdal3(staticPath, yrcsv, 'ID')
    #     yrcsv = pd.concat([yrcsv,stcDf],axis=1,join='outer')
    #
    #     '''Dynamic raster sample'''
    #     dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(dynamicParasD.keys()),list(dynamicParasD.values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    #     dymtif = sum(dymtif, [])    # 解决List嵌套
    #     dymDf = EsRaster.SampleRaster_gdal3(dymtif,yrcsv,'ID')
    #     yrcsv = pd.concat([yrcsv, dymDf], axis=1, join='outer')
    #
    #     '''Merge'''
    #     yrcsv = yrcsv[(yrcsv[r'FPAR0_FPAR'] > 0) | (yrcsv[r'FPAR1_FPAR'] > 0) | (yrcsv[r'FPAR2_FPAR'] > 0)]
    #     yrcsv = yrcsv[yrcsv[r'Soil_Clay.tif'] > -9999]
    #     allyr = pd.concat([allyr, yrcsv], axis=0, join='outer')
    #
    # # Change allyr
    # Ycols = ['ID', 'LON', 'LAT', 'AGB', 'Year', 'DEM', 'Parameters_LAT', 'Slope', 'Aspect', 'CGrass', 'Soil_Clay',
    #          'Soil_CoarseSand', 'Soil_FineSand','Soil_OrganicMass', 'Soil_PH_h2o', 'Soil_PowderedSand', 'NDVI',
    #          'Yearly_TAVG', 'Season1_TAVG', 'Season2_TAVG',
    #          'Yearly_PRCP', 'Season1_PRCP', 'Season2_PRCP',
    #          'Yearly_SWRS', 'Season1_SWRS', 'Season2_SWRS',
    #          'Yearly_FPAR', 'Season1_FPAR', 'Season2_FPAR']
    #         ['AGB', 'ID', 'LON', 'Parameters','Slope','Aspect','Soil_PH_h2o','Year','CGrass','Field1']]
    # allyr.columns = Ycols
    # allyr['NDVI'] = allyr['NDVI']*1.2/255 - 0.2
    # allyr.to_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr_V2.csv")
    dynamicParasD['ndvi'] = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\AfterPreProcess2'

    '''Algth Prepare'''
    # allyr = pd.read_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr.csv", index_col=0)
    # allyr = pd.read_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr_NDVI0-1.csv",index_col=0)
    allyr = pd.read_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr_V2_Deleted.csv",index_col=0)
    allyr = allyr[(allyr[r'Soil_Clay'] > -9999) & (allyr[r'CGrass'] < 100) & (allyr[r'AGB'] <= 1240)] #& (allyr[r'CGrass'] > 0)
    allyr = allyr.iloc[:,1:]


    print(allyr.columns)
    # region zscore标准化
    # from sklearn import preprocessing
    # zscore = preprocessing.StandardScaler()
    # zscore = zscore.fit_transform(allyr)
    # allyr = pd.DataFrame(zscore, index=allyr.index, columns=allyr.columns)
    # endregion


    Ycols = [i for i in allyr.columns if
             i not in ['AGB', 'ID', 'LON', 'Parameters','Slope','Aspect','Soil_PH_h2o','Year','CGrass','Field1',
                       'Yearly_FPA','Season1_FP','Season2_FP']]#,'Soil_Clay.tif', 'Soil_CoarseSand.tif', 'Soil_FineSand.tif','Soil_OrganicMass.tif', 'Soil_PH', 'Soil_PowderedSand.tif', 'Parameters_cgrass']]  # 'Year', or Ycols = allyr[allyr.columns.difference(['A', 'B'])]

    ## Vari describe
    # # region Description-Heatmap
    # allvar = allyr[['AGB'] + Ycols]
    # corr_mat = allvar.corr()
    # f, ax = plt.subplots(figsize=(16, 12))
    # mask = np.zeros_like(corr_mat)
    # for i in range(1, len(mask)):
    #     for j in range(0, i):
    #         mask[j][i] = True
    # sns.heatmap(corr_mat, cmap='PiYG', annot=True, mask=mask, linewidths=.05, square=True, annot_kws={'size': 6.5, 'weight':'bold'}, fmt=".2f")
    # # print(corr_mat)
    # plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    # plt.savefig(outPath+os.sep+r'PIC'+os.sep+'Corr_Heatmap_4.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.show()
    # # endregion

    # # region draw_distribution_histogram
    # plt.style.use('ggplot')     # 设置绘图风格
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 处理中文乱码
    # plt.rcParams['axes.unicode_minus'] = False # 坐标轴负号的处理
    # # 绘制直方图
    # # 绘制直方图
    # plt.hist(x=allvar.AGB,  # 指定绘图数据
    #          bins=42,  # 指定直方图中条块的个数
    #          color='steelblue',  # 指定直方图的填充色
    #          edgecolor='black'  # 指定直方图的边框色
    #          )
    # # allvar.AGB.plot(kind='hist', bins=60, color='steelblue', edgecolor='black', density=True, label='Frequency Histogram')
    # # # 绘制核密度图
    # # allvar.AGB.plot(kind='kde', color='red', label='Density')
    # plt.xlim(0, max(allvar.AGB))            # 轴范围
    # # plt.ylim(y1, y2)
    # plt.xlabel('AGB Value')     # 添加x轴和y轴标签
    # plt.ylabel('Frequency')
    # plt.title('AGB频数分布')      # 添加标题
    # # plt.legend()                # 显示图例
    # plt.savefig(outPath + os.sep + r'PIC' + os.sep + 'Distribution_hist_4.png', dpi=500, bbox_inches='tight')  # , transparent=True
    # plt.show()
    # print('')
    # # # # endregion

    # # region RF
    allyr = allyr.reset_index()
    deleteIndex = [91, 1451, 2066, 2247, 2962, 3018, 3032, 3458, 3459]+\
                  [71,  235, 3018, 3458, 3459]+\
                  [90, 1446, 2460, 3469, 4017, 4199, 4201, 4206]+\
                  [90, 1446, 3014, 3015, 3022, 3121, 3437, 3438]+\
                  [234, 2056, 2057, 2058, 2448, 2449, 2450]
    deleteIndex = list(dict.fromkeys(deleteIndex))
    allyr = allyr.drop(deleteIndex)

    allyr = allyr.reset_index(drop=True)
    # allyr = allyr.reset_index()
    deleteIndex = [1441, 3002, 3003, 3004, 3009, 3106, 3163, 3421, 3424]+\
                  [   3, 1445, 3005, 3449]+\
                  [1850, 2240]
    deleteIndex = list(dict.fromkeys(deleteIndex))
    allyr = allyr.drop(deleteIndex)


    # allyr = allyr.reset_index(drop=True)
    # deleteIndex = [87,  104,  128,  376, 1754, 1837, 2011, 2027, 2828, 2988, 3097]+\
    #               [16, 1287, 1474, 1837, 2536, 2584, 2828, 2944, 2948, 3172, 3241, 3293, 3407, 3433, 3450, 3839, 3901]
    # deleteIndex = list(dict.fromkeys(deleteIndex))
    # allyr = allyr.drop(deleteIndex)


    algY = allyr['AGB']
    algX = allyr[Ycols]
    print(Ycols)
    # RF_Algorithm_V2_RF5_NofPAR.RFEstimate(algX, algY,
    #                                [staticPath, dynamicParasD],
    #                                0.90)
    RF_Algorithm_V2_RF5_NofPAR.RFEstimate(algX, algY,
                                   [staticPath, dict([(key,dynamicParasD[key]) for key in list(dynamicParasD.keys())[0:-3]])],
                                   0.90)
    # # for i in range(1,24):
    # #     para = {'n_estimators': 566, 'max_features': i, 'bootstrap': True}
    # #     RF_Algorithm.RFEstimate(algX,algY,[staticPath,dynamicParasD],0.9,para,)
    # # endregion

    # region SVM
    # # region 归一化处理
    # stdAllyr = pd.DataFrame(allyr)
    # stdAllyr = stdAllyr.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # algY = allyr['AGB'].values
    # algX = stdAllyr[Ycols].values
    # # endregion
    # # region zscore标准化
    # # from sklearn import preprocessing
    # # zscore = preprocessing.StandardScaler()
    # # zscore = zscore.fit_transform(allyr)
    # # allyr = pd.DataFrame(zscore, index=allyr.index, columns=allyr.columns)

    # SVM_Algorithm.SVMEstimate(algX, algY, [staticPath, dynamicParasD], 0.9)
    # print()
    # endregion


    # # Cubist
    # region zscore标准化
    # from sklearn import preprocessing
    # zscore = preprocessing.StandardScaler()
    # zscore = zscore.fit_transform(allyr)
    # allyr = pd.DataFrame(zscore, index=allyr.index, columns=allyr.columns)
    # endregion

    # Cubist_Algorithm.CBEstimate(allyr[Ycols], allyr['AGB'], [staticPath, dynamicParasD], 0.925)

    print('')
    # for sheet in sheetList[:-1]:







    # for sheet in sheetList[:-1]:
    #     print(sheet)
    #     csv = pd.read_excel(exlFile, sheet_name=sheet)
    #     csv[r'Year'] = [int(sheet) for y in range(csv.shape[0])]
    #     allcsv = pd.concat([allcsv, csv])
    #     csv.to_csv(exlFile.replace(r'ALL_SITES_select.xlsx', r'ALL_SITES_' + str(sheet) + r'.csv'))
    # allcsv.to_csv(r'G:\1_BeiJingUP\AUGB\Table\06-21yrs_Sites.csv')
    # allcsv.to_excel(r'G:\1_BeiJingUP\AUGB\Table\06-21yrs_Sites.xlsx', index=False)
    #
    # # Active Data
    # activeCalc = ['mean/8', 'max/1000', 'sum', 'sum']
    # for iactive in range(len(activePath)):
    #     filelist = glob(activePath[iactive]+os.sep+r'*'+activeKey[iactive]+r'*.tif')
    #     filelist.sort()
    #     print(filelist)
    #
    #     if activeKey[iactive] == 'NDVI':
    #         startDate = 4  # 5月开始
    #         endDate = 7  # 7月结束
    #     else:
    #         startDate = 16  # 第16个8天开始
    #         endDate = 30  # 第30个8天结束
    #     filelist = filelist[startDate:endDate]
    #     value = EsRaster.SampleRaster(filelist, sitedf, 'SiteName')
    #     if activeCalc[iactive].lower() == 'mean/3':
    #         valueCalc = [sum(v)/(endDate-startDate) for v in value]
    #     elif activeCalc[iactive].lower() == 'mean/8':
    #         valueCalc = [sum(v)/(endDate-startDate) for v in value]
    #     elif activeCalc[iactive].lower() == 'max/1000':
    #         valueCalc = [max(v)/1000 for v in value]
    #     else:
    #         valueCalc = [sum(v) for v in value]
    #     sitedf[activeKey[iactive]] = valueCalc
    #     # activeData.append(EsRaster.RasterCalc(filelist,activeCalc[iactive]))
    #
    # # Static Data
    # filelist = staticData
    # value = EsRaster.SampleRaster(filelist, sitedf, 'SiteName')
    # rfdf = pd.concat([sitedf, pd.DataFrame(value)], axis=1, join='outer')
    # rfdf.columns = list(sitedf.columns) + staticKey
    # # activeData.append(EsRaster.RasterCalc(filelist,activeCalc[iactive]))


    print('')

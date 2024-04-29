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
import EsRaster,readConfig,RF_Algorithm_V2_RF5_20240428
#SVM_Algorithm,Cubist_Algorithm


if __name__=="__main__":
    '''读取参数文件，并初始化参数'''
    if os.path.exists(EsInitialPath+os.sep+"Parameters20240428.ini"):
        inoutParas,excelParas,staticParasD,dynamicParasD,dymDayselect = readConfig.get_stcAnddym(EsInitialPath+os.sep+"Parameters20240428.ini")
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
        outPath = r'J:\1_BeiJingUP\AUGB\Data\20220629\Results'

        # SiteExcel
        siteFile = r'J:\1_BeiJingUP\AUGB\Table\ALL_SITES2010.csv'
        exlFile = r"J:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_2024\DataTable\AUGB汇总_AGB汇总ALLdata.xlsx"

        # Topography
        dem = r'J:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif'
        lat = r'J:\1_BeiJingUP\AUGB\Data\20220629\Parameters\LAT_China1km.tif'
        slp = r'J:\1_BeiJingUP\AUGB\Data\20220629\Parameters\slope_china1km.tif'
        asp = r'J:\1_BeiJingUP\AUGB\Data\20220629\Parameters\aspect_china1km.tif'
        # Vegetation
        cgc = r'J:\1_BeiJingUP\AUGB\Data\20220629\Parameters\cgrass_China1km.tif'
        staticPath = [dem,lat,slp,asp,cgc]
        staticKey = [f.split(os.sep)[-1].split(r'.')[0] for f in staticPath]

        # ==== Active Paramerters=====
        dynamicPath = [r'J:\1_BeiJingUP\AUGB\Data\20220629\TAVG',
                       r'J:\1_BeiJingUP\AUGB\Data\20220629\NDVI',
                       r'J:\1_BeiJingUP\AUGB\Data\20220629\PRCP',
                       r'J:\1_BeiJingUP\AUGB\Data\20220629\SWRS']

        dynamicKey = [f.split(os.sep)[-1] for f in dynamicPath]
        activeData = [] #dict(zip(activeKey,activePath))

    '''Pre-Process of Dynamic raster'''
    # 过程文件夹创建
    for vari in range(len(dynamicPreduceKey)):
        for seasoni in range(int(len(dynamicPreduceDays)/2)+1):
            if not os.path.exists(outPath + os.sep + dynamicPreduceKey[vari].upper() + str(seasoni)):
                os.makedirs(outPath + os.sep + dynamicPreduceKey[vari].upper() + str(seasoni))

    # # 预处理数据 产生过程文件
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
    # # J:\1_BeiJingUP\CommonData\temp\Export_Output_2.shp
    # # sitedf = pd.read_csv(exlFile, header=0)
    # # Load CSV files and Delete unstable Data
    # wb = openpyxl.load_workbook(exlFile)
    # sheetList = wb.sheetnames
    # allcsv = pd.DataFrame([])
    # for sheet in sheetList[:-1]:
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
    # # allcsv = allcsv[(allcsv['LON'] >= 73) & (allcsv['LON'] <= 135)]
    # # allcsv = allcsv[(allcsv['LAT'] > 0) & (allcsv['LAT'] <= 53)]
    # # allcsv.to_excel(outPath + os.sep + r'Table\All_yrs_Sites.xlsx', index=False)
    # allcsv.to_excel(r'J:\1_BeiJingUP\AUGB\Data\20220629\随机森林样本点\AUGB汇总_AGB汇总ALLdata_allyr1.xlsx', index=False)
    # print()
    #
    # '''## Sample ##'''
    # print(r'Sample : ',end=r'')
    # allyr = pd.DataFrame([])
    # for yr in range(2000,2020+1):
    #     yrcsv = allcsv[allcsv['Year']==yr]
    #     print(yr)
    #     if yrcsv.shape[0]==0:
    #         print(str(yr)+r'无数据跳过')
    #         continue
    #     '''Static raster Sample'''
    #     stcDf = EsRaster.SampleRaster_gdal3(staticPath+[r'J:\1_BeiJingUP\CommonData\中国地图-审图号GS(2019)1822号-shp格式\中国地图-审图号GS(2019)1822号-shp格式\全国县级数据_(唐古拉单独处理并更新青藏高原国界)\CHNCounty_ProjAlbers.tif'],
    #                                         yrcsv, 'ID', LonFieldName=r'LON', LatFieldName=r'LAT')
    #     yrcsv = pd.concat([yrcsv,stcDf],axis=1,join='outer')
    #
    #     '''Dynamic raster sample'''
    #     # npptif = [glob(r'J:\1_BeiJingUP\AUGB\Data\OtherNPPData' + os.sep + mdname + os.sep + r'Tiff_China\*' + str(yr) + r'*.tif') for mdname in
    #     #           ['BMA','GLOPEM-CEVSA','BEPS','MODIS','GLASS','MuSyQ']]  # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    #     # npptif = sum(npptif, [])  # 解决List嵌套
    #
    #     dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(dynamicParasD.keys()),list(dynamicParasD.values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    #     dymtif = sum(dymtif, [])    # 解决List嵌套
    #     dymDf = EsRaster.SampleRaster_gdal3(dymtif,yrcsv,'ID', LonFieldName=r'LON', LatFieldName=r'LAT',
    #                                         colName=['NDVI', 'Yearly_TAVG', 'Season1_TAVG', 'Season2_TAVG',
    #                                                  'Yearly_PRCP', 'Season1_PRCP', 'Season2_PRCP',
    #                                                  'Yearly_SWRS', 'Season1_SWRS', 'Season2_SWRS',
    #                                                  'Yearly_FPAR', 'Season1_FPAR', 'Season2_FPAR'])
    #                                                  #'BMA','GLOPEM-CEVSA','BEPS','MODIS','GLASS','MuSyQ'])
    #     yrcsv = pd.concat([yrcsv, dymDf], axis=1, join='outer')
    #
    #     '''Merge'''
    #     # yrcsv = yrcsv[(yrcsv[r'FPAR0_FPAR'] > 0) | (yrcsv[r'FPAR1_FPAR'] > 0) | (yrcsv[r'FPAR2_FPAR'] > 0)]
    #     # yrcsv = yrcsv[yrcsv[r'Soil_Clay.tif'] > -9999]
    #     allyr = pd.concat([allyr, yrcsv], axis=0, join='outer')
    #
    # # # Change allyr
    # # Ycols = ['ID', 'LON', 'LAT', 'AGB(gCm-2)', 'DataType', 'Ref', 'Ref_ID', 'AGB(gDMm-2)', 'Year', 'DEM', 'Parameters_LAT', 'Slope', 'Aspect', 'CGrass', 'Soil_Clay',
    # #          'Soil_CoarseSand', 'Soil_FineSand','Soil_OrganicMass', 'Soil_PH_h2o', 'Soil_PowderedSand', 'NDVI',
    # #          'Yearly_TAVG', 'Season1_TAVG', 'Season2_TAVG',
    # #          'Yearly_PRCP', 'Season1_PRCP', 'Season2_PRCP',
    # #          'Yearly_SWRS', 'Season1_SWRS', 'Season2_SWRS',
    # #          'Yearly_FPAR', 'Season1_FPAR', 'Season2_FPAR']
    # # # ['AGB', 'ID', 'LON', 'Parameters','Slope','Aspect','Soil_PH_h2o','Year','CGrass','Field1']]
    # # allyr.columns = Ycols
    # allyr.to_excel(r'J:\1_BeiJingUP\AUGB\Data\20220629\随机森林样本点\AUGB汇总_AGB汇总ALLdata_allyr2.xlsx', index=False)

    allyr = pd.read_excel(r'J:\1_BeiJingUP\AUGB\Data\20220629\随机森林样本点\AUGB汇总_AGB汇总ALLdata_allyr2.xlsx')

    mx = np.quantile(allyr['AGB(gDMm-2)'], 0.975)
    # allyr = allyr[allyr['AGB(gDMm-2)']<=mx]


    # a = allyr[((allyr[r'Province'] == 54))].loc[:, 'AGB(gDMm-2)']
    # np.mean(a)
    allyr = allyr[ ((allyr[r'Province'] == 54) & (allyr[r'AGB(gDMm-2)'] < 55) & (allyr[r'DataType'].isna()))
                   | ((allyr[r'Province'] == 54) & (~allyr[r'DataType'].isna()))
                   | (allyr[r'Province']!=54) ]


    '''Algth Prepare'''
    Ycols = ['LAT', 'Parameters_dem', #'Parameters_slope', 'Parameters_aspect',
             'Soil_Clay.tif', 'Soil_CoarseSand.tif', 'Soil_FineSand.tif',
             'Soil_OrganicMass.tif', 'Soil_PH', 'Soil_PowderedSand.tif',
             'NDVI', 'Yearly_TAVG', 'Season1_TAVG', 'Season2_TAVG',
             'Yearly_PRCP', 'Season1_PRCP', 'Season2_PRCP',
             'Yearly_SWRS', 'Season1_SWRS', 'Season2_SWRS',
             'Yearly_FPAR', 'Season1_FPAR', 'Season2_FPAR']
    # Ycols = [i for i in allyr.columns
    #          if i not in ['ID', 'LON', 'AGB(gCm-2)', 'DataType', 'Ref', 'Ref_ID',
    #                       'Year', 'Parameters_LAT', 'Parameters_cgrass','全国县级数据_(唐古拉单独处理并更新青藏高原国界)_CHNCounty',
    #                       'BMA', 'GLOPEM-CEVSA', 'BEPS', 'MODIS', 'GLASS', 'MuSyQ']]#,'Soil_Clay.tif', 'Soil_CoarseSand.tif', 'Soil_FineSand.tif','Soil_OrganicMass.tif', 'Soil_PH', 'Soil_PowderedSand.tif', 'Parameters_cgrass']]  # 'Year', or Ycols = allyr[allyr.columns.difference(['A', 'B'])]
    # [dynamicParasD.pop(ki) for ki in ['tavg1','tavg2','prcp1','prcp2','swrs1','swrs2','fpar1','fpar2']]

    allyr = allyr[['LON','Year','AGB(gDMm-2)']+Ycols].groupby(['LON','LAT','Year']).mean()
    allyr = allyr.reset_index(drop=False)

    print(allyr)

    algY = allyr['AGB(gDMm-2)']
    algX = allyr[Ycols]
    print(Ycols)
    RF_Algorithm_V2_RF5_20240428.RFEstimate(algX, algY, [staticPath, dynamicParasD], 0.92)

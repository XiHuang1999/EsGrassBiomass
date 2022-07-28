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

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster
import readConfig


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

    for yr in range(2000,2021):         # 按年遍历
        print(yr,end=' ==>> ')
        for vari in dynamicPreduceKey:  # 每年的每个变量
            # 列举tif文件
            filelist = glob(dynamicParasD[vari] + os.sep + r'*' + vari.upper()+ r'*_' + str(yr) + r'*.tif')
            filelist.sort()

            # 计算8天数据的季节值
            for seasoni in range(int(len(dynamicPreduceDays)/2)):
                # 输出判断
                print(str(seasoni), end=' ')
                outRasterFileName = outPath + os.sep + vari.upper() + str(seasoni+1) + os.sep + vari.upper()+r'_'+ str(yr) +r'.tif'
                if os.path.exists(outRasterFileName):
                    continue
                # 转换闰年天数
                if yr % 4 != 0:
                    startDay = dynamicPreduceDays[seasoni * 2]
                    endDay = dynamicPreduceDays[seasoni * 2 + 1]
                else:
                    startDay = dynamicPreduceDays[seasoni * 2] +1
                    endDay = dynamicPreduceDays[seasoni * 2 + 1] +1

                # 将第i天转换为第j个8天 [upper integer]
                start8Day = -(-startDay // 8)    # 第16个8天开始 此处的16开始为1 而非0开始
                end8Day = -(-endDay // 8)  # 第31个8天结束
                bdays = 8 - startDay % 8 + 1  # 在要处理的8天数据中，第一个8天的天数
                edays = endDay % 8  # 在要处理的8天数据中，最后一个8天的天数

                # 根据时间范围处理 [8天数据处理]
                eightDaysFilelist = filelist[start8Day-1:end8Day]
                if vari.upper() == r'TAVG':
                    rasterR, proj, geotrans = EsRaster.RasterMean_8Days(eightDaysFilelist,bdays,edays)
                else:
                    rasterR, proj, geotrans = EsRaster.RasterSum_8Days(eightDaysFilelist, bdays, edays)

                # 输出过程栅格
                EsRaster.write_img(outRasterFileName, proj, geotrans, rasterR)
                # del rasterR
                # # 下一次循环变量赋值
                # start8Day = dynamicPreduceDays[seasoni+1 * 2] // 8   # 第16个8天开始 此处的16开始为1
                # end8Day = dynamicPreduceDays[seasoni+1 * 2 + 1] // 8  # 第30个8天结束

            # 计算8天数据的年值
            print(vari,end=' ')
            outRasterFileName = outPath + os.sep + vari.upper() + r'0' + os.sep + \
                                vari.upper() + r'_' + str(yr) + r'.tif'
            if os.path.exists(outRasterFileName):
                continue
            # 数据处理
            bdays = 8
            if yr % 4 != 0:
                bdays = 5
            else:
                bdays = 6
            if vari.upper() == r'TAVG':
                rasterR, proj, geotrans = EsRaster.RasterMean_8Days(filelist, bdays, edays)
            else:
                rasterR, proj, geotrans = EsRaster.RasterSum_8Days(filelist, bdays, edays)
            # 输出过程栅格
            EsRaster.write_img(outRasterFileName, proj, geotrans, rasterR)
        print('')

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


    '''读取站点文件'''
    # sitedf = pd.read_csv(exlFile, header=0)
    # Load CSV files and Delete unstable Data
    wb = openpyxl.load_workbook(exlFile)
    sheetList = wb.sheetnames
    allcsv = pd.DataFrame([])
    for sheet in sheetList[:-19]:
        print(sheet)
        csv = pd.read_excel(exlFile, sheet_name=sheet)
        # print(type(csv.LON[0]))
        csv[r'Year'] = [int(sheet) for y in range(csv.shape[0])]
        allcsv = pd.concat([allcsv, csv])

    allcsv1 = allcsv
    allcsv = allcsv[allcsv.AGB < allcsv.AGB.std() * 3 + allcsv.AGB.mean()]
    allcsv = allcsv[(allcsv['LON'] >= 73) & (allcsv['LON'] <= 135)]
    allcsv = allcsv[(allcsv['LAT'] > 0) & (allcsv['LAT'] <= 53)]
    # allcsv.to_excel(outPath + os.sep + r'Table\All_yrs_Sites.xlsx', index=False)
    # allcsv.to_csv(outPath + os.sep + r'Table\All_yrs_Sites1.csv', index=False)


    '''## Sample ##'''
    allyr = pd.DataFrame([])
    for yr in range(2000,2021):
        yrcsv = allcsv[allcsv['Year']==yr]
        '''Static raster Sample'''
        stcDf = EsRaster.SampleRaster(staticPath, yrcsv, 'ID')
        yrcsv = pd.concat([yrcsv,stcDf],axis=1,join='outer')

        '''Dynamic raster sample'''
        dymtif = [dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif' for dymKey,dymPath in zip(list(dynamicParasD.keys())[1:],list(dynamicParasD.values())[1:])]
        dymDf = EsRaster.SampleRaster(dymtif,yrcsv,'ID')
        yrcsv = pd.concat([yrcsv, dymDf], axis=1, join='outer')

        '''Merge'''
        allyr = pd.concat([allyr, yrcsv], axis=0, join='outer')
    print('')
    # for sheet in sheetList[:-1]:







    for sheet in sheetList[:-1]:
        print(sheet)
        csv = pd.read_excel(exlFile, sheet_name=sheet)
        csv[r'Year'] = [int(sheet) for y in range(csv.shape[0])]
        allcsv = pd.concat([allcsv, csv])
        csv.to_csv(exlFile.replace(r'ALL_SITES_select.xlsx', r'ALL_SITES_' + str(sheet) + r'.csv'))
    allcsv.to_csv(r'G:\1_BeiJingUP\AUGB\Table\06-21yrs_Sites.csv')
    allcsv.to_excel(r'G:\1_BeiJingUP\AUGB\Table\06-21yrs_Sites.xlsx', index=False)

    # Active Data
    activeCalc = ['mean/8', 'max/1000', 'sum', 'sum']
    for iactive in range(len(activePath)):
        filelist = glob(activePath[iactive]+os.sep+r'*'+activeKey[iactive]+r'*.tif')
        filelist.sort()
        print(filelist)

        if activeKey[iactive] == 'NDVI':
            startDate = 4  # 5月开始
            endDate = 7  # 7月结束
        else:
            startDate = 16  # 第16个8天开始
            endDate = 30  # 第30个8天结束
        filelist = filelist[startDate:endDate]
        value = EsRaster.SampleRaster(filelist, sitedf, 'SiteName')
        if activeCalc[iactive].lower() == 'mean/3':
            valueCalc = [sum(v)/(endDate-startDate) for v in value]
        elif activeCalc[iactive].lower() == 'mean/8':
            valueCalc = [sum(v)/(endDate-startDate) for v in value]
        elif activeCalc[iactive].lower() == 'max/1000':
            valueCalc = [max(v)/1000 for v in value]
        else:
            valueCalc = [sum(v) for v in value]
        sitedf[activeKey[iactive]] = valueCalc
        # activeData.append(EsRaster.RasterCalc(filelist,activeCalc[iactive]))

    # Static Data
    filelist = staticData
    value = EsRaster.SampleRaster(filelist, sitedf, 'SiteName')
    rfdf = pd.concat([sitedf, pd.DataFrame(value)], axis=1, join='outer')
    rfdf.columns = list(sitedf.columns) + staticKey
    # activeData.append(EsRaster.RasterCalc(filelist,activeCalc[iactive]))


    print('')
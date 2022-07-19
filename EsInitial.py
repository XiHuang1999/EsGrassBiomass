# -*- coding: utf-8 -*-

# @Time : 2022-06-29 15:41
# @Author : XiHuang O.Y.
# @Site : 
# @File : EsInitial.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os, sys
from glob import glob
import openpyxl,os

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster
import readConfig


if __name__=="__main__":
    '''读取参数文件，并初始化参数'''
    if os.path.exists(EsInitialPath+os.sep+"Parameters.ini"):
        inoutParas,excelParas,staticParas,dynamicParas = readConfig.get_stcAnddym()
        outPath = inoutParas['out']
        exlFile = excelParas['exlfile']
        staticKey = list(staticParas.keys())
        dynamicKey = list(dynamicParas.keys())
        staticPath = list(staticParas.values())
        dynamicPath = list(dynamicParas.values())
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
        activePath = [r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\PRCP',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS']

        activeKey = [f.split(os.sep)[-1] for f in activePath]
        activeData = [] #dict(zip(activeKey,activePath))

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

    '''Static raster Sample'''
    vv = EsRaster.SampleRaster(staticPath,r'G:\1_BeiJingUP\CommonData\temp\Export_Output_2.shp','ID')
    '''Dynamic raster Sample'''
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
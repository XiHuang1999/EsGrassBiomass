# -*- coding: utf-8 -*-

# @Time : 2022-07-05 11:34
# @Author : XiHuang O.Y.
# @Site :
# @File : Other_procedure.py
# @Software: PyCharm

import openpyxl,os
import pandas as pd

exlFile = r"G:\1_BeiJingUP\AUGB\Table\ALL_SITES_select.xlsx"
wb = openpyxl.load_workbook(exlFile)
sheetList = wb.sheetnames
allcsv = pd.DataFrame([])
for sheet in sheetList[:-1]:
    print(sheet)
    csv = pd.read_excel(exlFile,sheet_name=sheet)
    csv[r'Year']=[int(sheet) for y in range(csv.shape[0])]
    allcsv = pd.concat([allcsv,csv])
    csv.to_csv(exlFile.replace(r'ALL_SITES_select.xlsx',r'ALL_SITES_'+str(sheet)+r'.csv'))
allcsv.to_csv(r'G:\1_BeiJingUP\AUGB\Table\06-21yrs_Sites.csv')
allcsv.to_excel(r'G:\1_BeiJingUP\AUGB\Table\06-21yrs_Sites.xlsx',index=False)

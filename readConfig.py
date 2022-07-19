#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/16 8:57
# @Author : Xihuang O.Y.
import os
import configparser

# # Reading txt config file
# f=open(r'E:\A_UCAS_Study\ParametersEsAGB.txt', encoding='gbk')
# for var in ['stc','dym']:
#     exec(var + 'ParaName=[]\n' + var + 'ParaValues=[]')
# for line in f:
#     if line.strip().split(u'\t')[0].lower()=='dynamic':
#         break
#     stcParaName.append(line.strip().split(u'\t')[0])
#     stcParaValues.append(line.strip().split(u'\t')[1])
# for line in f:
#     stcParaName.append(line.strip().split(u'\t')[0])
#     stcParaValues.append(line.strip().split(u'\t')[1])
# staticPara = dict(zip(staticParaName,staticParaValues))
# print(staticPara)

#os.path.realpath：获取当前执行脚本的绝对路径。
#os.path.split：如果给出的是一个目录和文件名，则输出路径和文件名
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "Parameters.ini")


class ReadConfig:
    def __init__(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(configPath)

    def get_inout(self, param):
        value = self.cf.get("inout", param)
        return value

    def get_static(self, param):
        value = self.cf.get("static", param)
        return value

    def get_dynamic(self, param):
        value = self.cf.get("dynamic", param)
        return value

def get_stcAnddym():
    '''
    函数读取参数文件
    '''
    # Creat configparaser objection
    configP = configparser.ConfigParser()
    # Read config file
    configP.read(configPath)
    # Get all sections
    sections = configP.sections()
    print(sections, end='\n')
    # Read para
    exec('inout=[]\nstcParaName=[]\ndymParaValues=[]')
    inout = configP.items('inout')
    tbName = configP.items('table')
    stcParaName = configP.items('static')
    dymParaName = configP.items('dynamic')
    inout = dict(inout)
    excelTb = dict(tbName)
    stcParaName = dict(stcParaName)
    dymParaName = dict(dymParaName)
    print(inout)
    return inout,excelTb,stcParaName,dymParaName

if __name__ == '__main__':
    para = get_stcAnddym()
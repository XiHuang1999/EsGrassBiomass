# -*- coding: utf-8 -*-

# @Time : 2022-08-07 11:50
# @Author : XiHuang O.Y.
# @Site : 
# @File : Cubist_Algorithm.py
# @Software: PyCharm

import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from time import time
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as st
from cubist import Cubist
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
# import pydotplus
import os, sys
from glob import glob

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig,RF_Algorithm


def CBEstimate(X,
               Y,
               para_Output,
               train_size=0.9,CB_kwargs={'neighbors': 9, 'n_rules': 465, 'n_committees': 16, 'composite': True}

               # {'neighbors': 9, 'n_rules': 465, 'n_committees': 16, 'composite': True}
                ):
    '''
    使用Cubist预测地上生物量
    :param X: 自变量,X = dataset.iloc[:, 1:end]
    :param Y: 因变量,y = dataset.iloc[:, 0]
    :param para_Output: list,[staticDataPath,dynamicDataPath]
    :param train_size: float,训练样本
    :param CB_kwargs: dict,Parameters of RF
    :return:
    '''
    print(r"Start Cubist Predict AGB:")
    t1 = time()

    # 将数据分为训练样本和验证样本
    X = pd.DataFrame(X)
    Y = pd.Series(Y)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1-train_size,
                                                        random_state=0)
    # regressor = RandomForestRegressor()
    # RF_kwargs = {'n_estimators': 561, 'max_features': 5, 'bootstrap': True}#{'n_estimators': 538, 'max_leaf_nodes': 10, 'max_features': 19, 'bootstrap': True}#
                # {'n_estimators': 589, 'max_features': 7, 'bootstrap': True}
    regressor = Cubist(**CB_kwargs)

    # # Parameters before
    # regressor.fit(X_train, y_train)
    # outResults1 = regressor.predict(X_test)
    # score = explained_variance_score(y_test, outResults1)
    # print(score)  # 得到预测结果区间[0,1]

    # # # Parameters 1
    # param_distribs = {
    #     # 均匀离散随机变量
    #     'n_rules': [int(x) for x in np.linspace(start=400, stop=600, num=80)],
    #     'n_committees': [int(x) for x in np.linspace(start=1, stop=20, num=20)],  # 寻找最佳分割时要考虑的特征数量
    #     'neighbors': [int(x) for x in np.linspace(start=1,stop=9,num=9)],
    #     'composite': [True]
    # }
    # regressor = RandomizedSearchCV(regressor, param_distributions=param_distribs,
    #                               n_iter=5000, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # # # Parameters 2
    # param_distribs = {
    #     # 均匀离散随机变量
    #     'n_estimators': [int(x) for x in np.linspace(start=300, stop=700, num=400)],
    #     'max_features': [int(x) for x in np.linspace(start=1, stop=15, num=13)],  # 寻找最佳分割时要考虑的特征数量
    #     #'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=10,num=8)],
    #     #'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=20)],           # 树的最大深度
    #     #'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=12,num=10)],    # 在叶节点处需要的最小样本数
    #     #'min_samples_split': [int(x) for x in np.linspace(start=2,stop=10,num=8)],   # 拆分内部节点所需的最少样本数
    #     "bootstrap": [True, False]
    # }
    # regressor = GridSearchCV(regressor, param_grid=param_distribs, cv=5, n_jobs=-1, scoring='r2')

    # Regression
    regressor.fit(X_train, y_train)

    outResults1 = regressor.predict(X_train)
    score = explained_variance_score(y_train, outResults1)
    print('Training score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    outResults2 = regressor.predict(X_test)
    score = explained_variance_score(y_test, outResults2)
    print('Test score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    outResults = regressor.predict(X)
    plt.scatter(Y, outResults)
    plt.show()
    score = explained_variance_score(Y, outResults)
    print('Allset score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    print('Time Using:%.2f min' % ((time()-t1)/60),end=' / ')


    # # region Drawing
    # xy = np.vstack([Y, outResults])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
    # pltx = [int(x) for x in np.linspace(start=0, stop=max(Y), num=1000)]
    # plty = [linreg.slope * x + linreg.intercept for x in pltx]
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(pltx, plty, '-', color='red', alpha=0.8, linewidth=2, label='Fitting Line')    # color='#4169E1'
    # plt.plot(pltx, pltx, '-', color='black', alpha=0.8, linewidth=2, label='1:1')
    # plt.scatter(Y, outResults,c=z,s=1.3,cmap='Spectral')
    # plt.text(400,235,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
    #          '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
    #          '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
    #          ,fontsize=8,color = "r",fontweight='bold')
    # plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    # plt.xlabel('AGB Site Value')  # 添加x轴和y轴标签
    # plt.ylabel('Model Value')
    # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'CB_results.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.show()
    # # endregion


    # print(regressor.get_params().values())
    # print(regressor.best_params_,end='\n\n')
    # print(regressor.best_estimator_)

    # Out PIC
    # importances = regressor.feature_importances_

    # read and predict
    # read and predict
    para_Output[0]=[r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\LAT_China1km.tif',
                    r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\Clay.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\CoarseSand.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\FineSand.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\OrganicMass.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\PowderedSand.tif']
    for yr in range(2000,2021):
        # print(yr)
        # '''Data Tif List'''
        # # staticPath
        # dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(para_Output[1].keys()),list(para_Output[1].values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
        # dymtif = sum(dymtif, [])    # 解决List嵌套
        #
        # yrTifList = para_Output[0]+dymtif
        # dataDf, im_proj, im_geotrans = EsRaster.read_tifList(yrTifList)
        #
        # # Block split
        # xid = dataDf[(dataDf.iloc[:, 3] > -1) & (dataDf.iloc[:, 4] > 0) & (dataDf.iloc[:, 4] < 100) & (dataDf.iloc[:, 5] > -9999) & (dataDf.iloc[:, 11] >= 0) & (dataDf.iloc[:, 12] > -9999) & (dataDf.iloc[:, 23] > 0)].index.tolist()
        # indf = dataDf.iloc[xid, :]
        # indf = np.array_split(indf,200)
        # for i in range(0,200):
        #     indf[i].columns = X.columns
        #
        # estimators = [regressor for i in range(200)]
        #
        # '''Predict'''
        # kwgs = list(zip(estimators, indf))
        # outResults_ = EsRaster.run_imap_mp(EsRaster.generate_mulcpu_vars_Predict,kwgs,num_processes=30, is_tqdm=True)
        # outResults_ = np.hstack(tuple(outResults_))
        # outResults_ = outResults_.reshape([len(xid), 1])
        # outdat = np.zeros((4998 * 4088, 1)) - 9999
        # outdat[xid] = outResults_
        # outdat = outdat.reshape(4088,4998)

        print(yr)
        '''Data Tif List'''
        # staticPath
        dymtif = [glob(dymPath + os.sep + r'*' + str(yr) + r'*.tif') for dymKey, dymPath in
                  zip(list(para_Output[1].keys()),
                      list(para_Output[1].values()))]  # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
        dymtif = sum(dymtif, [])  # 解决List嵌套
        yrTifList = para_Output[0] + dymtif
        dataDf, im_proj, im_geotrans = EsRaster.read_tifList(yrTifList)

        # Block split
        # # clay>-1        0<cgrass<100      NDVI>=0              prcp>-9999               fpar>0
        # xid = dataDf[(dataDf.iloc[:, 3] > -1) & (dataDf.iloc[:, 2] > 0) & (dataDf.iloc[:, 2] < 100) & (dataDf.iloc[:, 5] > -9999) & (dataDf.iloc[:, 8] >= 0) & (dataDf.iloc[:, 12] > -9999) & (dataDf.iloc[:, 18] > 0)].index.tolist()
        # clay>-1        0<cgrass<100      NDVI>=0              prcp>-9999               fpar>0
        xid = dataDf[(dataDf.iloc[:, 2] > -1) & (dataDf.iloc[:, 3] > -9999) & (dataDf.iloc[:, 7] >= 0) & (
                    dataDf.iloc[:, 12] > -9999) & (dataDf.iloc[:, 18] > 0)].index.tolist()
        indf = dataDf.iloc[xid, :]
        indf.columns = X.columns  # [X_columns[0]] + X_columns[2:]
        # indf = indf.reindex(columns=X_columns, fill_value=yr)
        indf = np.array_split(indf, 200)
        for i in range(0,200):
            indf[i].columns = X.columns
        estimators = [regressor for i in range(200)]

        '''Predict'''
        kwgs = list(zip(estimators, indf))
        outResults_ = EsRaster.run_imap_mp(EsRaster.generate_mulcpu_vars_Predict, kwgs, num_processes=20, is_tqdm=True)
        outResults_ = np.hstack(tuple(outResults_))
        outResults_ = outResults_.reshape([-1, 1])
        outdat = np.zeros((4998 * 4088, 1)) - 9999
        outdat[xid] = outResults_
        outdat = outdat.reshape(4088, 4998)

        EsRaster.write_img(r"G:\1_BeiJingUP\AUGB\Data\20220629\Results\CB_AGB_4"+os.sep+r'CB_AGB_'+str(yr)+r'.tif', im_proj, im_geotrans, outdat)
    print()

    return


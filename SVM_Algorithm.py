# -*- coding: utf-8 -*-

# @Time : 2022-08-07 11:41
# @Author : XiHuang O.Y.
# @Site : 
# @File : SVM_Algorithm.py
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
import scipy.stats as st

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
# import pydotplus
import os, sys
from glob import glob
from sklearn.svm import SVR


EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig

def SVMEstimate(X,
               Y,
               para_Output,
               train_size=0.9,RF_kwargs={'n_estimators':566,'max_features':5 ,'bootstrap':True}
                            # 树的棵树，默认是100     # 寻找最佳分割时要考虑的特征数量
                          #'max_depth':None,
                          #'random_state':None, # 控制构建树时样本的随机抽样
                          # 'bootstrap':True,   # 建立树木时是否使用bootstrap抽样
                          # 'max_depth':10,          # 树的最大深度
                          # 'max_features':2,         # 寻找最佳分割时要考虑的特征数量
                          # 'min_samples_leaf':3,   # 在叶节点处需要的最小样本数
                          # 'min_samples_split':5  # 拆分内部节点所需的最少样本数
               # {'n_estimators': 561, 'max_features': 5, 'bootstrap': True}
                # {'n_estimators': 589, 'max_features': 7, 'bootstrap': True}
                ):
    '''

    :param X: 自变量,X = dataset.iloc[:, 0:4].values
    :param Y: 因变量,y = dataset.iloc[:, 4].values
    :param para_Output: list, 输出参数
    :param train_size: float, 训练样本比例
    :param SVM_kwargs: 参数
    :return:
    '''

    print(r"Start SVM Predict AGB:")
    t1 = time()

    # 将数据分为训练样本和验证样本
    # Y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1-train_size,
                                                        random_state=0)

    # 线性核函数 对大样本多变量很难找到线性拟合
    # lin_svr = SVR(kernel='linear',verbose=True,tol = 1)
    # lin_svr.fit(X_train, y_train)
    # lin_svr_pred = lin_svr.predict(X_test)
    # print("正确率为：", explained_variance_score(y_test,lin_svr_pred))   #r2_score(y_test,rbf_svr_pred))
    # print('Time Using:%.2f min' % ((time()-t1)/60),end=' / ')

    # region 多项式核函数
    regressor = SVR()
    # # Parameters 1
    param_distribs = {
        # 均匀离散随机变量
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [int(x) for x in np.linspace(start=7, stop=13, num=7)],
        'gamma': ['scale','auto'],
        'tol': [0.001,0.1,1,10,20,30,50,100,200,300,1000]
    }
    regressor = RandomizedSearchCV(regressor, param_distributions=param_distribs,
                                  n_iter=460, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    regressor.fit(X_train, y_train)
    allResults = regressor.predict(X)
    trainsResults = regressor.predict(X_train)
    testResults = regressor.predict(X_test)

    linreg1 = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(allResults, dtype=np.float64))
    alls = (linreg1.rvalue) ** 2
    linreg2 = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(trainsResults, dtype=np.float64))
    trains = (linreg2.rvalue) ** 2
    linreg3 = st.linregress(pd.Series(y_test, dtype=np.float64), pd.Series(testResults, dtype=np.float64))
    tests = (linreg3.rvalue) ** 2

    print()
    # print("正确率为：", explained_variance_score(y_test, poly_svr_pred))  # r2_score(y_test,rbf_svr_pred))
    # poly_svr_pred = poly_svr.predict(X_test)
    # endregion

    # 径向基核函数
    rbf_svr = SVR(kernel='rbf')
    rbf_svr.fit(X_train, y_train)
    rbf_svr_pred = rbf_svr.predict(X_test)
    print("正确率为：", explained_variance_score(y_test, rbf_svr_pred))  # r2_score(y_test,rbf_svr_pred))
    # sigmoid核
    rbf_svr = SVR(kernel='sigmoid')
    rbf_svr.fit(X_train, y_train)
    sig_svr_pred = rbf_svr.predict(X_test)
    print("正确率为：", explained_variance_score(y_test, sig_svr_pred))  # r2_score(y_test,rbf_svr_pred))

    plt.scatter(Y, y, color='darkorange', label='data')
    plt.hold('on')
    lw = 2
    # plt.plot(X, lin_svr_pred, color='c', lw=lw, label='Linear model')
    plt.plot(X, rbf_svr_pred, color='navy', lw=lw, label='RBF model')
    plt.plot(X, poly_svr_pred, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.plot(X, sig_svr_pred, color='cornflowerblue', lw=lw, label='Sigmoid model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    print()

    # # Parameters 1
    # param_distribs = {
    #     # 均匀离散随机变量
    #     'n_estimators': [int(x) for x in np.linspace(start=500, stop=600, num=60)],
    #     'max_features': [int(x) for x in np.linspace(start=1, stop=20, num=20)],  # 寻找最佳分割时要考虑的特征数量
    #     'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=10,num=8)],
    #     #'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=20)],           # 树的最大深度
    #     #'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=12,num=10)],    # 在叶节点处需要的最小样本数
    #     #'min_samples_split': [int(x) for x in np.linspace(start=2,stop=10,num=8)],   # 拆分内部节点所需的最少样本数
    #     "bootstrap": [True, False]
    # }
    # regressor = RandomizedSearchCV(regressor, param_distributions=param_distribs,
    #                               n_iter=16000, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # # Parameters 2
    param_distribs = {
        # 均匀离散随机变量
        'n_estimators': [int(x) for x in np.linspace(start=300, stop=700, num=400)],
        'max_features': [int(x) for x in np.linspace(start=1, stop=15, num=13)],  # 寻找最佳分割时要考虑的特征数量
        #'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=10,num=8)],
        #'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=20)],           # 树的最大深度
        #'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=12,num=10)],    # 在叶节点处需要的最小样本数
        #'min_samples_split': [int(x) for x in np.linspace(start=2,stop=10,num=8)],   # 拆分内部节点所需的最少样本数
        "bootstrap": [True, False]
    }
    regressor = GridSearchCV(regressor, param_grid=param_distribs, cv=5, n_jobs=-1, scoring='r2')

    # Regression
    regressor.fit(X_train, y_train)

    outResults1 = regressor.predict(X_train)
    score = explained_variance_score(y_train, outResults1)
    print('Training score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    outResults2 = regressor.predict(X_test)
    score = explained_variance_score(y_test, outResults2)
    print('Test score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    outResults = regressor.predict(X)
    # plt.scatter(Y, outResults)
    # plt.show()
    score = explained_variance_score(Y, outResults)
    print('Allset score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    print('Time Using:%.2f min' % ((time()-t1)/60),end=' / ')

    print(regressor.get_params().values())
    print(regressor.best_params_,end='\n\n')
    print(regressor.best_estimator_)

    # Out PIC
    # importances = regressor.feature_importances_
    #
    # # read and predict
    # for yr in range(2000,2021):
    #     print(yr)
    #     '''Data Tif List'''
    #     # staticPath
    #     dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(para_Output[1].keys()),list(para_Output[1].values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    #     dymtif = sum(dymtif, [])    # 解决List嵌套
    #
    #     yrTifList = para_Output[0]+dymtif
    #     dataDf, im_proj, im_geotrans = EsRaster.read_tifList(yrTifList)
    #
    #     # Block split
    #     xid = dataDf[(dataDf.iloc[:, 3] > -1) & (dataDf.iloc[:, 4] > 0) & (dataDf.iloc[:, 4] < 100) & (dataDf.iloc[:, 5] > -9999) & (dataDf.iloc[:, 11] >= 0) & (dataDf.iloc[:, 12] > -9999) & (dataDf.iloc[:, 23] > 0)].index.tolist()
    #     indf = dataDf.iloc[xid, :]
    #     indf = np.array_split(indf,200)
    #     estimators = [regressor for i in range(200)]
    #
    #     '''Predict'''
    #     kwgs = list(zip(estimators, indf))
    #     outResults_ = run_imap_mp(generate_mulcpu_vars,kwgs,num_processes=20, is_tqdm=True)
    #     outResults_ = np.hstack(tuple(outResults_))
    #     outResults_ = outResults_.reshape([-1, 1])
    #     outdat = np.zeros((4998 * 4088, 1)) - 9999
    #     outdat[xid] = outResults_
    #     outdat = outdat.reshape(4088,4998)
    #
    #     EsRaster.write_img(r"G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB"+os.sep+r'RF_AGB_'+str(yr)+r'.tif', im_proj, im_geotrans, outdat)
    # print()

    return


def generate_mulcpu_vars(args):
    '''
    并行多参数调用
    :param args: 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    :return:
    '''
    return predict_Block(args[0],args[1])


def predict_Block(est,dataX):
    '''
    Predict by block
    :param est: obj, estimator
    :param dataX: ndarray, X var after np.split(a, 2, axis=1)
    :return: estimator result
    '''
    r = est.predict(dataX)
    return r

def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    '''
    并行计算启动器,形象化并行计算并合理分配内存。
    :param func: function,函数
    :param argument_list: list,参数列表
    :param num_processes: int,进程数，不填默认为总核心3
    :param is_tqdm: bool,是否展示进度条，默认True
    :return: 并行返回值
    '''
    result_list_tqdm = []
    try:
        if num_processes == '':
            num_processes = multiprocessing.cpu_count()-3
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func,argument_list))
    return result_list_tqdm


if __name__=="__main__":
    '''读取参数文件，并初始化参数'''
    if os.path.exists(EsInitialPath + os.sep + "Parameters.ini"):
        inoutParas, excelParas, staticParasD, dynamicParasD, dymDayselect = readConfig.get_stcAnddym()
        outPath = inoutParas['out']  # 输出路径
        exlFile = excelParas['exlfile']  # excel文件
        staticKey = list(staticParasD.keys())  # 静态数据关键字
        staticPath = list(staticParasD.values())  # 静态数据文件路径
        # dynamicKey = list(dynamicParasD.keys())  #动态数据关键字数
        # dynamicPath = list(dynamicParasD.values())   #动态数据文件路径
        exec('dynamicPreduceKey=' + dymDayselect['8daysdataname'])  # 动态数据中8天数据
        exec('dynamicPreduceDays=' + dymDayselect['daysscope'])  # 动态数据中选择范围
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
        staticPath = [dem, lat, slp, asp, cgc]
        staticKey = [f.split(os.sep)[-1].split(r'.')[0] for f in staticPath]

        # ==== Active Paramerters=====
        dynamicPath = [r'G:\1_BeiJingUP\AUGB\Data\20220629\TAVG',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\PRCP',
                       r'G:\1_BeiJingUP\AUGB\Data\20220629\SWRS']

        dynamicKey = [f.split(os.sep)[-1] for f in dynamicPath]
        activeData = []  # dict(zip(activeKey,activePath))

    '''Pre-Process of Dynamic raster'''
    # 过程文件夹创建
    for vari in range(len(dynamicPreduceKey)):
        for seasoni in range(int(len(dynamicPreduceDays) / 2) + 1):
            if not os.path.exists(outPath + os.sep + dynamicPreduceKey[vari].upper() + str(seasoni)):
                os.makedirs(outPath + os.sep + dynamicPreduceKey[vari].upper() + str(seasoni))

    # 并更新数据路径
    try:
        for vari in dynamicPreduceKey:
            dynamicParasD.pop(vari.lower())
            for seasoni in range(int(len(dynamicPreduceDays) / 2) + 1):
                dynamicParasD.update({vari + str(seasoni): outPath + os.sep + vari.upper() + str(seasoni)})
    except Exception as E_results:
        print('预处理有异常：', E_results)
        print('\n\n\n我不管，预处理错误,你看着办吧!\nPreProcessing File Direction is ERROR！Please Check it！')
        time.sleep(100000)
    finally:  # finally的代码是肯定执行的，不管是否有异常,但是finally语块是可选的。
        print('', end='')

    dynamicParasD['ndvi'] = r'G:\1_BeiJingUP\AUGB\Data\20220629\NDVI\AfterPreProcess2'

    '''Algth Prepare'''
    # allyr = pd.read_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr.csv", index_col=0)
    # allyr = pd.read_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr_NDVI0-1.csv",index_col=0)
    allyr = pd.read_csv(r"G:\1_BeiJingUP\AUGB\Data\20220629\allyr_SetRange2-600.csv", index_col=0)
    allyr = allyr[(allyr[r'Soil_Clay'] > -9999) & (allyr[r'CGrass'] < 100) & (allyr[r'CGrass'] > 0)]


    allyr = allyr.iloc[:, 1:]

    Ycols = [i for i in allyr.columns if
             i not in ['AGB', 'ID', 'LON', 'Parameters_LAT', 'Year']]
    algY = allyr['AGB'].values
    algX = allyr[Ycols].values

    # region zscore标准化
    from sklearn import preprocessing
    zscore = preprocessing.StandardScaler()
    algX = zscore.fit_transform(algX)
    # endregion

    SVMEstimate(algX, algY, [staticPath, dynamicParasD], 0.9)
    print()
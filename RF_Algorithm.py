# -*- coding: utf-8 -*-

# @Time : 2022-07-01 16:30
# @Author : XiHuang O.Y.
# @Site : 
# @File : RF_Algorithm.py
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

EsInitialPath = os.getcwd()
sys.path.append(EsInitialPath)      # 添加函数文件位置
import EsRaster,readConfig,RF_Algorithm

def RFEstimate(X,
               Y,
               para_Output,
               train_size=0.9,RF_kwargs={'n_estimators': 589, 'max_features': 7, 'bootstrap': True}
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
    使用随机森林预测地上生物量
    :param X: 自变量,X = dataset.iloc[:, 1:end].values
    :param Y: 因变量,y = dataset.iloc[:, 0].values
    :param para_Output: list,[staticDataPath,dynamicDataPath]
    :param train_size: float,训练样本
    :param RF_kwargs: dict,Parameters of RF
    :return:
    '''
    print(r"Start RF Predict AGB:")
    t1 = time()

    # 将数据分为训练样本和验证样本
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1-train_size,
                                                        random_state=0)
    # regressor = RandomForestRegressor()
    # RF_kwargs = {'n_estimators': 561, 'max_features': 5, 'bootstrap': True}#{'n_estimators': 538, 'max_leaf_nodes': 10, 'max_features': 19, 'bootstrap': True}#
                # {'n_estimators': 589, 'max_features': 7, 'bootstrap': True}
    regressor = RandomForestRegressor(**RF_kwargs)

    # # Parameters before
    # regressor.fit(X_train, y_train)
    # outResults1 = regressor.predict(X_test)
    # score = explained_variance_score(y_test, outResults1)
    # print(score)  # 得到预测结果区间[0,1]

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
    # plt.scatter(Y, outResults)
    # plt.show()
    score = explained_variance_score(Y, outResults)
    print('Allset score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]

    print('Time Using:%.2f min' % ((time()-t1)/60),end=' / ')

    # print(regressor.get_params().values())
    # print(regressor.best_params_,end='\n\n')
    # print(regressor.best_estimator_)

    # Out PIC
    # importances = regressor.feature_importances_
    #
    # read and predict
    for yr in range(2017,2018):
        print(yr)
        '''Data Tif List'''
        # staticPath
        dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(para_Output[1].keys()),list(para_Output[1].values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
        dymtif = sum(dymtif, [])    # 解决List嵌套

        yrTifList = para_Output[0]+dymtif
        dataDf, im_proj, im_geotrans = EsRaster.read_tifList(yrTifList)

        # Block split
        xid = dataDf[(dataDf.iloc[:, 3] > -1) & (dataDf.iloc[:, 4] > 0) & (dataDf.iloc[:, 4] < 100) & (dataDf.iloc[:, 5] > -9999) & (dataDf.iloc[:, 11] >= 0) & (dataDf.iloc[:, 12] > -9999) & (dataDf.iloc[:, 23] > 0)].index.tolist()
        indf = dataDf.iloc[xid, :]
        indf = np.array_split(indf,200)
        estimators = [regressor for i in range(200)]

        '''Predict'''
        kwgs = list(zip(estimators, indf))
        outResults_ = run_imap_mp(EsRaster.generate_mulcpu_vars_Predict,kwgs,num_processes=20, is_tqdm=True)
        outResults_ = np.hstack(tuple(outResults_))
        outResults_ = outResults_.reshape([-1, 1])
        outdat = np.zeros((4998 * 4088, 1)) - 9999
        outdat[xid] = outResults_
        outdat = outdat.reshape(4088,4998)

        EsRaster.write_img(r"G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB"+os.sep+r'RF_AGB_'+str(yr)+r'.tif', im_proj, im_geotrans, outdat)
    print()

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


    # outResults3 = regressor.predict(X)
    # plt.scatter(Y, outResults3)
    # plt.show()
    # linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults3, dtype=np.float64))
    # print((linreg.rvalue) ** 2)

    # a = pd.DataFrame()
    # a['预测值'] = list(outResults1)
    # a['实际值'] = list(y_test)
    #score = accuracy_score(outResults1, y_test)

    # 参数优化
    # param_distribs = {
    #     # 均匀离散随机变量
    #     'n_estimators': randint(low=300, high=700),
    #     'max_features': randint(low=7, high=20),        # 寻找最佳分割时要考虑的特征数量
    #     'max_leaf_nodes': randint(low=1, high=10),
    #     'max_depth': randint(low=5, high=15),           # 树的最大深度
    #     'min_samples_leaf': randint(low=1, high=10),    # 在叶节点处需要的最小样本数
    #     #'min_samples_split': randint(low=1, high=10),   # 拆分内部节点所需的最少样本数
    #     "bootstrap": [True, False],
    # }



    # # 参数优化——后
    # tree_reg.fit(X_train, y_train)
    # # 返回最优的训练器
    # print(tree_reg.best_estimator_)
    # outResults2 = tree_reg.predict(X_test)
    #
    # # =================================
    # plt.scatter(y_test, outResults2)
    # plt.show()
    # linreg = st.linregress(pd.Series(y_test, dtype=np.float64), pd.Series(outResults2, dtype=np.float64))
    # print((linreg.rvalue)**2)
    #
    # outResults3 = tree_reg.predict(X)
    # plt.scatter(Y, outResults3)
    # plt.show()
    # linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults3, dtype=np.float64))
    # print((linreg.rvalue) ** 2)
    #
    #
    # a = pd.DataFrame()
    # a['预测值'] = list(outResults2)
    # a['实际值'] = list(y_test)
    # score = accuracy_score(outResults2, y_test)
    # y_pred_proba = tree_reg.predict_proba(X_test)
    # b = pd.DataFrame(y_pred_proba, columns=['label=0', 'label=1'])
    # print('得分：', score)
    # print()



    # 执行一次
    # os.environ['PATH'] = os.environ['PATH']+';'+r"D:\CLibrary\Graphviz2.44.1\bin\graphviz"
    # dot_data = StringIO()
    # export_graphviz(pipe.named_steps['regressor'].estimators_[0],
    #                 out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('tree.png')
    # Image(graph.create_png())



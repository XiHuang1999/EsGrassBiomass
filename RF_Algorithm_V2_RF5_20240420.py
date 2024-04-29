# -*- coding: utf-8 -*-

# @Time : 2022-07-01 16:30
# @Author : XiHuang O.Y.
# @Site : 
# @File : RF_Algorithm.py
# @Software: PyCharm
'''纠正了样本点的重量单位（一致的单位：gDM/m-2）'''
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from time import time
import time as tm
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import scipy.stats as st
import random
from itertools import product
import pickle

from six import StringIO
# from IPython.display import Image
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

    import os
    os.environ['PROJ_LIB'] = r'I:\代码\PyCharmPythonFiles\PythonFiles_IGSNRR5\Lib\site-packages\osgeo\data\proj' #r"E:\Anaconda\Anaconda\Lib\site-packages\osgeo\data\proj"
    print(r"Start RF Predict AGB:")
    t1 = time()

    X_columns = list(X.columns)
    X = np.array(X)
    Y = np.array(Y)

    # 将数据分为训练样本和验证样本
    # random.seed(999)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1-train_size,
                                                        # shuffle=True,
                                                        random_state=99)
    # lii = []
    # li = []
    # for i in range(288,900,2):
    #     for ii in range(1,25):
    #         print('==> '+str(i)+' + '+str(ii))
    #         RF_kwargs = {'n_estimators': i,  'bootstrap': True, "random_state": 99, #'max_features': 3,
    #                      'min_samples_split': 10}
    #         regressor = RandomForestRegressor(**RF_kwargs)
    #         regressor.fit(X_train, y_train)
    #         outResults1 = regressor.predict(X_train)
    #         linreg1 = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
    #         print('Training score:%.4f' % linreg1.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]
    #
    #         outResults2 = regressor.predict(X_test)
    #         linreg2 = st.linregress(pd.Series(y_test, dtype=np.float64), pd.Series(outResults2, dtype=np.float64))
    #         print('Testing score:%.4f' % linreg2.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]
    #
    #         outResults = regressor.predict(X)
    #         linreg3 = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
    #         print('Allsite score:%.4f' % linreg3.rvalue ** 2, end=' /\n ')  # 得到预测结果区间[0,1]
    #
    #         lii.append([i,ii])
    #         li.append([linreg1.rvalue ** 2,linreg2.rvalue ** 2,linreg3.rvalue ** 2])
    # outexcel = pd.concat([pd.DataFrame(li), pd.DataFrame(lii)], axis=1)
    # outexcel.to_excel(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_2024\DataTable\GridCV.xlsx')



    # l1 = list(range(30,630,1)) #64 list(outexcel2.iloc[0:80,4])
    # l1 = np.array_split(l1, 200)
    # l2 = l1

    # l2 = list(range(1,71))
    # l1 = [64 for i in range(2,71)]
    # l3 = [2 for i in range(2,71)]
    # l1 = np.array_split(l1, 35)
    # l2 = np.array_split(l2, 35)
    # l3 = np.array_split(l2, 35)


    # l2 = list(range(1,71))
    # l1 = [i for i in range(30,130,20)]
    # temp = pd.DataFrame(list(product(l1, l2)))
    # l1 = np.array_split(temp.iloc[:, 0], 35)
    # l2 = np.array_split(temp.iloc[:, 1], 35)
    # l3 = l2

    # l1 = [64, 66, 65, 59, 60, 58, 63, 232, 234, 233, 241, 238, 52, 62, 236, 239, 240, 235, 230, 67, 231, 237, 242, 229, 228, 68, 226, 243, 252, 253, 51, 227, 225, 254, 61, 57, 244, 56, 272, 245, 223, 69, 267, 224, 255, 271, 269, 217, 268, 266, 216, 251, 246, 247, 250, 249, 270, 256, 213, 212, 248, 215, 265, 258, 257, 219, 264, 275, 55, 218, 273, 222, 259, 46, 220, 261, 274, 221, 263, 260]
    # l2 = list(range(1,35,2))
    # l3 = list(range(4,35,2))
    # temp = pd.DataFrame(list(product(l1, l2, l3)))
    # l1 = np.array_split(temp.iloc[:, 0], 200)
    # l2 = np.array_split(temp.iloc[:, 1], 200)
    # l3 = np.array_split(temp.iloc[:, 2], 200)
    #
    # X_train_ = [X_train for i in range(200)]
    # y_train_ = [y_train for i in range(200)]
    # X_test_ = [X_test for i in range(200)]
    # y_test_ = [y_test for i in range(200)]
    # X_ = [X for i in range(200)]
    # Y_ = [Y for i in range(200)]
    # kwgs = list(zip(X_train_, y_train_, X_test_, y_test_, X_, Y_, l1, l2, l3))
    #
    # outexcel = run_imap_mp(generate_mulcpu_vars_Est, kwgs, num_processes=30, is_tqdm=True)
    # outexcel = sum(outexcel,[])
    # save_variable(outexcel, r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_2024\DataTable\outexcel_GridCV---n_estimators-max_depth-max_features.pickle') #---n_estimators-max_features2
    # # outexcel = load_variavle(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_2024\DataTable\outexcel_GridCV---n_estimators-max_depth-max_features.pickle')
    # outexcel1 = pd.DataFrame(outexcel)
    # outexcel2 = outexcel1.sort_values(by=1, ascending=False)
    # # 打印最高的交叉验证准确度及其索引
    # # list.index([object]) 返回这个object在列表list中的索引
    # scorel = list(outexcel1.iloc[:,1])
    # print(max(scorel), (scorel.index(max(scorel))))
    #
    # plt.figure(figsize=[10, 5])
    # plt.plot(list(outexcel1.iloc[:,4]), scorel)
    # plt.show()


    RF_kwargs = {'n_estimators': 63, 'bootstrap': True, "random_state": 99, 'max_depth':11, 'max_features': 5,
                 'min_samples_leaf':1, 'min_samples_split':2  }
                 # 'criterion':'squared_error'}
    # # RF_kwargs = {'n_estimators': 603, 'max_features': 7, 'bootstrap': True,
    # #              'min_samples_split':4 ,'oob_score':True,'max_samples':0.985}
    #              #'criterion':'squared_error'}
    # RF_kwargs = {'n_estimators': 699, 'max_features': 7, 'bootstrap': True,
    #              'min_samples_split':4 ,'oob_score':True,'max_samples':0.985}

    # RF_kwargs = {'n_estimators': 566, 'max_features': 3, 'bootstrap': True,
    #              'max_depth':3, 'min_samples_split':0.1 , 'min_samples_leaf':5,
    #              'oob_score':True,'max_samples':0.985}
    # # RF_kwargs = {'n_estimators': 561, 'max_features': 5, 'bootstrap': True}
    # # # #{'n_estimators': 538, 'max_leaf_nodes': 10, 'max_features': 19, 'bootstrap': True}#
    # #             # {'n_estimators': 589, 'max_features': 7, 'bootstrap': True}
    regressor = RandomForestRegressor(**RF_kwargs)

    # # Parameters before
    # regressor.fit(X_train, y_train)
    # outResults1 = regressor.predict(X_test)
    # score = explained_variance_score(y_test, outResults1)
    # print(score)  # 得到预测结果区间[0,1]

    # # # Parameters 1
    # # regressor = RandomForestRegressor(random_state=99,bootstrap=True)
    # # param_distribs = {
    # #     # 均匀离散随机变量
    # #     'n_estimators': [int(pi) for pi in np.linspace(300,900,300)], #[926]
    # #     'max_features': [int(pi) for pi in np.linspace(1,20,20)],  # 寻找最佳分割时要考虑的特征数量
    # #     # 'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=15,num=7)],
    # #     # 'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=10)],           # 树的最大深度
    # #     # 'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=20,num=10)],    # 在叶节点处需要的最小样本数
    # #     # 'min_samples_split': [int(x) for x in np.linspace(start=2,stop=20,num=10)],   # 拆分内部节点所需的最少样本数
    # #     # 'max_samples': [0.82]
    # # }
    # # regressor = GridSearchCV(regressor, param_grid=param_distribs, n_jobs=10, pre_dispatch=20, cv=3, verbose=10,
    # #                          scoring=r'r2')

    # # # Parameters 2
    # # param_distribs = {
    # #     # 均匀离散随机变量
    # #     'n_estimators': [int(x) for x in np.linspace(start=480, stop=600, num=100)],
    # #     'max_features': [int(x) for x in np.linspace(start=1, stop=20, num=20)],  # 寻找最佳分割时要考虑的特征数量
    # #     'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=10,num=8)],
    # #     #'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=20)],           # 树的最大深度
    # #     #'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=12,num=10)],    # 在叶节点处需要的最小样本数
    # #     #'min_samples_split': [int(x) for x in np.linspace(start=2,stop=10,num=8)],   # 拆分内部节点所需的最少样本数
    # #     "bootstrap": [True, False]
    # # }
    # # regressor = RandomizedSearchCV(regressor, param_distributions=param_distribs,
    # #                               n_iter=31500, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    # print(regressor.best_params_)
    # print(regressor.best_estimator_)
    # print(regressor.best_score_)
    # importances = pd.DataFrame(regressor.feature_importances_,index=X_columns)

    # Parameters 2



    # # # Regression
    regressor.fit(X_train, y_train)

    outResults1 = regressor.predict(X_train)
    linreg = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
    print('Training score:%.4f' % linreg.rvalue**2,end=' / ')  # 得到预测结果区间[0,1]

    outResults2 = regressor.predict(X_test)
    linreg = st.linregress(pd.Series(y_test, dtype=np.float64), pd.Series(outResults2, dtype=np.float64))
    print('Testing score:%.4f' % linreg.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]

    outResults = regressor.predict(X)
    linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
    print('Allsite score:%.4f' % linreg.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]


    # region Drawing
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    xy = np.vstack([Y, outResults])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
    pltx = [int(x) for x in np.linspace(start=0, stop=max(Y), num=1000)]
    plty = [linreg.slope * x + linreg.intercept for x in pltx]
    f, ax = plt.subplots(figsize=(14/3/2.54+2, 14/3/2.54+2))
    plt.plot(pltx, plty, '-', color='red', alpha=0.8, linewidth=2, label='Fitting Line')    # color='#4169E1'
    plt.plot(pltx, pltx, '-', color='black', alpha=0.8, linewidth=2, label='1:1')
    plt.scatter(Y, outResults,c=z,s=1.3,cmap='Spectral')
    plt.text(50,320,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
             '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
             '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
             ,fontsize=12,color = "black",fontweight='bold')
    plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    plt.xlabel('Literature and field based ANPP value')  # 添加x轴和y轴标签
    plt.ylabel('Machine-learning based ANPP value')
    # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'RF_results.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results' + os.sep + r'PIC' + os.sep + 'RF_results_test2024-2.pdf',
    #             dpi=600, bbox_inches='tight',format="pdf")  # , transparent=True
    plt.show()

    # xy = np.vstack([y_train, outResults1])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # linreg = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
    # pltx = [int(x) for x in np.linspace(start=0, stop=max(Y), num=1000)]
    # plty = [linreg.slope * x + linreg.intercept for x in pltx]
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.scatter(y_train, outResults1, c=z, s=1.8,cmap='Spectral', label='Training')
    # plt.scatter(y_test, outResults2, s=1, c='Black', label='Validation')
    # plt.plot(pltx, plty, '-', color='red', alpha=0.8, linewidth=2, label='Fitting Line')    # color='#4169E1'
    # plt.plot(pltx, pltx, '--', color='black', alpha=0.8, linewidth=2, label='1:1')
    # plt.text(290,200,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
    #          '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
    #          '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
    #          ,fontsize=8,color = "r",fontweight='bold')
    # plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    # plt.xlabel('Observed AGB')  # 添加x轴和y轴标签
    # plt.ylabel('Estimated AGB')
    # plt.legend()
    # # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'RF4_results.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.show()

    # xy = np.vstack([y_train, outResults1])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # linreg = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
    # pltx = [x for x in np.linspace(start=0, stop=max(y_train), num=1000)]
    # plty = [linreg.slope * x + linreg.intercept for x in pltx]
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(pltx, plty, '-', color='red', alpha=0.8, linewidth=2, label='Fitting Line')    # color='#4169E1'
    # plt.plot(pltx, pltx, '-', color='black', alpha=0.8, linewidth=2, label='1:1')
    # plt.scatter(y_train, outResults1,c=z,s=1.3,cmap='Spectral')
    # plt.text(400,235,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
    #          '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
    #          '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
    #          ,fontsize=8,color = "r",fontweight='bold')
    # plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    # plt.xlabel('AGB Site Value')  # 添加x轴和y轴标签
    # plt.ylabel('Model Value')
    # # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'RF_results.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.show()
    # endregion

    print('Time Using:%.2f min' % ((time()-t1)/60),end=' / ')

    # did = np.where((xy[0,:]>630)&(xy[1,:]<322))
    # did = np.where((xy[0,:]>265)&(xy[1,:]<185))
    # did = np.where((xy[0,:]>638)&(xy[1,:]<397))
    # xy = np.vstack([Y, outResults])
    # np.where((xy[0, :] > 586) & (xy[1, :] < 429))

    print("  ",end='\n\n')
    # print(regressor.get_params(),end='\n\n')
    # print(regressor.get_params().values(),end='\n\n')
    # print(regressor.best_params_,end='\n\n')
    # print(regressor.best_estimator_)

    # Out PIC
    # importances = regressor.feature_importances_

    # 获取特征重要性得分
    feature_importances = regressor.feature_importances_
    # 创建特征名列表
    feature_names = X_columns
    # 创建一个DataFrame，包含特征名和其重要性得分
    feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    # 对特征重要性得分进行排序
    feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    # 可视化特征重要性
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
    ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
    ax.set_xlabel('特征重要性', fontsize=12)  # 图形的x标签
    ax.set_title('随机森林特征重要性可视化', fontsize=16)
    for i, v in enumerate(feature_importances_df['importance']):
        ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)
    # # 设置图形样式
    # plt.style.use('default')
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    # ax.spines['left'].set_linewidth(0.5)#左边框粗细
    # ax.spines['bottom'].set_linewidth(0.5)#下边框粗细
    # ax.tick_params(width=0.5)
    # ax.set_facecolor('white')#背景色为白色
    # ax.grid(False)#关闭内部网格线
    # 保存图形
    plt.show()

    # read and predict
    para_Output[0]=[r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\LAT_China1km.tif',
                    r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\Clay.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\CoarseSand.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\FineSand.tif',
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\OrganicMass.tif',
                    r"G:\1_BeiJingUP\AUGB\Data\20220629\Soil\PH_H2O.tif",
                     r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\PowderedSand.tif']
    for yr in range(2000,2021):
        print(yr)
        '''Data Tif List'''
        # staticPath
        dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(para_Output[1].keys()),list(para_Output[1].values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
        dymtif = sum(dymtif, [])    # 解决List嵌套
        yrTifList = para_Output[0]+dymtif
        dataDf, im_proj, im_geotrans = EsRaster.read_tifList(yrTifList)

        # Block split  pd.DataFrame(np.array(dataDf.iloc[:, 3]).reshape([4998,4088])).plot()
        # clay>-1        0<cgrass<100      NDVI>=0              TAVG>-9999               fpar>0
        xid = dataDf[(dataDf.iloc[:, 2] > -1) & (dataDf.iloc[:, 3] > -9999) & (dataDf.iloc[:, 8] >= 0) &
                     (dataDf.iloc[:, 9] > -9999) & (dataDf.iloc[:, 18] > 0)].index.tolist()
        indf = dataDf.iloc[xid, :]
        indf.columns = X_columns#[X_columns[0]] + X_columns[2:]
        # indf = indf.reindex(columns=X_columns, fill_value=yr)
        indf = np.array(indf)
        indf = np.array_split(indf,200)
        estimators = [regressor for i in range(200)]

        '''Predict'''
        kwgs = list(zip(estimators, indf))
        outResults_ = run_imap_mp(EsRaster.generate_mulcpu_vars_Predict,kwgs,num_processes=30, is_tqdm=True)
        # print(outResults_)
        outResults_ = np.hstack(tuple(outResults_))
        # print(outResults_)
        outResults_ = outResults_.reshape([-1, 1])
        outdat = np.zeros((4998 * 4088, 1)) #- 9999
        outdat[xid] = outResults_
        outdat = outdat.reshape(4088,4998)

        EsRaster.write_img(r"G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_2024\全区域"+os.sep+r'RF_AGB_'+str(yr)+r'.tif', im_proj, im_geotrans, outdat)
        del outResults_,dataDf,indf
        tm.sleep(3)
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



def generate_mulcpu_vars_GridCV(args):
    '''
    并行多参数调用
    :param args: 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    :return:
    '''
    return predict_Block_GridCV(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7])


def predict_Block_GridCV(X_train, y_train, X_test, y_test, X, Y, i, ii):
    '''
    Predict by block
    :param est: obj, estimator
    :param dataX: ndarray, X var after np.split(a, 2, axis=1)
    :return: estimator result
    '''

    # print(X_train.shape)
    li = []
    for ri,rii in zip(i,ii):
        RF_kwargs = {'n_estimators': ri, 'max_features': rii, 'bootstrap': True, "random_state": 99}

        regressor = RandomForestRegressor(**RF_kwargs)
        regressor.fit(X_train, y_train)
        outResults1 = regressor.predict(X_train)
        linreg1 = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
        # print('Training score:%.4f' % linreg1.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]

        outResults2 = regressor.predict(X_test)
        linreg2 = st.linregress(pd.Series(y_test, dtype=np.float64), pd.Series(outResults2, dtype=np.float64))
        # print('Testing score:%.4f' % linreg2.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]

        outResults = regressor.predict(X)
        linreg3 = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
        # print('Allsite score:%.4f' % linreg3.rvalue ** 2, end=' /\n ')  # 得到预测结果区间[0,1]

        li.append([linreg1.rvalue ** 2,linreg2.rvalue ** 2,linreg3.rvalue ** 2, ri, rii])

    return li


def generate_mulcpu_vars_Est(args):
    '''
    并行多参数调用
    :param args: 读入了两个变量，需要计算的wins下标，以及Manager Namespace
    :return:
    '''
    return predict_Block_Est(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[7])

def predict_Block_Est(X_train, y_train, X_test, y_test, X, Y, i, ii, iii):
    '''
    Predict by block
    :param est: obj, estimator
    :param dataX: ndarray, X var after np.split(a, 2, axis=1)
    :return: estimator result
    '''
    li = []
    for ri,rii,riii in zip(i,ii,iii):
        # RF_kwargs = {'n_estimators': 210, 'bootstrap': True, "random_state": 99, 'max_features': 7,
        #              'max_depth':25, 'min_samples_leaf':ri}  #, 'min_samples_split':rii
        RF_kwargs = {'n_estimators': ri, 'bootstrap': True, "random_state": 99, 'max_depth':rii, 'max_features': riii }  #, 'min_samples_split':rii , 'min_samples_leaf':ri
        # RF_kwargs = {'n_estimators': ri, 'bootstrap': True, "random_state": 99}  #, 'min_samples_split':rii

        regressor = RandomForestRegressor(**RF_kwargs)

        score = cross_val_score(regressor, X_test, y_test, cv=10, scoring ='r2').mean()  # 交叉验证

        regressor.fit(X_train, y_train)

        outResults1 = regressor.predict(X_train)
        linreg1 = st.linregress(pd.Series(y_train, dtype=np.float64), pd.Series(outResults1, dtype=np.float64))
        # print('Training score:%.4f' % linreg1.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]

        outResults2 = regressor.predict(X_test)
        linreg2 = st.linregress(pd.Series(y_test, dtype=np.float64), pd.Series(outResults2, dtype=np.float64))
        # print('Testing score:%.4f' % linreg2.rvalue ** 2, end=' / ')  # 得到预测结果区间[0,1]

        outResults = regressor.predict(X)
        linreg3 = st.linregress(pd.Series(Y, dtype=np.float64), pd.Series(outResults, dtype=np.float64))
        # print('Allsite score:%.4f' % linreg3.rvalue ** 2, end=' /\n ')  # 得到预测结果区间[0,1]

        li.append([linreg1.rvalue ** 2,linreg2.rvalue ** 2,linreg3.rvalue ** 2, ri, rii, score])

    return li

def save_variable(v, filename):
    f = open(filename, 'wb')  # 打开或创建名叫filename的文档。
    pickle.dump(v, f)  # 在文件filename中写入v
    f.close()  # 关闭文件，释放内存。
    return filename


def load_variavle(filename):
    try:
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r

    except EOFError:
        return ""
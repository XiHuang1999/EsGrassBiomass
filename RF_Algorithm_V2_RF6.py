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
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from time import time
import time as tm
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as st
import random
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

    import os
    os.environ['PROJ_LIB'] = r"E:\Anaconda\Anaconda\Lib\site-packages\osgeo\data\proj"
    print(r"Start RF Predict AGB:")
    t1 = time()

    X_columns = list(X.columns)
    X = np.array(X)
    Y = np.array(Y)

    # 将数据分为训练样本和验证样本
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1-train_size,
                                                        random_state=0)
    regressor = RandomForestRegressor()
    # RF_kwargs = {'n_estimators': 699, 'max_features': 7, 'bootstrap': True,
    #              'min_samples_split':4 ,'oob_score':True,'max_samples':0.985}
    #              # 'criterion':'squared_error'}
    # # RF_kwargs = {'n_estimators': 566, 'max_features': 16, 'bootstrap': True}
    # # RF_kwargs = {'n_estimators': 561, 'max_features': 5, 'bootstrap': True}
    # # #{'n_estimators': 538, 'max_leaf_nodes': 10, 'max_features': 19, 'bootstrap': True}#
    #             # {'n_estimators': 589, 'max_features': 7, 'bootstrap': True}
    # regressor = RandomForestRegressor(**RF_kwargs)

    # # Parameters before
    # regressor.fit(X_train, y_train)
    # outResults1 = regressor.predict(X_test)
    # score = explained_variance_score(y_test, outResults1)
    # print(score)  # 得到预测结果区间[0,1]

    # # Parameters 1
    # param_distribs = {
    #     # 均匀离散随机变量
    #     'n_estimators': [int(x) for x in np.linspace(start=480, stop=600, num=100)],
    #     'max_features': [int(x) for x in np.linspace(start=1, stop=20, num=20)],  # 寻找最佳分割时要考虑的特征数量
    #     'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=10,num=8)],
    #     #'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=20)],           # 树的最大深度
    #     #'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=12,num=10)],    # 在叶节点处需要的最小样本数
    #     #'min_samples_split': [int(x) for x in np.linspace(start=2,stop=10,num=8)],   # 拆分内部节点所需的最少样本数
    #     "bootstrap": [True, False]
    # }
    # regressor = RandomizedSearchCV(regressor, param_distributions=param_distribs,
    #                               n_iter=31500, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # # Parameters 2
    param_distribs = {
        # 均匀离散随机变量
        'n_estimators': [int(x) for x in np.linspace(start=700, stop=800, num=100)],
        'max_features': [int(x) for x in np.linspace(start=1, stop=60, num=30)]  # 寻找最佳分割时要考虑的特征数量
        #'max_leaf_nodes': [int(x) for x in np.linspace(start=2,stop=10,num=8)],
        #'max_depth': [int(x) for x in np.linspace(start=1,stop=20,num=20)],           # 树的最大深度
        #'min_samples_leaf': [int(x) for x in np.linspace(start=1,stop=12,num=10)],    # 在叶节点处需要的最小样本数
        #'min_samples_split': [int(x) for x in np.linspace(start=2,stop=10,num=8)],   # 拆分内部节点所需的最少样本数
        # "bootstrap": [True, False]
    }

    # dict_values(
    #     [5, nan, True, 0.0, 'squared_error', None, 1.0, None, None, 0.0, 1, 2, 0.0, 100, None, False, None, 0, False,
    #      RandomForestRegressor(), 10, {
    #          'n_estimators': [300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334,
    #                           336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370,
    #                           372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406,
    #                           408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442,
    #                           444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478,
    #                           480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 501, 503, 505, 507, 509, 511, 513, 515,
    #                           517, 519, 521, 523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545, 547, 549, 551,
    #                           553, 555, 557, 559, 561, 563, 565, 567, 569, 571, 573, 575, 577, 579, 581, 583, 585, 587,
    #                           589, 591, 593, 595, 597, 599, 601, 603, 605, 607, 609, 611, 613, 615, 617, 619, 621, 623,
    #                           625, 627, 629, 631, 633, 635, 637, 639, 641, 643, 645, 647, 649, 651, 653, 655, 657, 659,
    #                           661, 663, 665, 667, 669, 671, 673, 675, 677, 679, 681, 683, 685, 687, 689, 691, 693, 695,
    #                           697, 700], 'max_features': [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]}, 16, True, False, 'r2',
    #      300])
    # {'max_features': 1, 'n_estimators': 308}
    #
    # RandomForestRegressor(max_features=1, n_estimators=308)
    # Training
    # score: 0.9449 / Testing
    # score: 0.4087 / Allsite
    # score: 0.9007 / Time
    # Using: 270.36 min

    # dict_values(
    #     [5, nan, True, 0.0, 'squared_error', None, 1.0, None, None, 0.0, 1, 2, 0.0, 100, None, False, None, 0, False,
    #      RandomForestRegressor(), 10, {
    #          'n_estimators': [301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323, 325, 327, 329, 331, 333, 335,
    #                           337, 339, 341, 343, 345, 347, 349, 351, 353, 355, 357, 359, 361, 363, 365, 367, 369, 371,
    #                           373, 375, 377, 379, 381, 383, 385, 387, 389, 391, 393, 395, 397, 399, 401, 403, 405, 407,
    #                           409, 411, 413, 415, 417, 419, 421, 423, 425, 427, 429, 431, 433, 435, 437, 439, 441, 443,
    #                           445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479,
    #                           481, 483, 485, 487, 489, 491, 493, 495, 497, 499, 502, 504, 506, 508, 510, 512, 514, 516,
    #                           518, 520, 522, 524, 526, 528, 530, 532, 534, 536, 538, 540, 542, 544, 546, 548, 550, 552,
    #                           554, 556, 558, 560, 562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588,
    #                           590, 592, 594, 596, 598, 600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622, 624,
    #                           626, 628, 630, 632, 634, 636, 638, 640, 642, 644, 646, 648, 650, 652, 654, 656, 658, 660,
    #                           662, 664, 666, 668, 670, 672, 674, 676, 678, 680, 682, 684, 686, 688, 690, 692, 694, 696,
    #                           698, 701], 'max_features': [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]}, 16, True, False, 'r2',
    #      300])
    # {'max_features': 1, 'n_estimators': 576}
    #
    # RandomForestRegressor(max_features=1, n_estimators=576)
    # Training
    # score: 0.9465 / Testing
    # score: 0.3951 / Allsite
    # score: 0.9001 / Time
    # Using: 265.99
    # min /

    # dict_values(
    #     [5, nan, True, 0.0, 'squared_error', None, 1.0, None, None, 0.0, 1, 2, 0.0, 100, None, False, None, 0, False,
    #      RandomForestRegressor(), 10, {
    #          'n_estimators': [700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717,
    #                           718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735,
    #                           736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753,
    #                           754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771,
    #                           772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789,
    #                           790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807,
    #                           808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825,
    #                           826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843,
    #                           844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861,
    #                           862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879,
    #                           880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897,
    #                           898, 900]}, 16, True, False, 'r2', 300])
    # {'n_estimators': 757}
    #
    # RandomForestRegressor(n_estimators=757)
    # Training
    # score: 0.9392 / Testing
    # score: 0.4059 / Allsite
    # score: 0.8921 / Time
    # Using: 70.59
    # min /


    regressor = GridSearchCV(regressor, param_grid=param_distribs, cv=5, n_jobs=10,
                             pre_dispatch=20, verbose=1, scoring='r2') #

    # Regression
    random.seed(0)
    regressor.fit(X_train, y_train)

    print(regressor.get_params().values())
    print(regressor.best_params_,end='\n\n')
    print(regressor.best_estimator_)

    # importances = pd.DataFrame(regressor.feature_importances_,index=X_columns)

    # outResults1 = regressor.predict(X_train)
    # score = explained_variance_score(y_train, outResults1)
    # print('Training score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]
    #
    # outResults2 = regressor.predict(X_test)
    # score = explained_variance_score(y_test, outResults2)
    # print('Test score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]
    #
    # outResults = regressor.predict(X)
    # score = explained_variance_score(Y, outResults)
    # print('Allset score:%.4f' % score,end=' / ')  # 得到预测结果区间[0,1]


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
    # plt.text(400,1500,r'y = '+str('%.2f' % linreg.slope)+r'*x + '+str('%.2f' % linreg.intercept)+
    #          '\n'+r'R Square = '+str('%.2f' % (linreg.rvalue**2))+
    #          '\n'+r'P Value = '+str('%.2f' % linreg.pvalue)
    #          ,fontsize=8,color = "r",fontweight='bold')
    # plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    # plt.xlabel('AGB Site Value')  # 添加x轴和y轴标签
    # plt.ylabel('Model Value')
    # # plt.savefig(r'G:\1_BeiJingUP\AUGB\Data\20220629\Results'+os.sep+r'PIC'+os.sep+'RF_results.png',dpi=500,bbox_inches='tight')#, transparent=True
    # plt.show()

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

    # did = np.where((xy[0,:]>192.7)&(xy[1,:]<145.3))
    # did = np.where((xy[0,:]>265)&(xy[1,:]<185))
    # did = np.where((xy[0,:]<32)&(xy[1,:]>170))
    # xy = np.vstack([Y, outResults])
    # np.where((xy[0, :] > 586) & (xy[1, :] < 429))


    # Out PIC
    # importances = regressor.feature_importances_

    # # read and predict
    # para_Output[0]=[r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\LAT_China1km.tif',
    #                 r'G:\1_BeiJingUP\AUGB\Data\20220629\Parameters\dem_china1km.tif']
    #                 #  r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\Clay.tif',
    #                 #  r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\CoarseSand.tif',
    #                 #  r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\FineSand.tif',
    #                 #  r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\OrganicMass.tif',
    #                 # r"G:\1_BeiJingUP\AUGB\Data\20220629\Soil\PH_H2O.tif",
    #                 #  r'G:\1_BeiJingUP\AUGB\Data\20220629\Soil\PowderedSand.tif']
    # for yr in range(2000,2001):
    #     print(yr)
    #     '''Data Tif List'''
    #     # staticPath
    #     dymtif = [glob(dymPath+os.sep+r'*'+str(yr)+r'*.tif') for dymKey,dymPath in zip(list(para_Output[1].keys()),list(para_Output[1].values()))]    # dymPath+os.sep+dymKey.upper()[:-1]+r'_'+str(yr)+r'.tif'
    #     dymtif = sum(dymtif, [])    # 解决List嵌套
    #     yrTifList = para_Output[0]+dymtif
    #     dataDf, im_proj, im_geotrans = EsRaster.read_tifList(yrTifList)
    #
    #     # Block split
    #     # clay>-1        0<cgrass<100      NDVI>=0              prcp>-9999               fpar>0
    #     xid = dataDf[(dataDf.iloc[:, 2] >= 0) &
    #                  (dataDf.iloc[:, 6] > -9999) & (dataDf.iloc[:, 13] >= 0)].index.tolist()
    #     # dataDf.insert(loc=1, column='Year', value=[yr for yri in range(dataDf.shape[0])])
    #
    #     indf = dataDf.iloc[xid, :]
    #     indf.columns = X_columns#[X_columns[0]] + X_columns[2:]
    #     # indf = indf.reindex(columns=X_columns, fill_value=yr)
    #     indf = np.array(indf)
    #     indf = np.array_split(indf,200)
    #     estimators = [regressor for i in range(200)]
    #
    #     '''Predict'''
    #     kwgs = list(zip(estimators, indf))
    #     outResults_ = run_imap_mp(EsRaster.generate_mulcpu_vars_Predict,kwgs,num_processes=30, is_tqdm=True)
    #     outResults_ = np.hstack(tuple(outResults_))
    #     outResults_ = outResults_.reshape([-1, 1])
    #     outdat = np.zeros((4998 * 4088, 1)) - 9999
    #     outdat[xid] = outResults_
    #     outdat = outdat.reshape(4088,4998)
    #
    #     EsRaster.write_img(r"G:\1_BeiJingUP\AUGB\Data\20220629\Results\RF_AGB_6不放Soil"+os.sep+r'RF_AGB_'+str(yr)+r'.tif', im_proj, im_geotrans, outdat)
    #     del outResults_,dataDf,indf
    #     tm.sleep(1)
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

    #




# -*- coding: utf-8 -*-

# @Time : 2022-07-01 16:30
# @Author : XiHuang O.Y.
# @Site : 
# @File : RF_Algorithm.py
# @Software: PyCharm

import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
# import pydotplus
import os

def RFEstimate(X,
               Y,
               train_size=0.8,
               RF_kwargs={'n_estimators':500,# 树的棵树，默认是100
                          'max_depth':None,
                          'random_state':None, # 控制构建树时样本的随机抽样
                          # 'bootstrap':True,   # 建立树木时是否使用bootstrap抽样
                          # 'max_depth':10,          # 树的最大深度
                          # 'max_features':2,         # 寻找最佳分割时要考虑的特征数量
                          # 'min_samples_leaf':3,   # 在叶节点处需要的最小样本数
                          # 'min_samples_split':5  # 拆分内部节点所需的最少样本数
                }):
    '''
    使用随机森林预测地上生物量
    :param X: 自变量,X = dataset.iloc[:, 0:4].values
    :param Y: 因变量,y = dataset.iloc[:, 4].values
    :param test_size: float,训练样本
    :return:
    '''


    # 将数据分为训练样本和验证样本
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=1-train_size,
                                                        random_state=0)
    # regressor = RandomForestRegressor()
    regressor = RandomForestRegressor(**RF_kwargs)
    # regressor = RandomForestRegressor(random_state=RF_kwargs['random_state'],
    #                              bootstrap=RF_kwargs['bootstrap'],
    #                              max_depth=RF_kwargs['max_depth'],
    #                              max_features=RF_kwargs['max_features'],
    #                              min_samples_leaf=RF_kwargs['min_samples_leaf'],
    #                              min_samples_split=RF_kwargs['min_samples_split'],
    #                              n_estimators=RF_kwargs['n_estimators'])


    regressor.fit(X_train, y_train)
    outResults = regressor.predict(X_test)

    print()



    # 执行一次
    # os.environ['PATH'] = os.environ['PATH']+';'+r"D:\CLibrary\Graphviz2.44.1\bin\graphviz"
    # dot_data = StringIO()
    # export_graphviz(pipe.named_steps['regressor'].estimators_[0],
    #                 out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('tree.png')
    # Image(graph.create_png())



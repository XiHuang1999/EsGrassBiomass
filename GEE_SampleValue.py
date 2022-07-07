# -*- coding: utf-8 -*-

# @Time : 2022-07-05 10:55
# @Author : XiHuang O.Y.
# @Site : 
# @File : GEE_SampleValue.py
# @Software: PyCharm

import os, ee, geemap, webbrowser
# method 1: 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
# method 2: 设置代理
geemap.set_proxy(port=10809)
ee.Initialize()

Map = geemap.Map(center=(40, -100), zoom=4)
Map
#  利用map.save 将map保存为html："Landsat 8 images.html"
Map.save("MapViewer.html")
#  webbrowser中打开"Landsat 8 images.html"
webbrowser.open("MapViewer.html")


'''========================================='''
import ee,os,json
import geemap
# method 1: 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'
# # method 2: 设置代理
# geemap.set_proxy(port=10809)
ee.Initialize()
Map = geemap.Map(center=(37, 102), zoom=4)

import ee,os,json
import geemap
# method 1: 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'
# # method 2: 设置代理
# geemap.set_proxy(port=10809)
ee.Initialize()
Map = geemap.Map(center=(37, 102), zoom=4)
​
'''===========Set Parameters==========='''
qtp_file = r'G:\1_BeiJingUP\AUGB\Data\20220629\Shp\QTP_WGS84.shp' # boundary shp
pt_file = r'G:\1_BeiJingUP\AUGB\Temp\20102.shp' # boundary shp
qtp = ee.FeatureCollection(r'projects/ucas-agb/assets/QTP_WGS84')
# pt = geemap.shp_to_ee(pt_shp)
​
'''===========Set Function==========='''
def clip_roi(image):
    '''
    Image裁剪
    :param image: Image,image of needed to clip
    :return: image after clip by qtp
    '''
    clip_img = image.clip(qtp)
    return clip_img
def jsonPrint(data):
    '''
    格式化输出JSON对象
    :param data: JSON object;
    :return: None
    '''
    print json.dumps(data, sort_keys=True, indent=2) # 排序并且缩进两个字符输出

# MCD15A3H.061: Leaf Area Index/FPAR 4-Day Global 500m ; From 2002-07-04  '''MODIS/061/MCD15A3H'''
# MOD15A2H.061: Terra Leaf Area Index/FPAR 8-Day Global 500m MODIS/061/MOD15A2H; From 2002-07-04 '''MODIS/061/MOD15A2H'''
collection = ee.ImageCollection("MODIS/061/MOD15A2H")\
                .filterDate('2010-05-01', '2010-09-30')\
                .map(clip_roi) #.filterBounds(bd)
​
​
colorizedVis = {'min': 0,'max': 100,
                'palette': ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
                          '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
                          '012E01', '011D01', '011301']}
​
​
Map.addLayer(bd, {}, 'QTP Boundary')
# Map.addLayer(pt, {}, 'ROI Point')
Map.addLayer(collection, {}, 'MOD15A2H')
Map

# -*- coding: utf-8 -*-

# @Time : 2022-07-24 17:51
# @Author : XiHuang O.Y.
# @Site : 
# @File : PreProcessing_InputParameters.py
# @Software: PyCharm


def Raster_SUM(image):
    '''
    Image裁剪
    :param image: Image,image of needed to clip
    :return: image after clip by qtp
    '''
    clip_img = image.clip(qtp)
    return clip_img
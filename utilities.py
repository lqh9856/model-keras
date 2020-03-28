#!/usr/bin/env python
# coding: utf-8



import numpy as np
from skimage import io, transform, filters
from config import _R_MEAN_, _G_MEAN_, _B_MEAN_


def padding(img, height=480, width=640, channels=3):
#     resize the img without distortion
    original_height = img.shape[0]
    original_width = img.shape[1]
    img_padded = np.zeros(shape=(height, width, channels), dtype=np.uint8)
        
    height_scale = original_height / height
    width_scale = original_width / width
    
    if height_scale > width_scale:
        new_width = np.int(original_width // height_scale)
        img = transform.resize(img, output_shape=(height, new_width)) * 255
        img_padded[:, (width-new_width)//2 : (width-new_width)//2 + new_width] = img
    else:
        new_height = np.int(original_height // width_scale)
        img = transform.resize(img, output_shape=(new_height, width)) * 255
        img_padded[(height-new_height)//2 : (height-new_height)//2 + new_height, :] = img
        
    return img_padded





def preprocess_images(image_paths, height=480, width=640, channels=3):
    imgs = np.zeros((len(image_paths), height, width, channels))
    for i, image_path in enumerate(image_paths):
        org_img = io.imread(image_path)
        if org_img.ndim == 2:
            org_img = org_img[:, :, np.newaxis]
            org_img = np.concatenate((org_img, org_img, org_img),axis=2)
        img_padded = padding(org_img, height, width, channels=3)
        imgs[i] = img_padded
    imgs[:, :, :, 0] -= _R_MEAN_
    imgs[:, :, :, 1] -= _G_MEAN_
    imgs[:, :, :, 2] -= _B_MEAN_
    imgs[:, :, :, ::-1]
    return imgs




def preprocess_maps(map_paths, height=480, width=640, channels=1):
    maps = np.zeros((len(map_paths), height, width, channels))
    for i, map_path in enumerate(map_paths):
        org_map = io.imread(map_path)
        org_map = org_map[:, :, np.newaxis]
        map_padded = padding(org_map, height, width, channels=1)
        maps[i] = map_padded
    maps /= 255.0
    return maps





def postprocess_map(pred, height, width):
    pred_height = pred.shape[0]
    pred_width = pred.shape[1]
        
    height_scale = height / pred_height
    width_scale = width / pred_width
    
    if height_scale > width_scale:
        new_width = np.int(pred_width * height_scale)
        pred = transform.resize(pred, output_shape=(height, new_width))
        s_map = pred[:, (new_width - width)//2 : (new_width - width)//2 + width]
    else:
        new_height = np.int(pred_height * width_scale)
        pred = transform.resize(pred, output_shape=(new_height, width))
        s_map = pred[(new_height - height)//2 : (new_height - height)//2 + height, :]
    
    s_map = filters.gaussian(s_map, sigma=7)
    s_map /= np.max(s_map)
    
    return s_map


#!/usr/bin/env python
# coding: utf-8

_B_MEAN_ = 103.939
_G_MEAN_ = 116.779
_R_MEAN_ = 123.68

img_height = 224
img_width = 224

frame_num = 5

train_steps = 10
train_epoch = 10
val_steps = 10


video_batch_size = 2

model_save_path = 'model-ckpt/'

data_path = 'dataset/'

test_data_path = 'video_test/'

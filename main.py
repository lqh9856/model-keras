#!/usr/bin/env python
# coding: utf-8

# In[4]:


from utilities import *
from config import *
import os
from models import crossLSTM, lr_schedule, kl_divergence, correlation_coefficient 
import numpy as np
import random
import keras 
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from skimage import io
from math import ceil
from skimage import io
import time


# In[ ]:


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        with open('logs/logs.txt', 'a+', encoding='utf-8') as f:
            f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n')
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        with open('logs/logs.txt', 'a+', encoding='utf-8') as f:
            f.write('loss: ' + str(logs.get('loss')) + '\t val_loss: ' + str(logs.get('val_loss')) + '\n')

def generator(phase='train'):
    base_path = data_path + phase + '/'
    videos = [base_path + f for f in os.listdir(base_path) if os.path.isdir(base_path + f)]
    while True:
        Ximgs = np.zeros((video_batch_size, frame_num, img_height, img_width, 3))
        Ymaps = np.zeros((video_batch_size, frame_num, img_height, img_width, 1))
        random.shuffle(videos)
        video_path = videos[0: video_batch_size]
        
        for i in range(video_batch_size):
            imgs = [video_path[i] + '/images/' + f for f in os.listdir(video_path[i] + '/images/') if f.endswith(('.jpg', '.jpeg', '.png'))]
            maps = [video_path[i] + '/maps/' + f for f in os.listdir(video_path[i] + '/maps/') if f.endswith(('.jpg', '.jpeg', '.png'))]
            start = np.random.randint(0, len(imgs)-frame_num)
            X = preprocess_images(imgs[start: start+frame_num], img_height, img_width)
            Y = preprocess_maps(maps[start: start+frame_num], img_height, img_width)
            Ximgs[i, :] = X
            Ymaps[i, :] = Y
        yield Ximgs, [Ymaps, Ymaps]
        
def test_generator(imgs):
    
    start = 0
    while True:
        Ximgs = np.zeros((1, frame_num, img_height, img_width, 3))
        X = preprocess_images(imgs[start: min(start+frame_num, len(imgs))], img_height, img_width)
        Ximgs[0, :] = X
        yield Ximgs
        start = min(frame_num + start, len(imgs))
        
if __name__ == '__main__':
    phase = 'train'
    if phase == 'train':
        stateful = False
        X_input = Input(batch_shape=(None, None, img_height, img_width, 3))
        model = Model(inputs=X_input, outputs=crossLSTM(X_input, stateful))
        model.load_weights('models/model.h5', by_name=True)
        model.compile(Adam(learning_rate=1e-4), loss=[kl_divergence, correlation_coefficient])
        model.fit_generator(generator(phase), train_steps, train_epoch, 
                            validation_data=generator('val'), validation_steps=val_steps, 
                            callbacks=[EarlyStopping(patience=15), 
                                       LearningRateScheduler(schedule=lr_schedule), 
                                       ModelCheckpoint(model_save_path + 'model.{epoch:02d}.{loss:.4f}.h5', save_best_only=False), 
                                       LossHistory()])

    elif phase == 'test':
        stateful = True
        X_input = Input(batch_shape=(1, None, img_height, img_width, 3))
        model = Model(inputs=X_input, outputs=crossLSTM(X_input, stateful))
        model.load_weights('models/model.h5')
        model.summary()
        video_paths = [test_data_path + f for f in os.listdir(test_data_path) if os.path.isdir(test_data_path + f)]
        for i in range(0,len(video_paths)):
            map_path = video_paths[i] + '/saliency/'
            if not os.path.exists(map_path):
                os.mkdir(map_path)
            imgs = [video_paths[i] + '/images/' + f for f in os.listdir(video_paths[i] + '/images/') if f.endswith(('.jpg', '.jpeg', '.png'))]
            imgs.sort()
            prediction = model.predict_generator(test_generator(imgs), max(2, ceil(len(imgs) / frame_num)))
            predictions = prediction[0]

            for j in range(0,len(imgs)):
                original_img = io.imread(imgs[j])
                div, mod = divmod(j, frame_num)
                s_map = postprocess_map(predictions[div, mod, :, :, 0],  original_img.shape[0], original_img.shape[1])
                io.imsave(map_path + imgs[j].split('/')[-1].split('.')[0] + '.png', s_map)
            model.reset_states()


# In[ ]:





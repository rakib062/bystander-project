
# coding: utf-8

# In[10]:

import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import math

import keras
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.utils import plot_model
from keras import backend as K
from keras.regularizers import Regularizer
import cv2
import tensorflow as tf

import keras_metrics
#from sklearn import cross_validation, grid_search

import matplotlib.pyplot as plt
from tqdm import tqdm

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) # reference: https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer

import sys
# sys.path.append('/nfs/juhu//data/rakhasan/reputation/jupyter-notes/bystander-project/') #for helper
# #import helper

#survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
resnet_feature_cols = ['resnet_feat_{}'.format(i) for i in range(131071)]

# In[11]:

IMG_SIZE = (256, 256)
IN_SHAPE = (256, 256, 3)



def resnet_model(averaged=True):
    '''Build model using resnet pretrained with ImageNet'''
    
    pretrained_model = ResNet50(
                include_top=False,
                input_shape=IN_SHAPE,
                weights='imagenet'
            )
    output = pretrained_model.output

    if averaged:
        output = GlobalAveragePooling2D()(output)
    
    #output = Dense(1, activation='sigmoid')(output)
    model = Model(pretrained_model.input, output)
#
    for layer in pretrained_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    return model


# def resnet_model(averaged=False):
#     '''Build model using resnet pretrained with ImageNet'''
    
#     pretrained_model = ResNet50(
#                 include_top=False,
#                 input_shape=IN_SHAPE,
#                 weights='imagenet'
#             )
#     output = pretrained_model.output
#     if averaged:
#       output = GlobalAveragePooling2D()(output)
#     output=Flatten(name="flatten")(output)
    
#     model = Model(pretrained_model.input, output)
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
#     return model

def get_cropped(path):
    try:
        img = cv2.imread(path)
        resized=cv2.resize(img, 
                       dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        print(n, e)


def extract_resnet_feature(survey_path, feature_df, update=True):
    '''
    If update is True, then it will load previously created feature set and add features for new images
    '''
    model = resnet_model(averaged=True)
    feature_dics = []
    photo_nos=list(set(feature_df.index))
    
    out_file = survey_path+'/resnet-features-df.pkl'
    
    old_features = None
    old_photos = set()
    if update and os.path.exists(out_file):
        old_features = pd.read_pickle(out_file)        
        old_photos = set(old_features.index)
        print('old_features: ',old_features.shape)
    
    for i in tqdm(range(len(photo_nos))):
        photo_no = photo_nos[i]
        if photo_no in old_photos:
            print("Features for {} already exists.".format(photo_no))
            continue
        img = get_cropped(survey_path+'cropped-photos/'+str(photo_no)+'.jpg')
        img = np.expand_dims(img, axis=0)
        features = model.predict(img)
        #print(features.shape)
        d={'photo_no':photo_no}
        for j in range(len(features[0])):
          d['resnet_feat_avg_{}'.format(j)] = features[0][j]
        feature_dics.append(d)

    rn_df = pd.DataFrame(feature_dics)

    rn_df.set_index('photo_no', inplace=True)
    rn_df['label'] = feature_df.label
    if old_features is not None:
        rn_df = pd.concat([rn_df, old_features])
    
    print(rn_df.shape)
    rn_df.to_pickle(out_file)
    rn_df.to_csv(survey_path+'/resnet-features-df.csv')

if __name__=='__main__':
    survey_path=sys.argv[1]
    feature_df = pd.read_pickle(survey_path+'/features-df.pkl')
    extract_resnet_feature(survey_path, feature_df)

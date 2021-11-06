
# coding: utf-8

# In[10]:

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc,accuracy_score
import math
from scipy import interp

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

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation, grid_search
import keras_metrics
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from scipy import stats

import matplotlib.pyplot as plt

import numpy as np
import random as rn


from keras import backend as K

from plot_helper import fine_tune_resnet_roc

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

survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'

# In[11]:

IMG_SIZE = (256, 256)
IN_SHAPE = (256, 256, 3)
BATCH_SIZE = 64

labelFont = 18
legendFont = 16
tickFont = 16

manuscriptColSize= 252
forLatex = True
latexDec = 6
# In[9]:

feature_df = pd.read_pickle(survey_path+'/feature-df-oct-28.pkl')
feature_df = feature_df[(feature_df.label==1)|(feature_df.label==-1)]
feature_df['label'] = feature_df.apply(lambda row: 1 if row.label==1 else 0, axis=1)
print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
     'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))

resnet_feature_cols = ['resnet_feat_{}'.format(i) for i in range(131071)]

input_dim = len(resnet_feature_cols)

def resnet_model():
    '''Build model using resnet pretrained with ImageNet'''
    
    pretrained_model = ResNet50(
                include_top=False,
                input_shape=IN_SHAPE,
                weights='imagenet'
            )
    output = pretrained_model.output
    #output = GlobalAveragePooling2D()(output)
    output=Flatten(name="flatten")(output)
    
    output = Dense(1, activation='sigmoid')(output)
    model = Model(pretrained_model.input, output)
#
    for layer in pretrained_model.layers:
        layer.trainable = False
#
    #model.summary(line_length=200)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    return model

def dense_model():    
    input_layer = Input(shape=(input_dim,), name = 'input_layer')    
    output_layer = Dense(1, kernel_regularizer=keras.regularizers.l2(1), 
                bias_regularizer=keras.regularizers.l2(1), 
                         activation = 'sigmoid')(input_layer)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), 
                  keras_metrics.recall(), keras_metrics.f1_score()])
    return model

def get_cropped(n):
    try:
        img = cv2.imread(survey_path+'cropped-photos/'+str(n)+'.jpg')
        resized=cv2.resize(img, 
                       dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        print(n, e)

# print("Loading cropped images.")

# resized = dict()
# for img in set(feature_df.index):
#     resized[img]=get_cropped(img)

# feature_df['resized_cropped_img'] = pd.Series(resized)

print("Loading resnet extracted features")
rn_df = pd.read_pickle(survey_path+'resnet-features-df-2019-10-31.pkl')
# rn_dicts = []
# for k in rn.keys():
#     feats = rn[k]
#     d= {'photo_no':str(k)}
#     for i in range(len(resnet_feature_cols)):
#         d[resnet_feature_cols[i]] = feats[0][i]
#     rn_dicts.append(d)
# rn_df = pd.DataFrame(rn_dicts)
# rn_df.set_index('photo_no', inplace=True)

#feature_df= pd.concat([feature_df, rn_df[resnet_feature_cols]], axis=1)

print("Testing with resnet extracted features")

fine_tune_resnet_roc(classifier_func=dense_model,
                   X= rn_df[resnet_feature_cols].values,# preprocessing.MinMaxScaler().fit_transform(rn_df[resnet_feature_cols]),
                   y=rn_df.label.values, 
                   batch_size=BATCH_SIZE,
                   epochs=30,
                   n_splits=10,
                    show_plot=False, 
                    forLatex=True,
                  square=True,
                    save_file=survey_path+'fine-tune-resnet-roc-extracted.pdf')

# fine_tune_resnet_roc(classifier_func=resnet_model,
#                    X=  np.array([x for x in feature_df.resized_cropped_img]),
#                    y=feature_df.label, 
#                    batch_size=BATCH_SIZE,
#                    #epochs=3,
#                     show_plot=False, 
#                     forLatex=True,
#                   square=True,
#                     save_file=survey_path+'fine-tune-resnet-roc-3epochs-no-avg.pdf')

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




# In[5]:

from keras import optimizers



def get_fig_size(figWidthPt):
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (math.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = figWidthPt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean       # height in inches
    return (fig_width,fig_height)


def resnet_model():
    '''Build model using resnet pretrained with ImageNet'''
    
    pretrained_model = ResNet50(
                include_top=False,
                input_shape=IN_SHAPE,
                weights='imagenet'
            )
    output = pretrained_model.output
    output = GlobalAveragePooling2D()(output)
    
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

def fine_tune_resnet_roc(X, y, classifier_func, n_splits=10, epochs=30, batch_size=256, verbose=1,
                     label_font = 18, legend_font=14, show_plot=True,forLatex=True, square=True, save_file=None):
    '''
    Fine tune ResNet model with raw cropped images. Do cross-validation and plot ROC.
    X is a numpy array containing the cropped images
    '''
    #
    random_state = np.random.RandomState(0)
    cv_scores = []
    #
    cv = StratifiedKFold(n_splits=n_splits)
    #
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    fig= plt.figure(facecolor='w')
    plt.fill(False)
    if(forLatex):
        print('setting latex size')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif='Times')
        w,h = get_fig_size(manuscriptColSize)
        fig_size = ((w, w if square else h))
        fig.set_size_inches(fig_size)
    else:
        fig.set_size_inches((12,8))
#
    i = 0
    for train, test in cv.split(X, y):
        print('\n\n***********Split {} of {}**************\n\n'.format(i, n_splits))
        model = classifier_func()
        model.random_state = random_state
        model.probability=True
        #
        model.fit(X[train], y[train], 
                  epochs=epochs, batch_size=batch_size, verbose=verbose)
        
#         predictions = model.predict(X[test])
#         cv_scores.append(accuracy_score(y_pred=predictions, y_true=y[test]))
#         print(classification_report(y_pred=predictions, y_true=y[test]))
        #
        probas_ = model.predict(X[test])
        # Compute ROC curve and area of the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
#
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
#
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                     label=None)#r'$\pm$ 1 std. dev.')
#
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=labelFont-forLatex*latexDec-2)
    plt.ylabel('True Positive Rate', fontsize=labelFont-forLatex*latexDec-2)
    plt.legend(fontsize=legendFont-forLatex*latexDec-2, markerscale=0.4, loc="lower right")
#   
    fig.set_facecolor("w")
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=300, frameon=True, facecolor=fig.get_facecolor(), edgecolor='black', transparent=True)
    if show_plot:
        plt.show()


def get_cropped(n):
    try:
        img = cv2.imread(survey_path+'cropped-photos/'+str(n)+'.jpg')
        resized=cv2.resize(img, 
                       dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        print(n, e)

print("Loading cropped images.")

resized = dict()
for img in set(feature_df.index):
    resized[img]=get_cropped(img)

feature_df['resized_cropped_img'] = pd.Series(resized)


fine_tune_resnet_roc(classifier_func=resnet_model,
                   X=  np.array([x for x in feature_df.resized_cropped_img]),
                   y=feature_df.label, 
                   n_splits=3, 
                   batch_size=BATCH_SIZE,
                    legend_font=14, 
                    show_plot=False, 
                    epochs=3,
                    save_file=survey_path+'fine-tune-resnet-roc.pdf')
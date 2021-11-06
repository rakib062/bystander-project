
# coding: utf-8

# In[2]:

import matplotlib
matplotlib.use('agg')

import os
import glob

#from tqdm import tqdm
import numpy as np
import scipy.ndimage
import scipy.misc
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from IPython.display import clear_output

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

from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc,accuracy_score
from scipy import interp
from sklearn import cross_validation, grid_search

import keras_metrics
from scipy import stats

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import random as rn


# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) # reference: https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer

import sys
sys.path.append('/nfs/juhu//data/rakhasan/reputation/jupyter-notes/bystander-project/') #for helper
import helper


# In[3]:


def do_cross_val_roc_custom_model(model_func, X,Y, n_splits=5, 
                        epochs=20, save_model = True,
                        save_prefix='', model_args=None):
    seed = 1234
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    
    fold = 0
    splits = []
    for train, test in kfold.split(X, Y):
        # create model
        if model_args:
            model = model_func(model_args['input_dim'], model_args['hidden_dims'])
        else:
            model = model_func()
        model.fit(X[train], Y[train], epochs=epochs, batch_size=BATCH_SIZE, verbose=1)
        #evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=1)
        print(model.metrics_names, scores)
        cvscores.append(scores)
        
        probas_ = model.predict_proba(X[test])
        # Compute ROC curve and area of the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=legend_font+2)
    plt.ylabel('True Positive Rate', fontsize=legend_font)
    #plt.title('Receiver Operating characteristic')
    plt.legend(loc="lower right", fontsize=legend_font)
    if save_file:
        plt.savefig(save_file, dpi=3)
    plt.show()
    
    return cvscores


# In[4]:


IMG_SIZE = (256, 256)
IN_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 64


# In[5]:


openImg_path = '/nfs/juhu/data/rakhasan/bystander-detection/google-img-db/'
survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
survey_photo_path = survey_path+'/photos/'

model_output_path = '/nfs/juhu/data/rakhasan/bystander-detection/code-repos/notebooks/model-output/'

print('loading features.')
'''Note: this data frame has all deep features, and the label is binary values (0,1)'''
feature_df = pickle.load(open(os.path.join(survey_path, 'features-df-all.pkl'), 'rb'))
#feature_df = feature_df[(feature_df.label==1)|(feature_df.label==-1)]
#feature_df['label'] = feature_df.apply(lambda row: 1 if row.label==1 else 0, axis=1)
print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
     'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))


# In[6]:


'''Feature names'''

#joint names labeled by openpose
body_joint_names = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',
               'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 
               'Leye', 'Reye', 'Lear', 'Rear']

#angles between pairs of body joint, from openpose
link_angle_features = ['angle_'+str(i) for i in range(17)]

#probability of detecting a body joint, from openpose
body_joint_prob_features = [j + '_prob' for j in body_joint_names]

face_exp_feaures = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

img_level_features = ['person_distance', 'person_size', 'num_people']

visual_features = img_level_features +    link_angle_features + body_joint_prob_features + face_exp_feaures

features_from_study = ['was_aware_num',  'posing_num',  'comfort_num',  'will_num', 'photographer_intention_num',
     'replacable_num',  'photo_place_num']

resnet_feature_cols = ['resnet_feat_{}'.format(i) for i in range(131071)]

resnet_feat_avg_cols = ['resnet_feat_avg_{}'.format(i) for i in range(2048)]

all_features = features_from_study + visual_features + resnet_feat_avg_cols


# In[14]:



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

helper.fine_tune_resnet_roc(classifier_func=resnet_model,
                   X=  np.array([x for x in feature_df.resized_cropped_img]),
                   y=feature_df.label, 
                   n_splits=10, 
                   batch_size=BATCH_SIZE,
                    legend_font=14, 
                    show_plot=False, 
                    save_file='fine-tune-resnet-roc.pdf')
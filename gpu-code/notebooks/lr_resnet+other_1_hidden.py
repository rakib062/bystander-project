'''Train a Logistic regression model including one hidden layer with resnet features'''

import os
import glob

#from tqdm import tqdm
import numpy as np
import scipy.ndimage
import scipy.misc
import pandas as pd
import pickle
from IPython.display import clear_output

import keras
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.utils import plot_model
from keras import backend as K


from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

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



IMG_SIZE = (256, 256)
IN_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 64

openImg_path = '/nfs/juhu/data/rakhasan/bystander-detection/google-img-db/'
survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
survey_photo_path = survey_path+'/photos/'

model_output_path = '/nfs/juhu/data/rakhasan/bystander-detection/code-repos/notebooks/model-output/'

print('loading features.')
feature_df = pickle.load(open(os.path.join(survey_path, 'features-df-all.pkl'), 'rb'))
print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
     'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))

link_angle_features = ['angle_'+str(i) for i in range(17)]

#joint names labeled by openpose
body_joint_names = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',
               'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 
               'Leye', 'Reye', 'Lear', 'Rear']

#probability of detecting a body joint, from openpose
body_joint_prob_features = [j + '_prob' for j in body_joint_names]

face_exp_feaures = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

img_level_features = ['person_distance', 'person_size', 'num_people']

visual_features = img_level_features +\
    link_angle_features + body_joint_prob_features + face_exp_feaures

features_from_study = ['was_aware_num',  'posing_num',  'comfort_num',  'will_num', 'photographer_intention_num',
     'replacable_num',  'photo_place_num']

resnet_feature_cols = ['resnet_feat_{}'.format(i) for i in range(131071)]

resnet_feat_avg_cols = ['resnet_feat_avg_{}'.format(i) for i in range(2048)]

all_features = features_from_study + visual_features + resnet_feat_avg_cols


def dense_model(input_dim, hidden_dims = None):
    '''Create a fully connected network with first layer as input with input_dim=input_dim,
    and len(hidden_dims) number of hidden layers.
    
    Currenly default activation is relu for all hidden layers, and a dropout(.5) is added.
    
    '''
    
    input_layer = Input(shape=(input_dim,), name = 'input_layer')
    hidden_layer = input_layer
    if hidden_dims:
        for hidden_dim in hidden_dims:
            hidden_layer = Dense(hidden_dim, activation='relu')(hidden_layer)
            hidden_layer = Dropout(.5)(BatchNormalization()(hidden_layer))
    
    output_layer = Dense(1, kernel_regularizer=keras.regularizers.l2(1), bias_regularizer=keras.regularizers.l2(1), 
                         activation = 'sigmoid')(hidden_layer)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def do_cross_validation(model_func, X,Y, n_splits=5, epochs=20, save_model = True, save_prefix='', model_args=None):
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
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        if save_model:
            model.save_weights(model_output_path+'model_{}_{}.weights'.format(save_prefix, fold))
            fold+=1
            splits.append((train, test))
    
    if save_model:
        pickle.dump(splits, open(model_output_path+'splits_{}'.format(save_prefix), 'wb'))

    return cvscores


if __name__== "__main__":
    scorelist=[]

    #resnet+survey features
    scores = do_cross_validation(dense_model, preprocessing.MinMaxScaler().fit_transform(
        feature_df[resnet_feat_avg_cols]),
     feature_df.label, epochs= 30, n_splits = 10, model_args ={'input_dim':len(resnet_feat_avg_cols), 
        'hidden_dims':[64]}, save_prefix='resnet_feat_no_hidden_layer')
    scorelist.append(scores)

    #resnet+survey +image level features
    feats = resnet_feat_avg_cols+img_level_features
    scores = do_cross_validation(dense_model,preprocessing.MinMaxScaler().fit_transform(
        feature_df[feats]),
     feature_df.label, epochs= 30, n_splits = 10, model_args ={'input_dim':len(feats), 
        'hidden_dims':[64]}, save_prefix='resnet_feat_no_hidden_layer')
    scorelist.append(scores)

    #resnet+all features
    scores = do_cross_validation(dense_model, preprocessing.MinMaxScaler().fit_transform(
        feature_df[all_features]),
     feature_df.label, epochs= 30, n_splits = 10, model_args ={'input_dim':len(all_features), 
        'hidden_dims':[64]}, save_prefix='resnet_feat_no_hidden_layer')
    scorelist.append(scores)
    
    for scores in scorelist:
        print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores))) 
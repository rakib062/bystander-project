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
from keras.layers import GlobalAveragePooling2D


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

import keras_metrics


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



def resnet_mixed_features(other_feat_dim, hidden_dims = []):
    '''
    Build model using resnet pretrained with ImageNet.
    Also add other features at the higher level of the network.
    '''
    pretrained_model = ResNet50(
                include_top=False,
                input_shape=IN_SHAPE,
                weights='imagenet'
            )
    
    output = pretrained_model.output
    cnn_feat = GlobalAveragePooling2D()(output)

    
    input2 = Input(shape=(other_feat_dim,), name = 'other_feat')
    
    merged_feat = Concatenate(name='merged_feat')([cnn_feat, input2])
    
    for hidden_dim in hidden_dims:
        merged_feat = Dense(hidden_dim, activation='relu')(merged_feat)
        merged_feat = Dropout(.5)(BatchNormalization()(merged_feat))
    
    final_output = Dense(1, activation='sigmoid')(merged_feat)
    
    model = Model([pretrained_model.input, input2], final_output)

    for layer in pretrained_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    return model
    


def fine_tune_cv_mixed_features(model_func, feature_df, other_feat, hidden_dims=[],
                                n_splits=10, epochs=20, save_model = True, save_prefix = ''):
    seed = 1234
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    
    fold = 0
    splits = []
    for train, test in kfold.split(feature_df, feature_df.label):
        
        X_train = np.array([x for x in feature_df.resized_cropped_img.values[train]])
        X_test = np.array([x for x in feature_df.resized_cropped_img.values[test]])

        model = resnet_mixed_features(len(other_feat), hidden_dims)
        model.fit([X_train, feature_df[other_feat].apply(stats.zscore).values[train]],
                  feature_df.label[train], epochs=epochs, batch_size=BATCH_SIZE, verbose=1)
        
        
        scores = model.evaluate([X_test, feature_df[other_feat].apply(stats.zscore).values[test]],
                                feature_df.label[test], verbose=1)
        print(model.metrics_names, scores)
        cvscores.append(scores)
        
        if save_model:
            model.save_weights(model_output_path+'model_{}_{}.weights'.format(save_prefix,fold))
            fold+=1
            splits.append((train, test))
    
    if save_model:
        pickle.dump(splits, open(model_output_path+'splits_{}'.format(save_prefix), 'wb'))
        
    return cvscores


if __name__== "__main__":
    print('loading features.')
    feature_df = pickle.load(open(os.path.join(survey_path, 'features-df-all.pkl'), 'rb'))
    print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
         'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))

    
    scores = fine_tune_cv_mixed_features(resnet_mixed_features, feature_df, 
        other_feat=features_from_study + visual_features,
        n_splits=2, epochs=20, save_model=False)

    print('acc: {:.4f} +/- {:.4f}\nprecision: {:.4f} +/- {:.4f}\nrecall: {:.4f} +/- {:.4f}\nf1: {:.4f} +/- {:.4f}'.format(
    np.mean([s[1] for s in scores]), np.std([s[1] for s in scores]),
    np.mean([s[2] for s in scores]), np.std([s[2] for s in scores]),
    np.mean([s[3] for s in scores]), np.std([s[3] for s in scores]),
    np.mean([s[4] for s in scores]), np.std([s[4] for s in scores])))

#acc: 0.8041 +/- 0.0107
#precision: 0.9305 +/- 0.0066
#recall: 0.7289 +/- 0.0252
#f1: 0.8171 +/- 0.0133
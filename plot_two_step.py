
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

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation, grid_search
import keras
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import keras_metrics


import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from scipy import stats

import matplotlib.pyplot as plt

import numpy as np
import random as rn


from keras import backend as K


import sys
from plot_helper import do_cross_val_roc

survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'


# In[11]:

IMG_SIZE = (256, 256)
IN_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 64


# In[9]
feature_df = pd.read_pickle(survey_path+'/feature-df-oct-28.pkl')
feature_df = feature_df[(feature_df.label==1)|(feature_df.label==-1)]
feature_df['label'] = feature_df.apply(lambda row: 1 if row.label==1 else 0, axis=1)
print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
     'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))


# In[7]:

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


# In[5]:

from keras import optimizers

def linear_regression_model(input_dim=38, hidden_dims = []):
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
                         activation = 'linear')(hidden_layer)
    
    model = Model(input_layer, output_layer)
    
    model.compile(optimizer=optimizers.SGD( lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5),
                  loss='mse',
                  metrics=['mse','mae'])
    return model


print('Generating predicted features.')
#feat = resnet_feat_avg_cols + img_level_features + link_angle_features + body_joint_prob_features + face_exp_feaures
feat = resnet_feat_avg_cols  + link_angle_features + body_joint_prob_features + face_exp_feaures

will_model = linear_regression_model(input_dim=len(feat), hidden_dims=[64])
will_model.fit(feature_df[feat].apply(stats.zscore).values, feature_df['will_num'],
          epochs=30, batch_size=BATCH_SIZE, verbose=0)
feature_df['predicted_will'] = will_model.predict(feature_df[feat].apply(stats.zscore).values)

comfort_model = linear_regression_model(input_dim=len(feat), hidden_dims=[64])
comfort_model.fit(feature_df[feat].apply(stats.zscore).values, feature_df.comfort_num,
          epochs=30, batch_size=BATCH_SIZE, verbose=0)
feature_df['predicted_comfort'] =comfort_model.predict(feature_df[feat].apply(stats.zscore).values)


aware_model = linear_regression_model(input_dim=len(feat), hidden_dims=[64])
aware_model.fit(feature_df[feat].apply(stats.zscore).values, feature_df.was_aware_num,
          epochs=30, batch_size=BATCH_SIZE, verbose=0)
feature_df['predicted_aware'] = aware_model.predict(feature_df[feat].apply(stats.zscore).values)

pose_model = linear_regression_model(input_dim=len(feat), hidden_dims=[64])
pose_model.fit(feature_df[feat].apply(stats.zscore).values, feature_df.posing_num,
          epochs=30, batch_size=BATCH_SIZE, verbose=0)
feature_df['predicted_posing'] = pose_model.predict(feature_df[feat].apply(stats.zscore).values)

replace_model = linear_regression_model(input_dim=len(feat), hidden_dims=[64])
replace_model.fit(feature_df[feat].apply(stats.zscore).values, feature_df.replacable_num,
          epochs=30, batch_size=BATCH_SIZE, verbose=0)
feature_df['predicted_replace'] =replace_model.predict(feature_df[feat].apply(stats.zscore).values)


intention_model = linear_regression_model(input_dim=len(feat), hidden_dims=[64])
intention_model.fit(feature_df[feat].apply(stats.zscore).values, feature_df.photographer_intention_num,
          epochs=30, batch_size=BATCH_SIZE, verbose=0)
feature_df['predicted_intention'] =intention_model.predict(feature_df[feat].apply(stats.zscore).values)


feat=['predicted_posing', 'predicted_replace', 'predicted_intention','person_size']
print('Features:',feat)

do_cross_val_roc(classifier_func=LogisticRegression,
                        X=feature_df[feat].apply(stats.zscore).values,
                        y = feature_df.label.values,
                        forLatex=True,
                        square=True,
                        save_file=survey_path+'predicted-pose-roc.pdf'
                        )


sys.path.append('/nfs/juhu//data/rakhasan/reputation/jupyter-notes/bystander-project/') #for helper
import helper

helper.do_cross_val_roc(classifier_func=LogisticRegression,
                        X=feature_df[feat].apply(stats.zscore).values,
                        y = feature_df.label.values,
                        legend_font=16,
                        label_font=28,
                        forLatex=True,
                        save_file='/Users/rakhasan/Publc/predicted-pose-roc-helper.pdf',                        
                        )

print("Comparing with human")
agreement_23 = pickle.load(open(survey_path+'agreement23.pkl','rb'))
agreement_100 = pickle.load(open(survey_path+'agreement100.pkl','rb'))

feat=['posing_num', 'replacable_num', 'photographer_intention_num','person_size']

df = feature_df.loc[agreement_23][feat+['label']]
do_cross_val_roc(classifier_func=LogisticRegression,
                   X= df[feat].apply(stats.zscore).values,
                   y=df.label, n_splits=10,
                    forLatex=True,
                 square=True,
                    save_file=survey_path+'67-perc-agreement-roc.pdf')

df = feature_df.loc[agreement_100][feat+['label']]
do_cross_val_roc(classifier_func=LogisticRegression,
                   X= df[feat].apply(stats.zscore).values,
                   y=df.label, n_splits=10,
                    forLatex=True,
                    square=True,
                    save_file=survey_path+'100-perc-agreement-roc.pdf')
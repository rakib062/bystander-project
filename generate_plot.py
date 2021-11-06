# generic imports
import sys
import os
#from definitions import *


import math
from collections import defaultdict
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import StratifiedKFold

#stats amd ml
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc,accuracy_score
from scipy import interp
import cv2
from scipy import stats


from plot_helper import do_cross_val_roc

min_max_scaler = preprocessing.MinMaxScaler()


IMG_SIZE = (256, 256)

survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
#feature_df = pd.read_pickle(os.path.join(survey_path, 'features-df.pkl'))
feature_df =pd.read_pickle(os.path.join(survey_path, 'all-features-df-2019-10-30.pkl'))
feature_df = feature_df[(feature_df.label==1) | (feature_df.label==-1)]
feature_df['label'] = feature_df.apply(lambda row: 1 if row.label==1 else 0, axis=1)
print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
     'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))

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

img_level_features = ['person_distance_axes_norm', 'person_size', 'num_people']

visual_features = img_level_features +\
    link_angle_features + body_joint_prob_features + face_exp_feaures

features_from_study = ['was_aware_num',  'posing_num',  'comfort_num',  'will_num', 'photographer_intention_num',
     'replacable_num',  'photo_place_num']

resnet_feature_cols = ['resnet_feat_{}'.format(i) for i in range(131071)]

resnet_feat_avg_cols = ['resnet_feat_avg_{}'.format(i) for i in range(2048)]

all_features = features_from_study + visual_features + resnet_feat_avg_cols


def get_cropped(n):
    try:
        img = cv2.imread(survey_path+'cropped-photos/'+str(n)+'.jpg')
        resized=cv2.resize(img, 
                       dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        print(n, e)


print('Plotting ground truth roc.')
feat=['posing_num', 'replacable_num', 'photographer_intention_num','person_size']
do_cross_val_roc(classifier_func=LogisticRegression,
                    X= min_max_scaler.fit_transform(
                    feature_df[feat]),
                    y=feature_df.label, n_splits=10,
                    save_file=survey_path+'ground-truth-pose-roc.pdf',
                    forLatex=True,
                     square=True
                    )

print('Plotting simple features roc.')
do_cross_val_roc(classifier_func=LogisticRegression,
                   X= feature_df[img_level_features].apply(stats.zscore).values,
                   y=feature_df.label, n_splits=10,
                     forLatex=True,
                 square=True,
                    save_file=survey_path+'simple-predictor-roc.pdf')


print('Plotting raw image features roc.')
do_cross_val_roc(classifier_func=LogisticRegression,
					X= min_max_scaler.fit_transform(
					   np.array([x.flatten() for x in feature_df.resized_cropped_img])),
					y=feature_df.label,
					forLatex=True,
					square=True,
					save_file=survey_path+'raw-cropped-img-roc.pdf')


print('Plotting all visual features roc.')
do_cross_val_roc(classifier_func=LogisticRegression,
                   X=  min_max_scaler.fit_transform(feature_df[resnet_feat_avg_cols+\
                   				body_joint_prob_features+\
                                link_angle_features+ face_exp_feaures]),
                   y=feature_df.label, 
                   n_splits=10,
                 forLatex=True,
                 square=True,
                  save_file=survey_path+'all-visual-features-roc.pdf')

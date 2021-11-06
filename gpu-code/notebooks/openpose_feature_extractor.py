import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.chdir('/nfs/juhu/data/rakhasan/bystander-detection/gpu-code/keras-openpose-reproduce/')
#sys.path.append('/nfs/nfs7/home/rakhasan/Desktop/reputation/jupyter-notes/bystander-project/')
sys.path.append('/nfs/juhu/data/rakhasan/bystander-detection/gpu-code/keras-openpose-reproduce/')
import openpose_wrapper as wrapper

import matplotlib.pyplot as plt
import cv2


import pandas as pd
import numpy as np
import math 

import pickle
from tqdm import tqdm

import tensorflow as tf
from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) # reference: https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer

body_joint_names = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',
               'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 
               'Leye', 'Reye', 'Lear', 'Rear']

#angles between pairs of body joint, from openpose
link_angle_features = ['angle_'+str(i) for i in range(17)]

#probability of detecting a body joint, from openpose
body_joint_prob_features = [j + '_prob' for j in body_joint_names]



def extract_openpose_feature(survey_path, feature_df, update=True):
    model=wrapper.get_model()

    photos = list(feature_df.index.values)
    photos = list(feature_df.index)#[-3:]
    dicts =[]
    
    out_file = survey_path+'/openpose-feature.pkl'
    
    old_features = None
    old_photos = set()
    if update and os.path.exists(out_file):
        old_features = pd.read_pickle(out_file)        
        old_photos = set(old_features.index)
        print('old_features: ',old_features.shape)
    
    
    for i in tqdm(range(len(photos))):
        if photos[i] in old_photos:
            print('Feature for photo {} already exists'.format(photos[i]))
            continue
        img = cv2.imread('{}/cropped-photos/{}.jpg'.format(survey_path, photos[i]))
        peaks, angles = wrapper.get_single_person_joint_angle(model=model, oriImg=img)
        peaks_prob = [p[0][2] for p in peaks]

        print(len(peaks), len(angles))

        d={'photo_no':photos[i]}
        for j in range(len(peaks_prob)):
            d[body_joint_prob_features[j]]=peaks_prob[j]
        for j in range(len(link_angle_features)):
            d[link_angle_features[j]]=angles[j]

        dicts.append(d)    

    df = pd.DataFrame(dicts)
    df.set_index('photo_no', inplace=True)
    
    if old_features is not None:
        df = pd.concat([df, old_features])
    
    
    df.to_pickle(out_file)
    df.to_csv(survey_path+'/openpose-feature.csv')

if __name__=='__main__':
    survey_path=sys.argv[1]
    feature_df = pd.read_pickle(survey_path+'/high-feature-df.pkl')
    extract_openpose_feature(survey_path, feature_df)
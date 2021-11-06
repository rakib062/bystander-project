import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tqdm import tqdm
import pandas as pd
import pickle 
import cv2


import tensorflow as tf
from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) # reference: https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer


BASE_PATH = '/nfs/juhu/data/rakhasan/bystander-detection/'
sys.path.append(BASE_PATH)

sys.path.append(BASE_PATH+'gpu-code/Tiny_Faces_in_Tensorflow/')
import detect_face

sys.path.append(BASE_PATH+'gpu-code/facial-expression-recognition/')
import face_exp

sys.path.append(BASE_PATH+'gpu-code/keras-openpose-reproduce/')
import openpose_wrapper as wrapper 

import extract_resnet_features

sys.path.append(BASE_PATH+'gpu-code/notebooks/')
import openpose_feature_extractor



def main():
    survey_path=sys.argv[1]
    feature_df = pd.read_pickle(survey_path+'/high-feature-df.pkl')
    #feature_df=feature_df.head()
    print('dataset:',len(feature_df), 'unique labels:', feature_df.label.unique(),
         'pos:',len(feature_df[feature_df.label==1]),'neg:',len(feature_df[feature_df.label==0]))

    print('\n\ndetecting faces...')
    detect_face.detect_all_faces(survey_path, set(feature_df.index))

    print('\n\nEstimating facial expression...')
    face_exp.estimate_facial_expression(survey_path)

    print('\n\nEstimating ResNet featues...')
    extract_resnet_features.extract_resnet_feature(survey_path, feature_df)

    #os.chdir('/nfs/juhu/data/rakhasan/bystander-detection/gpu-code/keras-openpose-reproduce/')
    #print('\n\nExtracting openpose features...')
    #openpose_feature_extractor.extract_openpose_feature(survey_path, feature_df)

    


if __name__=='__main__':
    main()
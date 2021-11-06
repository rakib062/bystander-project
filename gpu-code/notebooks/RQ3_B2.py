
# coding: utf-8

# In[39]:


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
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.utils import plot_model
from keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import random as rn

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

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


# In[2]:


IMG_SIZE = (256, 256)
IN_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 64


# In[11]:


openImg_path = '/nfs/juhu/data/rakhasan/bystander-detection/google-img-db/'
survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
survey_photo_path = survey_path+'/photos/'

model_output_path = '/nfs/juhu/data/rakhasan/bystander-detection/code-repos/notebooks/model-output/'

print('loading features.')
feature_df = pickle.load(open(os.path.join(survey_path, 'features-df.pkl'), 'rb'))


# In[59]:


#joint names labeled by openpose
body_joint_names = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',
               'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 
               'Leye', 'Reye', 'Lear', 'Rear']

#angles between pairs of body joint, from openpose
link_angle_features = ['angle_'+str(i) for i in range(17)]

#probability of detecting a body joint, from openpose
body_joint_prob_features = [j + '_prob' for j in body_joint_names]

face_exp_feaures = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

visual_features = ['person_distance', 'person_size', 'num_people'] +    link_angle_features + body_joint_prob_features + face_exp_feaures


# In[4]:


def split_data(X,Y, test_perc = 0.1):
    indices = rn.sample(range(1, len(Y)), int(len(Y)*test_perc))
    Xtest = X[indices]
    Ytest = Y[indices]
    Xtrain = X[list(set(range(len(Y))).difference(set(indices)))]
    Ytrain = Y[list(set(range(len(Y))).difference(set(indices)))]
    
    return (Xtrain,Ytrain),(Xtest,Ytest)


# In[99]:


def do_cross_validation(model_func, X,Y, n_splits=5, save_model = True):
    seed = 1234
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    
    fold = 0
    splits = []
    for train, test in kfold.split(X, Y):
        # create model
        model = model_func()
        model.fit(np.array([x for x in X[train]]), Y[train], epochs=20, batch_size=BATCH_SIZE, verbose=1)
        #evaluate the model
        scores = model.evaluate(np.array([x for x in X[test]]), Y[test], verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        if save_model:
            model.save_weights(model_output_path+'model_raw_img_{}.weights'.format(fold))
            fold+=1
            splits.append((train, test))
    
    if save_model:
        pickle.dump(splits, open(model_output_path+'splits_raw_img', 'wb'))
        
    return cvscores

def do_cross_validation_mixed_features(model_func, img_feat, other_feat, Y, n_splits, save_model = True):
    seed = 1234
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    
    fold = 0
    splits = []
    for train, test in kfold.split(img_feat, Y):
        # create model
        model = resnet_mixed_features((other_feat.shape[1],))
        model.fit([np.array([x for x in img_feat[train]]), other_feat.values[train]],
                  Y[train], epochs=20, batch_size=BATCH_SIZE, verbose=1)

        #evaluate the model
        scores = model.evaluate([np.array([x for x in img_feat[test]]), other_feat.values[test]], Y[test], verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        if save_model:
            model.save_weights(model_output_path+'model_mixed_all_features_{}.weights'.format(fold))
            fold+=1
            splits.append((train, test))
    
    if save_model:
        pickle.dump(splits, open(model_output_path+'splits_mixed_all_features', 'wb'))
        
    return cvscores


# ## Fine tune Resnet model pretrained with imagenet

# In[56]:


'''Build model using pretrained ImageNet'''
def resnet_with_imagenet():
    pretrained_model = ResNet50(
                include_top=False,
                input_shape=IN_SHAPE,
                weights='imagenet'
            )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
    else:
        output = pretrained_model.output

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(pretrained_model.input, output)

    for layer in pretrained_model.layers:
        layer.trainable = False

    #model.summary(line_length=200)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def resnet_mixed_features(feat_dim):
    pretrained_model = ResNet50(
                include_top=False,
                input_shape=IN_SHAPE,
                weights='imagenet'
            )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
    else:
        output = pretrained_model.output

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization(name='final_normalization')(output)
    cnn_feat = Dropout(0.5, name='final_dropout')(output)
    
    other_feat = Input(shape=feat_dim, name = 'other_feat')
    
    merged_feat = Concatenate(name='merged_feat')([cnn_feat, other_feat])
    
    final_output = Dense(1, activation='sigmoid')(merged_feat)
    
    model = Model([pretrained_model.input, other_feat], final_output)

    for layer in pretrained_model.layers:
        layer.trainable = False

    #model.summary(line_length=200)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model




def get_feature_layer_func(model):
    return K.function([model.layers[0].input],
                                  [model.layers[len(model.layers)-2].output])
#layer_output = get_cnn_feature([x])[0]


# #### Using only raw image

# In[14]:


'''Do cross-validation by first fine-tuning resnet model with raw cropped image.'''
# scores = do_cross_validation(resnet_with_imagenet,
#         feature_df.resized_cropped_img[:3], feature_df.label[:3], n_splits = 2)
# #clear_output()
# print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))


# In[22]:


# '''40% dilated box with ImageNet'''
# X,Y= load_XY(photo_path='dilated-box40')
# scores = do_cross_validation(imagenet, X, Y)
# clear_output()
# print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))


# #### Raw image with other features

# In[ ]:


scores2 = do_cross_validation_mixed_features(resnet_mixed_features,
        img_feat=feature_df.resized_cropped_img, other_feat=feature_df[visual_features],
                                    Y=feature_df.label, n_splits = 5)
#clear_output()
print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))


# In[ ]:


import os
 
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
 
def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
 
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)
 
    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]
 
    # request probability estimation
    svm = SVC(probability=True)
 
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=10, n_jobs=4, verbose=3)
 
    clf.fit(X_train, y_train)
 
    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))
 
    print("\nBest parameters set:")
    print(clf.best_params_)
 
    y_predict=clf.predict(X_test)
 
    labels=sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))
 
    print("\nClassification report:")
    print(classification_report(y_test, y_predict))


# In[25]:


#resnet_model = resnet_with_imagenet()

#resnet_feat = resnet_model.predict(np.array([feature_df.resized_cropped_img[5]]),batch_size=1)
#get_resnet_feature = get_feature_layer_func(model=resnet_model)

# resnet_feat = get_resnet_feature([np.array([feature_df.resized_cropped_img[5]])])
# resnet_feat[0].shape


# In[37]:


# '''Feed CNN features directly into SVM'''
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, cnn_feats, Y, cv=5)
# print("Linear kernel accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2*100))

# clf = svm.SVC(kernel='rbf', C=1)
# scores = cross_val_score(clf, cnn_feats, Y, cv=5)
# print("RBF kernel accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2*100))


# In[ ]:


# '''Confusion matrix'''
# #(Xtrain, Ytrain),(Xtest,Ytest) = split_data(cnn_feats, Y)
# clf = svm.SVC(kernel='linear', C=1)
# clf.fit(Xtrain,Ytrain)
# predictions = clf.predict(Xtest)
# print(confusion_matrix(Ytest, predictions))


# In[138]:



# '''Feed CNN features directly into SVM'''
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, np.array(
#     [v for v in all_feat_df['cnn_feat_transformed'].values]).reshape(437, 131072), all_feat_df.label, cv=5)
# print("Linear kernel accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2*100))

# clf = svm.SVC(kernel='rbf', C=1)
# scores = cross_val_score(clf, np.array(
#     [v for v in all_feat_df['cnn_feat_transformed'].values]).reshape(437, 131072), all_feat_df.label, cv=5)

# print("RBF kernel accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2*100))


# In[146]:


# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, np.array(
#     [v for v in all_feat_df['combined_feat'].values]).reshape(437, 131108), all_feat_df.label, cv=5)
# print("Linear kernel accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2*100))

# clf = svm.SVC(kernel='rbf', C=1)
# scores = cross_val_score(clf, np.array(
#     [v for v in all_feat_df['combined_feat'].values]).reshape(437, 131108), all_feat_df.label, cv=5)

# print("RBF kernel accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2*100))


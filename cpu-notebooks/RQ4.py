
# coding: utf-8

# In[6]:


# generic imports
import sys
import os
sys.path.append('../')
import helper
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
# data structure
import itertools
from collections import Counter,defaultdict

#stats amd ml
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm #https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.LogitResults.html
from scipy import stats
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import roc_curve

pipa_path = '/nfs/juhu/data/rakhasan/pipa-dataset/'
openImg_path = '/nfs/juhu/data/rakhasan/bystander-detection/google-img-db/'
survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
survey_photo_path = survey_path+'/photos/'


# In[2]:


high_level_concepts = ['was_aware', 'posing','comfort', 'will', 'photographer_intention', 'replacable', 'photo_place']
high_level_concepts_num = [c+'_num' for c in high_level_concepts]

human_parts,object_classes =pickle.load(open(survey_path+'human-object-classes.pkl','rb'))
survey_photo_annotations =pickle.load(open(survey_path+'survey_photo_annotations.pkl','rb'))
body_features = ['_'.join(human_parts[body_part].split()) for body_part in human_parts.keys()]

visual_features = ['person_distance', 'person_size', 'num_people'] + body_features

photo_df = pickle.load(open(os.path.join(survey_path, 'photo_df.csv'), 'rb'))
mapping = pickle.load(open(survey_path +'mappings_pilot2','rb'))
feature_df = pickle.load(open(os.path.join(survey_path, 'features-df.pkl'), 'rb'))
feature_df.shape, len(set(photo_df.index.values))


# In[3]:


binary_data = feature_df[(feature_df.label==1) | (feature_df.label==-1)]
binary_data['label'] = binary_data.apply(lambda row: 1 if row.label==1 else 0, axis=1)


# In[4]:


'''use predictors one by one'''
dicts = []
for pred in visual_features:
    model = helper.test_logit(binary_data, [pred])
    print(model.summary())
    d = helper.get_model_summary(model)
    d['Predictor'] = pred
    ors = helper.get_OR(model)
    d['OR'] = ors.loc[pred].OR
    d['2.5%'] = ors.loc[pred]['2.5%']
    d['97.5%'] = ors.loc[pred]['97.5%']
    dicts.append(d)
    
pd.DataFrame(dicts)[['Predictor', 'OR', '2.5%', '97.5%','Chi^2', 'p(Chi^2)', 'R^2']].set_index("Predictor").round(3)


# In[12]:


for i in tqdm(range(len(mapping))):
    m = mapping[i]
    img = cv2.imread(survey_photo_path+str(i)+'.jpg')
    cropped = helper.crop_img(img, [m[4:8]], pad=0)[0]
    cv2.imwrite(survey_path+'cropped-photos/'+str(i)+'.jpg',cropped)


from __future__ import print_function

import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import gzip
import scipy.stats as stats
from collections import defaultdict
import datetime
import time
import operator
from time import mktime
import matplotlib.dates as mdates
import math
from scipy.stats import gaussian_kde
import pandas as pd
import pickle
import scipy.stats.mstats as mstats
import statsmodels.api as sm
import scipy.stats 
import random
from scipy.stats.mstats import kruskalwallis

import os
import operator
import time 

from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc,accuracy_score
from scipy import interp
#from sklearn import cross_validation, grid_search

import cv2
import operator
import numpy as np

import matplotlib.image as mpimg

from tqdm import tqdm
import seaborn as sns

labelFont = 18
legendFont = 16
tickFont = 16

manuscriptColSize= 252
forLatex = True
latexDec = 6

from keras import optimizers

def get_fig_size(figWidthPt):
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (math.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = figWidthPt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean       # height in inches
    return (fig_width,fig_height)
    
def get_photos_with_n_people(num_person):
    '''
    Get paths of randomly selected sample_size photos containing 
    num_people persons in each
    '''
    anno = open(path+'/annotations/index.txt','r').readlines()
    d = defaultdict(list)
    for a in anno:
        tokens = a.split()
        d[tokens[0]+'_'+tokens[1]].append(
            (float(tokens[2]), float(tokens[3]),float(tokens[4]),float(tokens[5]))
            )
    
    photos = [k for k in d.keys() if len(d[k])==num_person]
    
    output=dict()
    for p in photos:
        if os.path.isfile(path+'train/'+p+'.jpg'):
            output[path+'train/'+p+'.jpg'] = d[p]
        elif os.path.isfile(path+'test/'+p+'.jpg'):
            output[path+'test/'+p+'.jpg'] = d[p]
        elif os.path.isfile(path+'val/'+p+'.jpg'):
            output[path+'val/'+p+'.jpg'] = d[p]
            
    return output
    


def draw_photos(photos, col_size=3, title=''):
    row_num =  math.ceil(len(photos)/col_size)
    fig, axes = plt.subplots(nrows = row_num, ncols = col_size, figsize=(row_num*6, 20), squeeze=False)
    print(row_num, col_size)
    c=0
    for i in range(0,row_num):
        for j in range(0, col_size):
            axes[i][j].imshow(photos[c])
            c+=1
            if c>= len(photos):
                break
                
    plt.suptitle(title, size=16)
    plt.tight_layout()
    plt.show()


def draw_photos_from_path(photo_paths, col_size=5, title=''):
    row_num =  math.ceil(len(photo_paths)/col_size)
    fig, axes = plt.subplots(nrows = row_num, ncols = col_size, figsize=(row_num*6, 20), squeeze=False)
    print(row_num, col_size)
    c=0
    for i in range(0,row_num):
        for j in range(0, col_size):
            img = mpimg.imread(photo_paths[c])
            axes[i][j].imshow(img)
            axes[i][j].set_title(os.path.basename(photo_paths[c]))
            
            c+=1
            if c>= len(photo_paths):
                break
                
    plt.suptitle(title, size=16)
    plt.tight_layout()
    plt.show()
    
def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect, compare their sizes
        a1 = math.fabs(x1-x1b) * math.fabs(y1-y1b)
        a2 = math.fabs(x2-x2b) * math.fabs(y2-y2b)
        if a1>a2:
            return a1/a2
        return a2/a1
    
def sort_photos_by_x_variance(photos):
    ordered = dict()
    for p in photos.keys():
        if len(photos[p]) <= 1: # less than two people, ignore
            continue
        x_var = np.var([v[0] for v in photos[p]])
        ordered[p] = x_var
    return [k[0] for k in sorted(ordered.items(), key=operator.itemgetter(1), reverse=True)]
        
def sort_photos_by_y_variance(photos):
    ordered = dict()
    for p in photos.keys():
        if len(photos[p]) <= 1: # less than two people, ignore
            continue
        y_var = np.var([v[1] for v in photos[p]])
        ordered[p] = y_var
    return [k[0] for k in sorted(ordered.items(), key=operator.itemgetter(1), reverse=True)]

def sort_photos_by_w_variance(photos):
    ordered = dict()
    for p in photos.keys():
        if len(photos[p]) <= 1: # less than two people, ignore
            continue
        w_var = np.var([v[2] for v in photos[p]])
        ordered[p] = w_var
    return [k[0] for k in sorted(ordered.items(), key=operator.itemgetter(1), reverse=True)]

def sort_photos_by_h_variance(photos):
    ordered = dict()
    for p in photos.keys():
        if len(photos[p]) <= 1: # less than two people, ignore
            continue
        h_var = np.var([v[3] for v in photos[p]])
        ordered[p] = h_var
    return [k[0] for k in sorted(ordered.items(), key=operator.itemgetter(1), reverse=True)]

def sort_photos_by_dist_variance(photos):
    ordered = dict()
    
    for p in photos.keys():
        if len(photos[p]) <= 1: # less than two people, ignore
            continue
        
        dist = 0
        area = 0
        rectangles = photos[p]
        for i in range(1, len(rectangles)):
            x1 = rectangles[i-1][0]
            y1 = rectangles[i-1][1]
            w1 = rectangles[i-1][2]
            h1 = rectangles[i-1][3]
            
            x2 = rectangles[i][0]
            y2 = rectangles[i][1]
            w2 = rectangles[i][2]
            h2 = rectangles[i][3]
            
            d =  rect_distance(x1, y1, x1+w1, x2+w2, x2, y2, x2+w2, y2+w2)
            
            area1= w1*h1
            area2 = w2*h2
            
            d = dist/(2*area1*area2/(area1**2+area2**2))
            if dist <d:
                dist = d

        ordered[p] = dist
    return [k[0] for k in sorted(ordered.items(), key=operator.itemgetter(1), reverse=True)]


def sort_photos_by_size_variance(photos):
    ordered = dict()
    for p in photos.keys():
        if len(photos[p]) <= 1: # less than two people, ignore
            continue
        s_var = np.var([v[2]*v[3] for v in photos[p]])
        ordered[p] = s_var
    return [k[0] for k in sorted(ordered.items(), key=operator.itemgetter(1), reverse=True)]

def draw_rect(img, rectangles, c=(0,255,0)):
    for x,y,w,h in rectangles:
        cv2.rectangle( img, (int(x), int(y)), (int(x+w), int(y+h)), color = c, thickness = 3 )
    return img



def get_person_size(img, r):
    '''
    Returns size of a person (both absolute and percentage) in a photo given the photo or its path and 
    rectangle 'r' containing the person.
    '''
    if isinstance(img, str):
        img= mpimg.imread(img)
        
    img_w, img_h = len(img[0]), len(img)
    xmin, xmax, ymin, ymax = float(r[0])*img_w, float(r[1])*img_w,float(r[2])*img_h,float(r[3])*img_h
    
    size = (float(xmax)-float(xmin))* (float(ymax)-float(ymin))
    return size, size/(img_w * img_h)

def get_distance_from_center(img, r,
        center_zero = True, norm_type = None, draw=False):
    '''
    Returns distance between a bounding box center and the image center.
    Optionally the distance will be overlaid on the image if draw=True

    Parameters:
        img: either mpi.img object or path to an image file
        r: the bounding box rectangle
        center_zero: if True and the rectangle overlapps with the photo center, then distance is zero
        norm_type: indicates how the distance is normalized. 
            None: not normalized
            Img_size: total distance divided by image area
            Axis_size: distance along x-axis (y-axis) is divided by the image length (height)
    '''
    if isinstance(img, str):
        img = mpimg.imread(img)
    img_w = len(img[0])
    img_h = len(img)
    xmin, xmax, ymin, ymax = float(r[0])*img_w, float(r[1])*img_w,float(r[2])*img_h,float(r[3])*img_h

    #center of the image
    cx = int(len(img[0])/2)
    cy = int(len(img)/2)
    
    #center of the rectangle
    x=int(xmin+(xmax-xmin)/2)
    y=int(ymin+(ymax-ymin)/2)
    #print(x,y,cx,cy)
    
    #if img center is contained in the rectangle
    if center_zero and (cx>=xmin and cx<=xmax and cy>=ymin and cy<=ymax):
        dist=0
    else:
        dist = math.sqrt((cx-x)**2+(cy-y)**2)
        if norm_type == 'Img_size':
            dist = dist/((xmax-xmin)*(ymax-ymin))
        elif norm_type == 'Axis_size':
            #print((cx-x)/img_w)
            #print(((cy-y)/img_h))
            dist = math.sqrt( ((cx-x)/img_w)**2 + ((cy-y)/img_h)**2 )
        
    if draw:
        cv2.circle(img, (int(cx),int(cy)),  5, (255,255,0), -1)
        cv2.circle(img, (int(x),int(y)),  5, (255,255,0), -1)
        cv2.line(img, (cx,cy),(x,y),(255,0,0),10)

    return dist,img


def data_stat(photo_df):
    total = len(photo_df.index.unique())
    print('Total:', total)
    a=photo_df.subject_bystander_num.groupby('photo_no').mean()
    print('Subject:{}, bystander:{}, neither:{}'.format( len(a[(a>0) & (a<=1)]), 
        len(a[(a<0) & (a>=-1)]), len(a[a==0])))
    print('Definitely subject:{}, Definitely bystander:{}\n'.format(len(a[a>1]), len(a[a<-1])))
    
    print('Subject: {} ({:.2f}%), Bystander: {} ({:.2f}%), Neither:{}({:.2f}%)'.format(
        len(a[a>0]), len(a[a>0])*100/total, len(a[a<0]), len(a[a<0])*100/total, len(a[a==0]), len(a[a==0])*100/total))

def prepare_svm_data(photo_df, features, train_perc = .8):
    
    m = photo_df.groupby('photo_no').mean()
    a = m.subject_bystander_num
    pos = list(a[a>0].index.values)
    neg = list(a[a<0].index.values)
    
    train_idx = pos[:int(len(pos)*train_perc)] + neg[:int(len(neg)*train_perc)]
    test_idx = pos[int(len(pos)*train_perc):] + neg[int(len(neg)*train_perc):]
    
    xtrain = m.loc[train_idx][features].values
    ytrain = m.loc[train_idx].subject_bystander_num.values
    ytrain= np.greater(ytrain,0).astype(int)

    xtest = m.loc[test_idx][features].values
    ytest = photo_df.loc[test_idx].subject_bystander_num.values
    ytest= np.greater(ytest,0).astype(int)
    return xtrain, ytrain, xtest, ytest

def plot_corr_matrix(mat, ticks=None, title='', ticks_fontsize=16):
    '''
    Plot a correlation matrix
    '''
    #%matplotlib inline
    sns.set(style="white")
    mask = np.zeros_like(mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    if ticks==None:
        ticks=mat.columns.values
    fix, ax =plt.subplots(figsize=(12,8))
    sns.heatmap(mat, mask=mask, center=0, cmap=cmap,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=ticks, yticklabels=ticks)
    plt.tick_params(axis='both', which='major', labelsize= ticks_fontsize)
    plt.tick_params(axis='x', which='major', rotation=70)
    plt.title(title, fontsize=20)
    plt.show()

def find_img_path(root, imgId):
    '''Find out in which train_0x folder of google image directory the image with imgId is contained.'''
    for i in range(9): 
        cur_dir = root+'home/ec2-user/train_0'+str(i)+'/'
        files = os.listdir(cur_dir)
        if imgId in files:
            return cur_dir+imgId
    return None


def rect_in_rect(rect1, rect2, overlap_thresh=100):
    '''
    Returns true if the first rectangle is overlap_thresh% inside of a second
    Rectangle in the form (x1,y1,x2,y2)
    '''    
    
    #annotated rectangles are in string format, convert to float
    rect1 = np.array(rect1).astype(float) 
    rect2 = np.array(rect2).astype(float)
    
    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]) 
    
    overlap_area = math.fabs(dx * dy)
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    overlap_perc = (overlap_area * 100) /rect1_area
    return  overlap_perc >= overlap_thresh



def draw_normed_rect(img, annotations, c=(0,255,0),for_objects=None):
    if for_objects:
        rectangles = [a[4:8] for a in annotations if a[2] in for_objects]
    else:
        rectangles = [a[4:8] for a in annotations]
    for r in rectangles:
        xmin, xmax, ymin, ymax = float(r[0])*len(img[0]), float(r[1])*len(img[0]),float(r[2])*len(img),float(r[3])*len(img)
        #print( xmin, xmax, ymin, ymax)
        cv2.rectangle( img, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)), color = c, thickness = 3 )
    return img

def draw_rect_r(img, rectangles, c=(0,255,0)):
    #print(len(rectangles))
    for r in rectangles:
        xmin, xmax, ymin, ymax = r[0],r[1],r[2],r[3]
        #print( xmin, xmax, ymin, ymax)
        cv2.rectangle( img, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)), color = c, thickness = 3 )
    return img

def get_normed_rect(img, annotations, for_objects=None):
    if for_objects:
        rectangles = [a[4:8] for a in annotations if a[2] in for_objects]
    else:
        rectangles = [a[4:8] for a in annotations]

    rects=[]
    for r in rectangles:
        xmin, xmax, ymin, ymax = float(r[0])*len(img[0]), float(r[1])*len(img[0]),float(r[2])*len(img),float(r[3])*len(img)
        rects.append([int(xmin), int(xmax),int(ymin), int(ymax)])
    return rects

def draw_photos_with_rectangle(photos, title='', col_size=1, for_objects = None, color=(0,255,0)):
    '''
    Draw photos after overlaying rectangles on them.
    Parameters:
        photos: dictionary with photo file paths as keys and rectangles as values.
    '''
    out = []
    for s in photos:
        img = mpimg.imread(s)
        out.append(draw_normed_rect(img, photos[s], for_objects=for_objects, c= color))
       
    draw_photos(out, col_size=col_size, title=title)


def test_logit(data, predictors, label='label', normalize=True):
    X = data[predictors]
    if normalize:
        X = X.apply(stats.zscore)
    X = sm.add_constant(X)
    y = data[label]
    return sm.Logit(endog=y, exog=X).fit(disp = False)

def print_chisq(model):
    print('\nChisq:{:.2f}, p:{:.2f}\n'.format(model.llr, model.llr_pvalue))
    
def get_Rsq(model):
    return (model.llr) / (- 2*model.llnull)

def get_model_summary(model):
    summary = dict()
    summary['Chi^2'] = model.llr
    summary['p(Chi^2)'] = model.llr_pvalue
    summary['R^2'] = (model.llr) / (- 2*model.llnull)
    return summary

def get_OR(model):
    output = model.conf_int()
    output['OR'] = model.params
    output.columns = ['2.5%', '97.5%', 'OR']
    return np.exp(output)[['OR', '2.5%', '97.5%']]

def get_null_rows(df):
    null_columns=df.columns[df.isnull().any()]
    return df[df.isnull().any(axis=1)][null_columns]


def mask_img(img, rectangles, pad=10):
    '''Blacken image except the regions inside the rectangles. Note: load image using cv2, not mpimg'''
    
    out_imgs = []
    for r in rectangles:
        out_img = np.zeros(img.shape)
        xmin, xmax, ymin, ymax = float(r[0])*len(img[0]), float(r[1])*len(img[0]),float(r[2])*len(img),float(r[3])*len(img)
        xmin = max(0,xmin-(xmax-xmin)*pad/100)
        xmax = min(len(img[0]), xmax+(xmax-xmin)*pad/100)
        ymin = max(0,ymin-(ymax-ymin)*pad/100)
        ymax = min(len(img), ymax+(ymax-ymin)*pad/100) 
        out_img[int(ymin):int(ymax), int(xmin):int(xmax), :] = img[int(ymin):int(ymax), int(xmin):int(xmax), :] 
        out_imgs.append(out_img.astype(int))
    return out_imgs

def crop_img(img, rectangles, pad=10):
    '''Crop rectangles from image. Note: load image using cv2, not mpimg'''
    out_imgs = []
    for r in rectangles:
        xmin = int(float(r[0])*len(img[0]))
        xmax = int(float(r[1])*len(img[0]))
        ymin = int(float(r[2])*len(img))
        ymax = int(float(r[3])*len(img))
        xmin, xmax, ymin, ymax = max(0,xmin-pad), min(len(img[0]), xmax+pad), max(0,ymin-pad), min(len(img), ymax+pad) 
        out_imgs.append(img[ymin:ymax, xmin:xmax, :])
    return out_imgs

def binary_mask(img, rectangles, pad=10):
    '''Blacken image except the regions inside the rectangles. Note: load image using cv2, not mpimg'''
    
    out_imgs = []
    for r in rectangles:
        out_img = np.zeros(img.shape)
        xmin, xmax, ymin, ymax = float(r[0])*len(img[0]), float(r[1])*len(img[0]),float(r[2])*len(img),float(r[3])*len(img)
        xmin, xmax, ymin, ymax = max(0,xmin-pad), min(len(img[0]), xmax+pad), max(0,ymin-pad), min(len(img), ymax+pad) 
        out_img[int(ymin):int(ymax), int(xmin):int(xmax), :] = 255 
        out_imgs.append(out_img.astype(int))
    return out_imgs

def do_cross_val_roc(X, y, classifier_func, n_splits=10, 
                     label_font = 28, legend_font=16, 
                     show_plot=True, save_file=None,
                    forLatex=False):
    '''
    Cross validation for a Logistic Regression model and plot ROC.    
    This was copied from the helper, to change plotting
    '''
    random_state = np.random.RandomState(0)
    cv_scores = []
    #
    cv = StratifiedKFold(n_splits=n_splits)
    #
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #
    fig = plt.figure()
    if(forLatex):
        print('setting latex size')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif='Times')
        fig.set_size_inches(get_fig_size(manuscriptColSize))
    else:
        fig.set_size_inches((12,8))
#
    i = 0
    for train, test in cv.split(X, y):
        trained_classifier = classifier_func().fit(X[train], y[train])
        trained_classifier.random_state = random_state
        trained_classifier.probability=True
        predictions = trained_classifier.predict(X[test])
        cv_scores.append(accuracy_score(y_pred=predictions, y_true=y[test]))
        print(classification_report(y_pred=predictions, y_true=y[test]))
        
        probas_ = trained_classifier.predict_proba(X[test])
        # Compute ROC curve and area of the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=None)
                 #label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label=None,#'Chance', 
             alpha=.8)
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
    #plt.text(.15,.03, 'Mean ROC (AUC = %0.2f $\pm$ %0.2f)'% (mean_auc, std_auc))
    plt.legend(fontsize=legendFont-forLatex*latexDec-2, markerscale=0.4, loc="lower right")
#             handletextpad=.01, bbox_to_anchor=(-0.03, 1.02),
#           ncol=1, frameon=True#,fancybox=True, shadow=True
               
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=3)
    if show_plot:
        plt.show()
    #
    print('Accuracy scores:',cv_scores)
    print('Mean accuracy:{}(+/-{:.2f})'.format(np.mean(cv_scores), np.std(cv_scores)))

    
def plot_roc(model, xtrain, xtest, ytrain, ytest, save_file=None):
    '''
    Given a model and train/test split, plot ROC curve.
    '''
    model = model.fit(xtrain, ytrain)
    
    probs = model.predict_proba(xtest)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(ytest, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right', fontsize=14)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()

def fine_tune_resnet_roc(X, y, classifier_func, n_splits=10, epochs=30, batch_size=256, verbose=1,
                     label_font = 18, legend_font=14, show_plot=True, save_file=None):
    '''
    Fine tune ResNet model with raw cropped images. Do cross-validation and plot ROC.
    X is a numpy array containing the cropped images
    '''
    
    random_state = np.random.RandomState(0)
    cv_scores = []
    
    cv = StratifiedKFold(n_splits=n_splits)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(16,10))

    i = 0
    for train, test in cv.split(X, y):
        model = classifier_func()
        model.random_state = random_state
        model.probability=True
        
        model.fit(X[train], y[train], 
                  epochs=epochs, batch_size=batch_size, verbose=verbose)
        
#         predictions = model.predict(X[test])
#         cv_scores.append(accuracy_score(y_pred=predictions, y_true=y[test]))
#         print(classification_report(y_pred=predictions, y_true=y[test]))
        
        probas_ = model.predict(X[test])
        # Compute ROC curve and area of the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
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
    plt.xlabel('False Positive Rate', fontsize=label_font)
    plt.ylabel('True Positive Rate', fontsize=label_font)
    plt.legend(loc="lower right", fontsize=legend_font)
    if save_file:
        plt.savefig(save_file, dpi=3)
    if show_plot:
        plt.show()
from __future__ import print_function

import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
import os
import math 

from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc,accuracy_score
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from scipy import interp
from scipy import stats


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


def do_cross_val_roc(X, y, classifier_func, n_splits=10, 
                     show_plot=True, save_file=None,
                        forLatex=False,
                        square=False #whether the plot is square or rectangular
                    ):
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
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    fig= plt.figure(facecolor='w')
    plt.fill(False)
    if(forLatex):
        print('setting latex size')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif='Times')
        w,h = get_fig_size(manuscriptColSize)
        fig_size = ((w, w if square else h))
        fig.set_size_inches(fig_size)
    else:
        fig.set_size_inches((12,8))
#
    i = 0
    for train, test in cv.split(X, y):
        print('Fold: ', i)
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
        plt.plot(fpr, tpr, lw=1, alpha=0.3, 
                 label=None)#'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', 
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
                     label=r'$\pm$ 1 std. dev.')
#
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=labelFont-forLatex*latexDec-2)
    plt.ylabel('True Positive Rate', fontsize=labelFont-forLatex*latexDec-2)
    #plt.text(.15,.03, 'Mean ROC (AUC = %0.2f $\pm$ %0.2f)'% (mean_auc, std_auc))
    plt.legend(fontsize=legendFont-forLatex*latexDec-2, markerscale=0.4, loc="lower right", frameon=True)
#             handletextpad=.01, bbox_to_anchor=(-0.03, 1.02),
#           ncol=1, frameon=True#,fancybox=True, shadow=True
       
    #fig.patch.set_color('w')
    fig.set_facecolor("w")
    
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=300, frameon=True, facecolor=fig.get_facecolor(), edgecolor='black', transparent=True)
    if show_plot:
        plt.show()
    #
    print('Accuracy scores:',cv_scores)
    print('Mean accuracy:{}(+/-{:.2f})'.format(np.mean(cv_scores), np.std(cv_scores)))

def fine_tune_resnet_roc(X, y, classifier_func, n_splits=10, epochs=30, batch_size=256, verbose=1,
                      show_plot=True,forLatex=False, square=False, save_file=None):
    '''
    Fine tune ResNet model with raw cropped images. Do cross-validation and plot ROC.
    X is a numpy array containing the cropped images
    '''
    #
    random_state = np.random.RandomState(0)
    cv_scores = []
    #
    cv = StratifiedKFold(n_splits=n_splits)
    #
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    fig= plt.figure(facecolor='w')
    plt.fill(False)
    if(forLatex):
        print('setting latex size')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif='Times')
        w,h = get_fig_size(manuscriptColSize)
        fig_size = ((w, w if square else h))
        fig.set_size_inches(fig_size)
    else:
        fig.set_size_inches((12,8))
#
    i = 0
    for train, test in cv.split(X, y):
        print('\n\n***********Split {} of {}**************\n\n'.format(i, n_splits))
        model = classifier_func()
        model.random_state = random_state
        model.probability=True
        #
        model.fit(X[train], y[train], 
                  epochs=epochs, batch_size=batch_size, verbose=verbose)        
        probas_ = model.predict(X[test])
        # Compute ROC curve and area of the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label=None)#'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
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
                     label=r'$\pm$ 1 std. dev.')
#
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=labelFont-forLatex*latexDec-2)
    plt.ylabel('True Positive Rate', fontsize=labelFont-forLatex*latexDec-2)
    plt.legend(fontsize=legendFont-forLatex*latexDec-2, markerscale=0.4, loc="lower right")
#   
    fig.set_facecolor("w")
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=300, frameon=True, facecolor=fig.get_facecolor(), edgecolor='black', transparent=True)
    if show_plot:
        plt.show()
        
    print('Accuracy scores:',cv_scores)
    print('Mean accuracy:{}(+/-{:.2f})'.format(np.mean(cv_scores), np.std(cv_scores)))
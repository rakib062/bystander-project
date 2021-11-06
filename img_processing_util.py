
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
import matplotlib.dates as mdates
import math
import random
import cv2
import os

import matplotlib.image as mpimg

from tqdm import tqdm
import seaborn as sns

    
def get_photos_with_n_people_genome(num_person):
    '''
    Get paths of randomly selected sample_size photos containing 
    num_people persons in each.
    Works only for VisualGenome dataset.
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

def get_distance_from_center(img, r, rel_rect=True,
        center_zero = True, norm_type = None, draw=False):
    '''
    Returns distance between a bounding box center and the image center.
    Optionally the distance will be overlaid on the image if draw=True

    Parameters:
        img: either mpi.img object or path to an image file
        r: the bounding box rectangle, can be relative to the img w/h or can be absolute coordinates.
        rel_rect: indicates whether the rectangle coordinates are relative values.
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
        xmin = int(float(r[0]))#*len(img[0]))
        xmax = int(float(r[1]))#*len(img[0]))
        ymin = int(float(r[2]))#*len(img))
        ymax = int(float(r[3]))#*len(img))
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
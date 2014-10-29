# -*- coding: utf-8 -*-
"""
Calculate morphological shape features from grayscale images.
    
Created on Fri Aug 30 15:20:49 2013

@author: John Quinn <jquinn@cit.ac.ug>
"""
import cv2
import ctypes
import numpy as np

def patchextract(img, size, step, attributes=None, 
                 filters=None, centiles=[10,30,50,70,90]):
    '''
    Split up a large image into square overlapping patches, and return all 
    features for each patch.
    '''
    height, width = img.shape
    nattributes = len(attributes)
    ncentiles = len(centiles)
    features = []
    y = step
    while y<height:
        x = step;
        while (x<width):
            left = x-(size/2)
            right = x+(size/2)
            top = y-(size/2)
            bottom = y+(size/2)
            patch = img[top:bottom,left:right]
            currentfeatures = extract(patch, attributes, filters)
            
            # Now calculate centiles of each attribute
            histfeatures = np.zeros((nattributes,ncentiles))
            for a in range(nattributes):
                bagsize = currentfeatures.shape[0]
                if bagsize>1:
                    histfeatures[a,:] = np.percentile(currentfeatures[:,a],list(centiles))      
                elif bagsize==1:
                    histfeatures[a,:] = np.ones(len(centiles))*currentfeatures[0,a]
            features.append(histfeatures.ravel())
            x+=step
        y+=step
        
    return np.vstack(features)
    
def patchextractbags(img, size, step, attributes=None, 
                 filters=None):
    '''
    Split up a large image into square overlapping patches, and return all 
    features for each patch.
    '''
    height, width = img.shape
    
    features = []
    y = step
    while y<height:
        x = step;
        while (x<width):
            left = x-(size/2)
            right = x+(size/2)
            top = y-(size/2)
            bottom = y+(size/2)
            patch = img[top:bottom,left:right]
            currentfeatures = extract(patch, attributes, filters)
            features.append(currentfeatures)
            
            x+=step
        y+=step
            
    instances = np.vstack(features)
    numbags = len(features)
    bags = []
    
    currentidx=0
    for i in range(numbags):
        numinstancesinthisbag = features[i].shape[0]
        bags.append(np.arange(currentidx, currentidx+numinstancesinthisbag))
        currentidx += numinstancesinthisbag
    
    return instances, bags
    
def extract_hist(img, attributes=None, filters=None, centiles=[25,50,75]):
    nattributes = len(attributes)
    ncentiles = len(centiles)  
    features = extract(img, attributes, filters)
    histfeatures = np.zeros((nattributes,ncentiles))
    for a in range(nattributes):
        bagsize = features.shape[0]
        if bagsize>1:
            histfeatures[a,:] = np.percentile(features[:,a],list(centiles))      
        elif bagsize==1:
            histfeatures[a,:] = np.ones(len(centiles))*features[0,a]
    return histfeatures.ravel()
    
def extract(img, attributes=None, filters=[], centiles=[10,25,50,75,90]):
    '''
    Extract all the shape features for a given image.
    
    Attributes can be either None (default, calculate everything), or some
    list of the following integer values.
    
    0: Area
    1: Area of min. enclosing rectangle
    2: Square of diagonal of min. enclosing rectangle
    3: Cityblock perimeter
    4: Cityblock complexity (Perimeter/Area)
    5: Cityblock simplicity (Area/Perimeter)
    6: Cityblock compactness (Perimeter^2/(4*PI*Area))
    7: Large perimeter
    8: Large compactness (Perimeter^2/(4*PI*Area))
    9: Small perimeter
    10: Small compactness (Perimeter^2/(4*PI*Area))
    11: Moment of Inertia
    12: Elongation: (Moment of Inertia) / (area)^2
    13: Mean X position
    14: Mean Y position
    15: Jaggedness: Area*Perimeter^2/(8*PI^2*Inertia)
    16: Entropy
    17: Lambda-max (Max.child gray level - current gray level)
    18: Gray level
    '''

    assert(len(img.shape)==2)
    assert(img.dtype=='uint8')
    
    img_scaled_vals = img/2  # the feature extraction code assumes vals in range 0-127
    
    features = np.zeros((1,len(attributes)))
    
    minmax = cv2.minMaxLoc(img_scaled_vals)
    if minmax[1]-minmax[0]>5:
        if attributes == None:
            attributes = range(19)
        allfeatures = []
        for attr in attributes:
            f = _singleattribute(img_scaled_vals, attr)
            allfeatures.append(f)
        features = np.vstack(allfeatures).transpose()
        
        validinstances = np.ones(features.shape[0])>0
        for filt in filters:
            att = filt[0]
            thresh = filt[2]
            if att in attributes:
                idx = attributes.index(att)
                if filt[1]=='<':
                    valid = features[:,idx] < thresh
                else:
                    valid = features[:,idx] >= thresh                
            validinstances = np.logical_and(validinstances, valid)  
        features = features[validinstances,:]
        
    return features

def _singleattribute(img, attribute):
    extractFeatures = ctypes.cdll.LoadLibrary('./extractFeatures.so')
    extractFeatures.MaxTreeAttributes.restype = ctypes.POINTER(ctypes.c_float) 
    extractFeatures.MaxTreeAttributes.argtype = [ctypes.c_int,
                                                 ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_ubyte),
                                                 ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_int)]
    count = (ctypes.c_int)(-1)
    imgvec = img.ravel()
    out = extractFeatures.MaxTreeAttributes(img.shape[0], img.shape[1],
                                            imgvec.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                            attribute,
                                            ctypes.byref(count))
    x = out[0:count.value]    
    return np.array(x)

if __name__=='__main__':
    img = cv2.imread('test.pgm', cv2.IMREAD_GRAYSCALE)
    features = extract(img)
    print features

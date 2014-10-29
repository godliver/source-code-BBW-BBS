'''
Created on Sep 29, 2013

@author: godliver
'''
import cv2, shapefeatures
import numpy as np
bins = 3

'''import shapefeatures, cv2
img = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE)
x = shapefeatures.extract_hist(img, attributes=[1,2,3])'''


def histogram(img):
	# Load an image
	img_bw = cv2.imread(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	x = shapefeatures.extract_hist(img_bw, attributes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], filters=[])
	#print x
	return x

def save_histogram(healthy,bbw,bbs):
    vals = []
    shape = []
        
    for health in healthy:
        #Append 1 to mean healthy type of image
        vals.append(1)
        vals.append(1)
        vals.append(1)
        
        #Append histograms for healthy images to a list for each color space
        shape.extend(histogram(health))
        
    for bbw in bbw:
        #Append 2 to mean diseased type of image
        vals.append(2)
        vals.append(2)
        vals.append(2)
        
        #Append histograms for diseased images to a list for each color space
        shape.extend(histogram(bbw))
    
    for bbs in bbs:
        #Append 3 to mean diseased type of image
        vals.append(3)
        vals.append(3)
        vals.append(3)
        
        #Append histograms for unkwon images to a list for each color space
        shape.extend(histogram(bbs))
    np.savetxt('shape.txt', shape, delimiter=',')
    np.savetxt('vals.txt', vals, delimiter=',')
    

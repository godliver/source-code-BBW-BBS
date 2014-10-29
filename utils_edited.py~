'''
Created on Sep 29, 2013

@author: godliver
'''
import cv2, shapefeatures
import numpy as np
bins = 50
def histogram(img_path, color_space="hsv", number_of_bins=50):
	#Load an image
	hist_item = None
	img_bw = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	# Calculate the mask
	ret, mask = cv2.threshold(img_bw, 254, 255, cv2.THRESH_BINARY_INV)
	colors = []
	histograms = []  # Holds histograms
	# Process HSV Image
	if color_space == "hsv":
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		colors = [[0, 180], [0, 255], [0, 255]]
	else: 
		# Process BGR Image
		if color_space == "bgr":
		    img = cv2.imread(img_path)
		    colors = [[0, 255], [0, 255], [0, 255]]
		# Process LBA Image
		elif color_space == "lba":
		    img = cv2.imread(img_path, cv2.COLOR_BGR2LAB)
		    colors = [[0, 100], [-127, 127], [-127, 127]]
		else:
		    print "Unkwon color space"
	
	
	#hist_item_1, hist_item_2 and hist_item_1 hold histograms color components e.g for h,s,v repectively
	hist_item_1 = cv2.calcHist([img], [0], mask, [number_of_bins], colors[0])
	hist_item_2 = cv2.calcHist([img], [1], mask, [number_of_bins], colors[1])
	#hist_item_3 = cv2.calcHist([img], [2], mask, [number_of_bins], colors[2])
	
	cv2.normalize(hist_item_1, hist_item_1, 0, 255, cv2.NORM_MINMAX)
	cv2.normalize(hist_item_2, hist_item_2, 0, 255, cv2.NORM_MINMAX)
	#cv2.normalize(hist_item_3, hist_item_3, 0, 255, cv2.NORM_MINMAX)
	
	#load in the shape features that were chosen as more important
	x = shapefeatures.extract_hist(img_bw, attributes=[12,9,2,10,11,1], filters=[])
		
	
	col1 = hist_item_1.flatten()
	col2 = hist_item_2.flatten()
	#col3 = hist_item_3.flatten()
	
	#this merges colour + shape together
	vec = np.hstack((col1,col2,x))
	return vec

def save_histogram(healthy, bbw, bbs, number_of_bins=50):
    type_file = []
    
    hist_vals_h = []
    hist_vals_b = []
    hist_vals_l = []
    
    for health in healthy:
        # Append 1 to mean healthy type of image
        type_file.append(1)
        # Append histograms for healthy images to a list for each color space
        hsv = histogram(health, "hsv", number_of_bins=number_of_bins)
        bgr = histogram(health, "bgr", number_of_bins=number_of_bins)
        lba = histogram(health, "lba", number_of_bins=number_of_bins)
        
        hist_vals_h.append(hsv)
        hist_vals_b.append(bgr)
        hist_vals_l.append(lba)
        
        
    
    for bbw in bbw:
        # Append 2 to mean diseased type of image
        type_file.append(2)
        # Append histograms for diseased images to a list for each color space
        hsv = histogram(bbw, "hsv", number_of_bins=number_of_bins)
        bgr = histogram(bbw, "bgr", number_of_bins=number_of_bins)
        lba = histogram(bbw, "lba", number_of_bins=number_of_bins)
        
        hist_vals_h.append(hsv)
        hist_vals_b.append(bgr)
        hist_vals_l.append(lba)
        
        
    
    for bbs in bbs:
        # Append 3 to mean diseased type of image
        type_file.append(3)
        # Append histograms for unkwon images to a list for each color space
        hsv = histogram(bbs, "hsv", number_of_bins=number_of_bins)
        bgr = histogram(bbs, "bgr", number_of_bins=number_of_bins)
        lba = histogram(bbs, "lba", number_of_bins=number_of_bins)
        
        hist_vals_h.append(hsv)
        hist_vals_b.append(bgr)
        hist_vals_l.append(lba)
        
    
    np.savetxt('vals.txt', type_file)

    with file('data_hs.txt', 'w') as outfile_h:
        np.savetxt(outfile_h, hist_vals_h) # write to text file
    with file('data_bg.txt', 'w') as outfile_b:
        np.savetxt(outfile_b, hist_vals_b)
    with file('data_lb.txt', 'w') as outfile_l:
        np.savetxt(outfile_l, hist_vals_l)
        

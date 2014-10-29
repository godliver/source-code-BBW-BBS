'''
Created on Sep 29, 2013

@author: godliver
'''
import numpy as np
#from save_histogram import number_of_images
#from save_histogram import img_i
from utils_edited import histogram

from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier


vals = np.loadtxt("vals.txt")
a_rgb = np.loadtxt("data_bg.txt")
a_lab = np.loadtxt("data_lb.txt")
a_hsv = np.loadtxt("data_hs.txt")
#number_of_bins=50

X_RGB = a_rgb
X_HSV = a_hsv
X_LAB = a_hsv

names = ["Nearest Neighbors", "Linear SVM", "Extra Trees", "RBF SVM", "Decision Tree","Random Forest", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0,compute_importances=True),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB()]

HSV = []
BGR = []
LAB = []
for name, clf in zip(names, classifiers):
    #Calculate data for the image under test for various color spaces
    hist_hsv = histogram('3.PNG','hsv',number_of_bins=50)
    hist_rgb = histogram('3.PNG','bgr',number_of_bins=50)
    hist_lab = histogram('3.PNG','lba',number_of_bins=50)
    hsv_fit = clf.fit(X_HSV, vals)
    rgb_fit = clf.fit(X_RGB, vals)
    lab_fit = clf.fit(X_LAB, vals)
    HSV.append(int(hsv_fit.predict([hist_hsv])[0]))
    BGR.append(int(rgb_fit.predict([hist_rgb])[0]))
    LAB.append(int(lab_fit.predict([hist_lab][0])))
       
mylist = [ ( 'HSV',HSV),
           ( 'BGR',BGR),
           ( 'LBA',LAB)]

header = ('Color',"Nearest Neighbors", "Linear SVM", "Extra Trees", "RBF SVM", "Decision Tree","Random Forest", "Naive Bayes")

longg = dict(zip((0,1,2,3,4,5,6,7),(len(str(x)) for x in header)))

for tu,x in mylist:
    longg.update(( i, max(longg[i],len(str(el))) ) for i,el in enumerate(tu))
    longg[7] = max(longg[7],len(str(x)))
fofo = ' | '.join('%%-%ss' % longg[i] for i in xrange(0,8))

print '\n'.join((fofo % header,
                 '-|-'.join( longg[i]*'-' for i in xrange(8)),
                 '\n'.join(fofo % (h,a,b,c,d,e,f,g) for h,(a,b,c,d,e,f,g) in mylist)))

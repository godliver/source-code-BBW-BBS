# Authour: Godliver Owomugisha
# MSC in Computer Science
import cv2
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
import numpy
import inspect
from sklearn import cross_validation
import cv
import numpy 
import numpy as np
import pylab as pl
import glob
from utils_edited import bins
#from save_histogram import img_i
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.ensemble import ExtraTreesClassifier

healthy = './healthy/'
bbw = './bbw/'
bbs = './bbs/'
healthy_files = glob.glob(healthy + '*.PNG')
bbw_files = glob.glob(bbw + '*.PNG')
bbs_files = glob.glob(bbs + '*.PNG')

vals = numpy.loadtxt("vals.txt")
a_rgb = numpy.loadtxt("data_bg.txt")
a_lab = numpy.loadtxt("data_lb.txt")
a_hsv = numpy.loadtxt("data_hs.txt")

number_of_images = len(healthy_files) + len(bbw_files) + len(bbs_files)

X_RGB = a_rgb
X_HSV = a_hsv
X_LAB = a_lab


cv = StratifiedKFold(vals, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 150)
all_tpr = []

n_samples, n_features = X_LAB.shape

names = ["Nearest Neighbors", "Linear SVM", "Extra Trees", "RBF SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0,compute_importances=True),
    SVC(gamma=0.00001, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=2, n_estimators=10, max_features=1),
    GaussianNB()]

# Loop over the classifiers
hsv_s = []
rgb_s = []
lab_s = []


X_HSV = StandardScaler().fit_transform(X_HSV)
X_RGB = StandardScaler().fit_transform(X_RGB)
X_LAB = StandardScaler().fit_transform(X_LAB)
graph = []
for name, clf in zip(names, classifiers):
    for i, (train, test) in enumerate(cv):
        # vals[train]
        probas_ = clf.fit(X_HSV[train], vals[train]).predict_proba(X_HSV[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(vals[test] == 2 , probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        # print roc_auc
        #pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    mean_tpr /= len(cv)
    mean_tpr =  mean_tpr/np.max(mean_tpr)
    #mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    graph.append(mean_fpr)
    pl.plot(mean_fpr, mean_tpr, label='%s mean roc (area = %0.2f)' % (name, mean_auc), lw=2)
    
pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random guess')
pl.xlim([-0.05, 1.05])
pl.ylim([-0.05, 1.05])
pl.xlabel('false positive rate')
pl.ylabel('true positive rate')
pl.title('receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()

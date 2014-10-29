from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from extract_shape import number_of_images
vals = np.loadtxt("vals.txt")
a_rgb = np.loadtxt("shape.txt")
print a_rgb.shape
X_RGB = np.reshape(a_rgb, (1869,19))
forest = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0,compute_importances=True)

forest.fit(X_RGB, vals)
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(indices)):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances")
pl.bar(range(len(indices)), importances[indices],
       color="r", yerr=std[indices], align="center")
pl.xticks(range(len(indices)), indices)
pl.xlim([-1, len(indices)])
pl.show()


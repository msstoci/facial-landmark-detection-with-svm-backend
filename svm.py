import numpy as np
import json

file = open("testfile.txt", "r") 
data = json.loads(file.read())

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])

X = np.array(data['training'])
y = np.array(data['target'])

from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print(clf.predict([[262, 262, 265, 271, 280, 298, 322, 349, 381, 413, 440, 464, 482, 490, 495, 499, 500, 269, 284, 307, 330, 353, 393, 416, 442, 467, 485, 374, 375, 376, 376, 353, 365, 378, 391, 404, 293, 307, 324, 341, 324, 306, 408, 425, 444, 459, 445, 427, 332, 351, 367, 379, 391, 409, 429, 410, 393, 380, 367, 350, 340, 367, 379, 392, 421, 391, 379, 366]]))


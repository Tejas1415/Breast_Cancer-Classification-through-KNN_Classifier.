"""
@author-Tejas_K_Reddy
10-05-2018 3:55 

Breast Cancer classification through KNN classifier
"""

import numpy as np            # Import necessery modules
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm

df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)  # create an outlier for missing values such that many algorithms detects that and penalises it
df.drop(['id'],1, inplace=True) # Remove Unnecessery data from the input CSV file before modelling it.

X= np.array(df.drop(['class'], 1))
y= np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.25)

# Use Knn classification model
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)                                 # around 97% accuracy found

# Use SVM classification model
clf1= svm.SVC()
clf1.fit(X_train, y_train)
accuracy = clf1.score(X_test,y_test)
print(accuracy)                                 # Comparitively lesser accuracy found.

# Now predict for a random example
example = np.array([[4,1,2,1,2,2,3,2,2], [4,1,2,2,2,1,3,2,2], [8,10,10,8,6,10,9,7,2]])    # Always declare this as lists of lists
example = example.reshape(len(example), -1)         # learn as idiom for all predictions

prediction = clf.predict(example)
prediction1 = clf1.predict(example)
print(prediction, prediction1)


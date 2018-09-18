import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm

df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)  # create an outlier for missing values such that many algorithms detect that and penalise it
df.drop(['id'],1, inplace=True)

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
print(accuracy)

# Now predict for a random example
example = np.array([[4,1,2,1,2,2,3,2,2], [4,1,2,2,2,1,3,2,2], [8,10,10,8,6,10,9,7,2]])    # Always declare this as lists of lists
example = example.reshape(len(example), -1)         # learn as idiom for all predictions

prediction = clf.predict(example)
prediction1 = clf1.predict(example)
print(prediction, prediction1)


# Breast Cancer Detector using Supervied Learning

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 1:dataset.shape[1] - 1].values
y = dataset.iloc[:, dataset.shape[1] - 1].values

# Handling the missing values
from sklearn.impute import SimpleImputer
X = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent').fit_transform(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories = 'auto')
y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()[:,0]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Logistic Regression
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0, solver = 'liblinear')
#classifier.fit(X_train, y_train)

# KNN
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)

# SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto')
#classifier.fit(X_train, y_train)

# Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)

# Decision Tree
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
#classifier.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
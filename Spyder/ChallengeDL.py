# Breast Cancer Detector using Deep Learning

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
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()

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
# Importing Keras library
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, input_dim = 9, activation = 'relu', kernel_initializer = 'uniform'))

#Adding the second hidden layer
classifier.add(Dense(units = 5, activation = 'relu', kernel_initializer = 'uniform'))

#Adding the output layer
classifier.add(Dense(units = 2, activation = 'softmax', kernel_initializer = 'uniform'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fiting ANN to training set
classifier.fit(X_train, y_train, batch_size = 1, epochs = 10)

#Predicting the results
y_pred = (classifier.predict(X_test) >= 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
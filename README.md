import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

dataset.head()

# splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#print(X_train)
#print(y_train)
 
#normalize data
X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_train_norm[0:6]

# Classification
from sklearn.neighbors import KNeighborsClassifier
#traning the data
k = 6
#Train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train_norm,y_train)


X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
X_test_norm[0:6]

# Predicting
yhat = neigh.predict(X_test_norm)
yhat[0:6]

# Accuracy Eval.
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train_norm)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

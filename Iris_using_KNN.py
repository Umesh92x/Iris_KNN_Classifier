# Predict Iris flower using KNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('C://Users//vinod//PycharmProjects//Excercise//Udemy//Iris.csv')

X=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values

# Spliting the train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0)

#import KNN Library package
from sklearn.neighbors import KNeighborsClassifier
neighbors=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

# Apply fit matrix for finding mean and variance
neighbors.fit(X_train,y_train)

# Now, its time to predict
y_pred=neighbors.predict(X_test)

# Checking correct predict and wrong predict Using matrics function
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

# Predict random data
data=np.array([[5.1,3.5,1.4,0.2]])
y_pred=neighbors.predict(data)

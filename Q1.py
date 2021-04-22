from Logisticregression.LogisticRegression import LogisticRegression
import pandas as pd
import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd

np.random.seed(42)
#number of folds
k = 3
X, y = sklearn.datasets.load_breast_cancer(as_frame= True, return_X_y=True)


LR = LogisticRegression()
X = LR.feature_scaler(X)
X.sort_index(axis = 1, inplace = True)
# 1 A using Gradient descent
print('\nUsing gradient Descent')
LR.fit_and_crossvalidate(X, y,100,0.001,k)
LR.accuracy()

#1 B using Autograd libraries
print('\nUsing Autograd')
LR.fit_and_crossvalidate_autograd(X, y,100,0.001,k)
LR.accuracy()

# 1 d plot decision boundary for the first two features

#use feature scaler again as it also adds the biases
print('\nPotting decision boundary')
n = 9
X = X.iloc[:,n:n+2]
X_copy= LR.feature_scaler(X)
#print(X_copy)
LR.fit_and_crossvalidate(X_copy, y,100,0.001,3)
LR.plot_boundary()
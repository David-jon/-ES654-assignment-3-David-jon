from Logisticregression.LogisticRegression2 import LogisticRegression
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
#1 B using Autograd libraries
print('\nUsing Autograd')

print('L1 Logistic regression')

Acc_hist_L1 = []
L1_reg_value = []
for i in np.linspace(2500,2900,4):
	LR.fit_and_crossvalidate_autograd_L1(X, y,100,0.001,k,i)
	Acc_hist_L1.append(LR.accuracy())
	L1_reg_value.append(i)
print('Highest accuracy is:', np.max(Acc_hist_L1),'With L1 regularization constant:',L1_reg_value[np.argmax(Acc_hist_L1)])
#print('Index list',L1_reg_value)
#print('Acc list',Acc_hist_L1)



#after using 200 points, max acc at 86.66238767650836 at L1  reg value at 2801.5075376884424
#


print('L2 Logistic regression')

Acc_hist_L2 = []
L2_reg_value = []

for i in np.linspace(2000,5000,20):
	LR.fit_and_crossvalidate_autograd_L2(X, y,100,0.001,k,i)
	Acc_hist_L2.append(LR.accuracy())
	L2_reg_value.append(i)

#Highest accuracy is: 82.45614035087719 With L2 regularization constant: 3105.2631578947367

print('Highest accuracy is:', np.max(Acc_hist_L2),'With L2 regularization constant:',L2_reg_value[np.argmax(Acc_hist_L2)])
#print('Index list',L2_reg_value)
#print('Acc list',Acc_hist_L2)
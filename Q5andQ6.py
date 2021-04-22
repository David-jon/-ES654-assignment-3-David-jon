from MLP.MLP import MLP
from sklearn import datasets
from autograd import grad
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.metrics import confusion_matrix as cm
warnings.filterwarnings('ignore')

# Digits section
X,y = datasets.load_digits(return_X_y=1)
from sklearn.model_selection import KFold
K = 3 #number of folds
kf = KFold(n_splits=K)
j = 0
hist = np.array([])
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = MLP(X_train,y_train,X_test,y_test,L=2,N_l=64)
    model.train(epochs=200,lr=1.0)
    y_pred = model.predict(X_test)
    y_pred_real= []
    for i in y_pred:
        y_pred_real.append(np.argmax(i))
    plt.plot(model.train_loss, label = 'Train Loss')
    plt.plot(model.val_loss, label = 'Validation Loss')
    plt.title('Loss function time for digits dataset')
    plt.legend(loc="upper right")
    plt.show()
    y_pred = model.predict(X_test)
    y_pred_real= []
    for i in y_pred:
        y_pred_real.append(np.argmax(i))
    from sklearn.metrics import confusion_matrix as cm
    conf_mat = cm(y_test,y_pred_real)
    j+=1
    print('Confusion matrx Digits dataset for fold',j,'\n',conf_mat)
    temp = 0
    for i in range(10):
        temp+=conf_mat[i][i]
    print('\n Overall accuracy for digits dataset: ', temp/np.sum(conf_mat))
    hist = np.append(hist, temp/np.sum(conf_mat))
print('Overall accuracy over all folds for digitis dataset:', np.mean(hist))



#Boston housing section
X_2,y_2 = datasets.load_boston(return_X_y=True)
X2_train = X[:int(0.8*X_2.shape[0])]
y2_train = y[:int(0.8*y_2.shape[0])]
X2_test = X[int(0.8*X_2.shape[0]):]
y2_test = y[int(0.8*y_2.shape[0]):]
model2 = MLP(X2_train,y2_train,X2_test,y2_test,L=2,N_l=64)
model2.train(epochs=500,lr=0.01)
plt.plot(model2.train_loss, label = 'Train Loss')
plt.plot(model2.val_loss, label = 'Validation Loss')
plt.title('Loss function time for Boston housing dataset')
plt.legend(loc="upper right")
plt.show()
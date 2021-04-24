from Logisticregression.MultiClassLogisticRegression import MultiClassLogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

print('USING UPDATE RULES')
from sklearn import datasets
X,y = datasets.load_digits(return_X_y=True)
X_train = X[0:X.shape[0]*3//4]
y_train = y[0:y.size*3//4]
X_test = X[X.shape[0]*3//4:]
y_test = y[y.size*3//4:]
lr = MultiClassLogisticRegression(thres=1e-5)
lr.fit(X_train,y_train,lr=0.0001)
print('Overall Accuracy:',lr.score(X, y))

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(lr.loss)), lr.loss)
plt.title("Development of loss during training")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()


pred = lr.predict_classes(X_test)
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
#df_confusion
print('Accuracy of digits using update rules')
for i in range(10):
    print(i,':',np.sum(df_confusion.iloc[i][i])/np.sum(y_test==i))
# 2 gets 100 percent accuracy, whereas 1 and 3 gets the lowest accuracy

print('USING AUTOGRAD')
lr2 = MultiClassLogisticRegression(1000,thres=1e-5)
lr2.fit_w_Autograd(X_train,y_train,lr=0.0001)
print('Overall Accuracy:',lr2.score(X, y))

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(lr.loss)), lr.loss)
plt.title("Development of loss during training")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()

pred = lr2.predict_classes(X_test)
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
#df_confusion
print('Accuracy of digits using Autograd')
for i in range(10):
    print(i,':',np.sum(df_confusion.iloc[i][i])/np.sum(y_test==i))
# 2 gets 100 percent accuracy, whereas 1 and 3 gets the lowest accuracy

#4 Fold digit accuracy

print('CALCULATING DIGIT ACCURACY ON 4 FOLDS')
from sklearn.model_selection import KFold
K = 4 #number of folds
kf = KFold(n_splits=K)
j = 0
Acc_hist = np.array([])
Acc_hist_fold = np.array([])
for train_index, test_index in kf.split(X):
    lr = MultiClassLogisticRegression(thres=1e-5)
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train,y_train,lr=0.0001)
    pred = lr.predict_classes(X_test)
    y_actu = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(pred, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    #df_confusion
    j+=1
    Acc_hist_fold = np.array([])
    print('Accuracy of digits using update rules for fold ',j)
    for i in range(10):
        Acc_percent = np.sum(df_confusion.iloc[i][i])/np.sum(y_test==i)
        Acc_hist_fold = np.append(Acc_hist_fold,Acc_percent)
        print(i,':',np.sum(Acc_percent))
    Acc_hist = np.append(Acc_hist,Acc_hist_fold)

print('Overall accuracy over',K,'Folds of individual digits')
temp2 = 0
#for i in range(K):
for iter in range(10):
    print('\nDigit:',iter,'Accuracy')
    temp = np.array([])
    for j in np.arange(0,K*10,10):
        temp = np.append(temp,Acc_hist[iter+j])
    print(np.mean(temp))
    temp2 += np.mean(temp)
print('\n\nOVERALL ACCURACY:',temp2/10)

# got highest accuracy for digit 0 and digit 3 and 9 got the lowest accuracy



#PCA CODE
digits = load_digits()
fig = plt.figure(figsize=(15,12))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)


pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.title('Using PCA on digits dataset with 2 dimensions')
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()
plt.show()



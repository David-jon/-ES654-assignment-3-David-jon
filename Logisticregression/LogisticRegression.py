import autograd.numpy as np
from autograd import grad
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:

	def __init__(self):
		self.theta = None
		self.fold = np.array([])
		self.y = None
		self.X = None

	def feature_scaler(self,X):
		X = (X - X.min()) / (X.max() - X.min())
		df = pd.DataFrame(np.ones(X.shape[0]), columns = ['Bias'])
		X = pd.concat([X,df], axis = 1)
		return X

	def predict(self,X_test):
		result_mat = SIGMOID(theta, X_test)
		y_pred = result_mat > 0.485
		return y_pred

	def accuracy(self):
		print('\nAverage accuracy over all folds', np.mean(self.fold),'%')


	def fit_and_crossvalidate(self,X, y, n_iter = 1000, lr = 0.001, K_Fold = 3):
		#fold = np.array([])
		#J will contain history of costs
		#for matrix multiplication 
		#use * for multiplying a constant with a matrix, but not matrix with a matrix
		#use @ for multiplying two matrices but not matrix with a constant
		#most of the predictions are very close to the 0.5 value so low accuracy
		#train test split 2:1
		#incr for calcualting the range for k folds
		self.y = y
		self.X = X
		def COST_and_GD(theta, X, y, n_iter):
			for i in range(n_iter):
				lr = 0.001
				J.append(np.sum(-y.T @ np.log(1/(1+ np.exp(-1* (X@theta)))) - (1-y).T @ np.log(1-1/(1+ np.exp(-1* (X@theta)))))/y.size)
				theta -= lr * X.T @ (1/(1+ np.exp(-1* (X@theta))) - y)
		def SIGMOID(theta, X):
		    return 1/(1+ np.exp(-1* (X@theta)))

		#X_train = X_train.values
		#X_test = X_test.values
		incr = 0
		#k fold cross validation
		k = K_Fold
		for j in range(k):
		    X_train = pd.DataFrame()
		    X_test = pd.DataFrame()
		    y_test = np.array([])
		    y_train = np.array([])
		    n1 = 0 + incr
		    n2 = X.shape[0]//k +incr
		    for i in range(X.shape[0]):
		        if i>=n1 and i<=n2:
		            X_test = X_test.append(X.iloc[i,])
		            y_test = np.append(y_test,y[i])
		        else:
		            X_train = X_train.append(X.iloc[i,])
		            y_train = np.append(y_train,y[i])
		    incr+=X.shape[0]//k
		    self.theta = np.zeros(X_train.shape[1])
		    #J var for storing the error histoy
		    #X_train = X_train.values
		    #X_test = X_test.values

		    J = []
		    COST_and_GD(self.theta, X_train, y_train,1000)

		    result_mat = SIGMOID(self.theta, X_test)
		    temp = result_mat > 0.485
		    self.fold = np.append(self.fold,np.sum(temp == y_test)/y_test.size*100)
		    #print('Baselines',np.sum(y_test==1)/(np.sum(y_test==1)+np.sum(y_test==0)))
		    print('Fold', j+1 ,'accuracy: ',self.fold[j],'%')

	def fit_and_crossvalidate_autograd(self,X, y, n_iter = 1000, lr = 0.001, K_Fold = 3):
		
		def COST(theta, X, y):
			return np.sum(np.dot(-y.T,np.log(1/(1+ np.exp(-1* (np.dot(X,theta)))))) - np.dot((1-y).T,np.log(1-1/(1+ np.exp(-1* (np.dot(X,theta)))))))/y.size
		def SIGMOID(theta, X):
		    return 1/(1+ np.exp(-1* (X@theta)))
		grad_cost = grad(COST,0)

		incr = 0
		#k fold cross validation
		k = K_Fold
		for j in range(k):
		    X_train = pd.DataFrame()
		    X_test = pd.DataFrame()
		    y_test = np.array([])
		    y_train = np.array([])
		    n1 = 0 + incr
		    n2 = X.shape[0]//k +incr
		    for i in range(X.shape[0]):
		        if i>=n1 and i<=n2:
		            X_test = X_test.append(X.iloc[i,])
		            y_test = np.append(y_test,y[i])
		        else:
		            X_train = X_train.append(X.iloc[i,])
		            y_train = np.append(y_train,y[i])
		    incr+=X.shape[0]//k
		    self.theta = np.zeros(X_train.shape[1])
		    #J var for storing the error histoy
		    X_train = X_train.values
		    X_test = X_test.values

		    J = []
		    #COST_and_GD(self.theta, X_train, y_train,1000)
		    self.theta -= lr *grad_cost(self.theta, X_train, y_train)
		    result_mat = SIGMOID(self.theta, X_test)
		    temp = result_mat > 0.485
		    self.fold = np.append(self.fold,np.sum(temp == y_test)/y_test.size*100)
		    #print('Baselines',np.sum(y_test==1)/(np.sum(y_test==1)+np.sum(y_test==0)))

		    print('Fold', j+1 ,'accuracy: ',self.fold[j],'%')

	def plot_boundary(self):
		X_pos_lst = []
		y_pos_lst = []

		X_neg_lst = []
		y_neg_lst = []
		for i in range(self.y.size):
		    if self.y[i]==1:
		        X_pos_lst.append(self.X.iloc[i,0])
		        y_pos_lst.append(self.X.iloc[i,1]) 
		    else:
		        X_neg_lst.append(self.X.iloc[i,0])
		        y_neg_lst.append(self.X.iloc[i,1])  
		#theta_mod = theta/theta[2]
		theta_mod = self.theta[:2]
		#print('X_pos_list',X_pos_lst)
		#print('y_pos_list',y_pos_lst)
		#rint('X_neg_list',X_neg_lst)
		#rint('y_neg_list',y_neg_lst)
		a = .3
		b = -30
		x = -1 * theta_mod[0] *np.linspace(-1 * a,b,10)
		y = x - theta_mod[1]
		plt.plot(x,y)
		plt.plot(X_pos_lst,y_pos_lst,'ro',markersize = 1)
		plt.plot(X_neg_lst,y_neg_lst,'bo',markersize = 1)
		plt.xlabel(self.X.columns[0])
		plt.ylabel(self.X.columns[1])
		plt.title('Logistic regression Decision surface')
		plt.show()
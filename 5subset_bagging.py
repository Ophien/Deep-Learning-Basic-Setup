import random
import numpy as np
import pickle

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.cross_decomposition import PLSRegression

#Number of samples in each class
firstItens = 1000

def loadfc7():

    with open('fc7.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)

    data_x_all = data_raw[0]
    data_y_all = data_raw[1]
    
    mean_sample = np.mean(data_x_all, axis=0)
    data_x_all -= mean_sample
    
    return data_x_all, data_y_all

data_x_all_fc7, data_y_all_fc7 = loadfc7()

#Number of splits of the feature vector
subset=5

#Number of classes available in the dataset
num_labels = 17

#Initialize arrays
data_x = []
data_y = []
fc7_x=[None]*subset

for i in range(subset):
    fc7_x[i] = []

#Number of features for each split
offset = np.int32(len(data_x_all_fc7[0]) / subset)

for k in range(num_labels):
    #Generate data with the X first items of each class
    a = [data_x_all_fc7[index] for index in [i for i,j in enumerate(data_y_all_fc7) if j == k][:firstItens]]
    data_x += a
    
    #Split the feature vector
    for sample in a:
        for i in range(subset):
            fc7_x[i].append( sample[i*offset:(i+1)*offset] )

    #Create the labels refering to the selected data
    data_y += [k]*len(a)

#With PLS the results improve in accuracy and computational time
pls = PLSRegression(n_components=10, scale=True)

for i in range(subset):
    pls.fit(fc7_x[i], data_y)
    fc7_x[i] = pls.transform(fc7_x[i])


fc7_X_train = [None]*subset
fc7_X_test = [None]*subset
fc7_y_train = [None]*subset
fc7_y_test = [None]*subset

#Generate train/test splits for all subsets
for i in range(subset):
    fc7_X_train[i], fc7_X_test[i], fc7_y_train[i], fc7_y_test[i] = train_test_split(fc7_x[i], data_y, test_size=0.33, random_state=42)
  
#Create parameters to choose in the grid search
C_values = np.logspace(-2, 5, 8)

parameters = [
    {'C':C_values, 'penalty': ['l1', 'l2'], 'loss': ['squared_hinge'], 'dual':[False]},
    {'C':C_values, 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'dual':[True]}
    ]

for line in parameters:
    for key in line:
        print(key,":", line[key])
    print("")

#Function to create a bagging of SVM classifiers. Returns the probabilities of each class for each sample  
def baggingClassifier(data_X, data_y, test_X, test_y, n_estimators=5):
    n_samples = len(data_x)
    print(len(data_x),len(data_y))
    samples_est = np.int32(n_samples / n_estimators)
    
    #Array to store the probabilities of each label
    fc7_dec_fun = [None]*n_estimators

    for i in range(n_estimators):
        #Pick random samples to each classifier 
        index = np.random.choice(np.arange(len(data_x)), size=samples_est, replace=False)
        
        data_X_est = [sample for i,sample in enumerate(data_X) if i in index]
        data_y_est = [labels for i,labels in enumerate(data_y) if i in index]
        
        #Grid Search to choose the best set of hyperparameters
        model_to_set = svm.LinearSVC()
        best_clf = GridSearchCV(model_to_set, parameters, cv=5, verbose=1, n_jobs=8)
    
        best_clf.fit(data_X_est, data_y_est)
        print ("estimator", i+1, "best score: ", best_clf.best_score_)
        print ("estimator", i+1, "best parameters: ", best_clf.best_params_)

        fc7_C = best_clf.best_params_['C']
        fc7_penalty = best_clf.best_params_['penalty']
        fc7_loss = best_clf.best_params_['loss']
        fc7_dual = best_clf.best_params_['dual']
        
        fc7_svm = svm.LinearSVC(C=fc7_C, penalty=fc7_penalty, loss=fc7_loss, dual=fc7_dual)
        fc7_svm.fit(data_X_est, data_y_est)
        fc7_dec_fun[i] = fc7_svm.decision_function(test_X)
    

    bagging_scores = []
    for i in range(len(test_X)):
        sum_scores = np.zeros(num_labels) #17 labels
        for j in range(n_estimators):
            sum_scores += np.array(fc7_dec_fun[j][i])
        
        bagging_scores.append(sum_scores)


    return bagging_scores

#Bagging using 4 estimators
for i in range(subset):
    s = np.array(baggingClassifier(fc7_X_train[i], fc7_y_train[i], fc7_X_test[i], fc7_y_test[i], n_estimators=4))
    if(i==0):
        sum_scores = s
    else:
        #Adds the probabilities found by each estimator
        np.add(sum_scores, s)

#The label with highest probability is chosen
y_hat = [np.argmax(score) for score in sum_scores]

#Since the order of the samples is the same for all subsets, we chose one of them as test
print("5_subset + Bagging(Linear SVM) accuracy score:", accuracy_score(fc7_y_test[1], y_hat))
print("5_subset + Bagging(Linear SVM) recall score:", recall_score(fc7_y_test[1], y_hat, average='macro'))
print(confusion_matrix(fc7_y_test[1], y_hat))

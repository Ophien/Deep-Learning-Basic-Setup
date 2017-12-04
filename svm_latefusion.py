import random
import numpy as np
import pickle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_decomposition import PLSRegression

#Number of samples in each class
firstItens = 50

#Number of classes available in the dataset
num_labels = 17

#Initialize arrays
data_x = np.array([])
data_y = []

def loadfc7():

    with open('fc7.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)

    data_x_all = data_raw[0]
    
    data_y_all = data_raw[1]

    mean_sample = np.mean(data_x_all, axis=0)
    data_x_all -= mean_sample
    
    return data_x_all, data_y_all

def loadconv5():
    with open('conv5.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)
    data_x_all = []
    for fmap in data_raw[0]:
        data_x_all.append(fmap.flatten())
    
    data_y_all = data_raw[1]

    mean_sample = np.mean(data_x_all, axis=0)
    data_x_all -= mean_sample
    
    return data_x_all, data_y_all
    
def loadconv1():
    data_x_all = []
    data_y_all = []
    
    with open('conv1.2.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)
    for fmap in data_raw[0]:
        data_x_all.append(fmap.flatten())
    data_y_all += data_raw[1]
        
    with open('conv1.3.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)
    for fmap in data_raw[0]:
        data_x_all.append(fmap.flatten())
    data_y_all += data_raw[1]
    
    with open('conv1.4.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)
    for fmap in data_raw[0]:
        data_x_all.append(fmap.flatten())
    data_y_all += data_raw[1]

    with open('conv1.5.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)
    for fmap in data_raw[0]:
        data_x_all.append(fmap.flatten())
    data_y_all += data_raw[1]
    
    mean_sample = np.mean(data_x_all, axis=0)
    data_x_all -= mean_sample
    
    return data_x_all, data_y_all

data_x_all_fc7, data_y_all_fc7 = loadfc7()
data_x_all_conv5, data_y_all_conv5 = loadconv5()
data_x_all_conv1, data_y_all_conv1 = loadconv1()


fc7_x = []
conv1_x = []
conv5_x = []

for k in range(num_labels):
    #Load features individually
    a = [data_x_all_fc7[index] for index in [i for i,j in enumerate(data_y_all_fc7) if j == k][:firstItens]]
    fc7_x += a
    
    b = [data_x_all_conv1[index] for index in [i for i,j in enumerate(data_y_all_conv1) if j == k][:firstItens]]    
    conv1_x += b
    
    c = [data_x_all_conv5[index] for index in [i for i,j in enumerate(data_y_all_conv5) if j == k][:firstItens]]
    conv5_x += c
    
    #Create the labels refering to the selected data
    data_y += [k]*len(a)
    
#With PLS the results improve in accuracy and computational time
pls = PLSRegression(n_components=10, scale=True)

pls.fit(fc7_x, data_y)
print ("before", fc7_x[0], len(fc7_x[0]))
fc7_x = pls.transform(fc7_x)   
print ("after", fc7_x[0], len(fc7_x[0]))

pls.fit(conv1_x, data_y)
print ("before", conv1_x[0], len(conv1_x[0]))
conv1_x = pls.transform(conv1_x)   
print ("after", conv1_x[0], len(conv1_x[0]))

pls.fit(conv5_x, data_y)
print ("before", conv5_x[0], len(conv5_x[0]))
conv5_x = pls.transform(conv5_x)   
print ("after", conv5_x[0], len(conv5_x[0]))

#Generate train/test splits
fc7_X_train, fc7_X_test, fc7_y_train, fc7_y_test = train_test_split(fc7_x, data_y, test_size=0.33, random_state=42)
conv1_X_train, conv1_X_test, conv1_y_train, conv1_y_test = train_test_split(conv1_x, data_y, test_size=0.33, random_state=42)
conv5_X_train, conv5_X_test, conv5_y_train, conv5_y_test = train_test_split(conv5_x, data_y, test_size=0.33, random_state=42)

#Create parameters to choose in the grid search
penalty = np.logspace(-2, 10, 5)

# Parameters to be evaluated
parameters = [
    {'C':penalty, 'multi_class': ['ovr', 'crammer_singer'], 'penalty': ['l1', 'l2'], 'loss': ['squared_hinge'], 'dual':[False]},
    {'C':penalty, 'multi_class': ['ovr', 'crammer_singer'], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'dual':[True]},
    ]

#Run a grid search for each classifier
model_to_set = svm.LinearSVC()
best_clf = GridSearchCV(model_to_set, parameters, cv=5, verbose=1, n_jobs=8)
                             
best_clf.fit(fc7_X_train, fc7_y_train)
print ("fc7 best score: ", best_clf.best_score_)
print ("fc7 best parameters: ", best_clf.best_params_)

fc7_C = best_clf.best_params_['C']
fc7_multi_class = best_clf.best_params_['multi_class']
fc7_penalty = best_clf.best_params_['penalty']
fc7_loss = best_clf.best_params_['loss']
fc7_dual = best_clf.best_params_['dual']


best_clf.fit(conv5_X_train, conv5_y_train)
print ("conv5 best score: ", best_clf.best_score_)
print ("conv5 best parameters: ", best_clf.best_params_)

conv5_C = best_clf.best_params_['C']
conv5_multi_class = best_clf.best_params_['multi_class']
conv5_penalty = best_clf.best_params_['penalty']
conv5_loss = best_clf.best_params_['loss']
conv5_dual = best_clf.best_params_['dual']


best_clf.fit(conv1_X_train, conv1_y_train)
print ("conv1 best score: ", best_clf.best_score_)
print ("conv1 best parameters: ", best_clf.best_params_)

conv1_C = best_clf.best_params_['C']
conv1_multi_class = best_clf.best_params_['multi_class']
conv1_penalty = best_clf.best_params_['penalty']
conv1_loss = best_clf.best_params_['loss']
conv1_dual = best_clf.best_params_['dual']

#Run the SVM saving the probabilities of each class for each sample
fc7_svm = svm.LinearSVC(C = fc7_C, multi_class=fc7_multi_class, penalty=fc7_penalty, loss=fc7_loss, dual=fc7_dual)
conv5_svm = svm.LinearSVC(C = conv5_C, multi_class=conv5_multi_class, penalty=conv5_penalty, loss=conv5_loss, dual=conv5_dual)
conv1_svm = svm.LinearSVC(C = conv1_C, multi_class=conv1_multi_class, penalty=conv1_penalty, loss=conv1_loss, dual=conv1_dual)

fc7_svm.fit(fc7_X_train, fc7_y_train)
conv5_svm.fit(conv5_X_train, conv5_y_train)
conv1_svm.fit(conv1_X_train, conv1_y_train)

fc7_dec_fun = fc7_svm.decision_function(fc7_X_test)
conv5_dec_fun = conv5_svm.decision_function(conv5_X_test)
conv1_dec_fun = conv1_svm.decision_function(conv1_X_test)

#The final prediction is the combination of individual scores
y_hat = np.zeros(len(fc7_X_test))
for i in range(len(fc7_X_test)):
    fc7_scores = np.array(fc7_dec_fun[i])
    conv5_scores = np.array(conv5_dec_fun[i])
    conv1_scores = np.array(conv1_dec_fun[i])
    
    sum_scores = fc7_scores + conv5_scores + conv1_scores
    y_hat[i] = np.argmax(sum_scores)
    
print("Late Fusion accuracy score:", accuracy_score(fc7_y_test, y_hat))
print("Late Fusion recall score:", recall_score(fc7_y_test, y_hat, average='macro'))

print(confusion_matrix(fc7_y_test, y_hat))

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
firstItens = 100

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

#Initizalize arrays
#data_x = np.array([])
data_y = []
fc7_x=[None]*5
for i in range(5):
    fc7_x[i] = []
#Number of features for each split
offset = np.int32(len(data_x_all_fc7[0]) / subset)


for k in range(num_labels):
    #Generate data with the X first items of each class
    a = [data_x_all_fc7[index] for index in [i for i,j in enumerate(data_y_all_fc7) if j == k][:firstItens]]
    
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

fc7_C = [None]*subset
fc7_penalty = [None]*subset
fc7_loss = [None]*subset
fc7_dual = [None]*subset

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

model_to_set = svm.LinearSVC()
best_clf = GridSearchCV(model_to_set, parameters, cv=5, verbose=1, n_jobs=8)

#Run a grid search to select the best hyperparameters for each split
for i in range(subset):                           
    best_clf.fit(fc7_X_train[i], fc7_y_train[i])
    print ("fc7", i, "best score: ", best_clf.best_score_)
    print ("fc7", i, "best parameters: ", best_clf.best_params_)

    fc7_C[i] = best_clf.best_params_['C']
    fc7_penalty[i] = best_clf.best_params_['penalty']
    fc7_loss[i] = best_clf.best_params_['loss']
    fc7_dual[i] = best_clf.best_params_['dual']


fc7_svm = [None]*subset
fc7_dec_fun = [None]*subset

for i in range(subset):
    fc7_svm[i] = svm.LinearSVC(C = fc7_C[i], penalty=fc7_penalty[i], loss=fc7_loss[i], dual=fc7_dual[i])
    fc7_svm[i].fit(fc7_X_train[i], fc7_y_train[i])
    fc7_dec_fun[i] = fc7_svm[i].decision_function(fc7_X_test[i])
   

fc7_scores = [None]*subset

y_hat = np.zeros(len(fc7_X_test[0]))

#Combines the probabilities for each split
for i in range(len(fc7_X_test[0])):
    sum_scores = np.zeros(len(fc7_dec_fun[0][0]))
    for j in range(subset):
        fc7_scores[j] = np.array(fc7_dec_fun[j][i])
        sum_scores += fc7_scores[j]
    #The label with highest probability is chosen
    y_hat[i] = np.argmax(sum_scores)
    
#Since the order of the samples is the same for all subsets, we chose one of them as test
print(accuracy_score(fc7_y_test[0], y_hat))
print(recall_score(fc7_y_test[0], y_hat, average='macro'))
print(confusion_matrix(fc7_y_test[0], y_hat))

import random
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.cross_decomposition import PLSRegression

#Number of samples in each class
firstItens = 50

#Number of classes available in the dataset
num_labels = 17

def loadfc7():

    with open('fc7.pickle', 'rb') as inputs:
        data_raw = pickle.load(inputs)

    data_x_all = data_raw[0]
    
    data_y_all = data_raw[1]
    
    #data_y_x = [None]*17

    mean_sample = np.mean(data_x_all, axis=0)
    data_x_all -= mean_sample
    
    return data_x_all, data_y_all

data_x_all, data_y_all = loadfc7()

data_x = []
data_y = []

for k in range(num_labels):
    #Generate data with the X first items of each class
    d = [data_x_all[index] for index in [i for i,j in enumerate(data_y_all) if j == k][:firstItens]]
    data_x += d
    #Create the labels refering to the selected data
    data_y += [k]*len(d)

data_x_all = []
data_y_all = []

#With PLS the results improve in accuracy and computational time
#pls = PLSRegression(n_components=10, scale=True)
#pls.fit(data_x, data_y)
#print ("before", data_x[0])
#data_x = pls.transform(data_x)   
#print ("after", data_x[0])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)

#Create parameters to choose in the grid search
n_estimators = np.linspace(60, 100, 5, dtype=np.int32)
depth = np.linspace(10, 50, 5, dtype=np.int32)

# Parameters to be evaluated
parameters = {'estimator__n_estimators': n_estimators, 
              'estimator__max_depth': depth}


model_to_set = OneVsRestClassifier(RandomForestClassifier())
best_clf = GridSearchCV(model_to_set, parameters, cv=5, verbose=1, n_jobs=8)
best_clf.fit(X_train, y_train)
print ("Best score: ", best_clf.best_score_)
print ("Best parameters: ", best_clf.best_params_)

y_hat = best_clf.predict(X_test)
print (confusion_matrix(y_test, y_hat))

#Measures the mean accuracy with standard deviation using the best parameters of the grid search
scoring = ['accuracy', 'recall_macro']

scores = cross_validate(OneVsRestClassifier(RandomForestClassifier(max_depth=best_clf.best_params_['estimator__max_depth'], n_estimators=best_clf.best_params_['estimator__n_estimators'])), data_x, data_y, scoring=scoring, cv=5, n_jobs=8, return_train_score=False)
print ('Tuned Random Forest accuracy for 5-fold', scores['test_accuracy'].mean(),'+-', scores['test_accuracy'].std())
print ('Tuned Random Forest recall for 5-fold', scores['test_recall_macro'].mean(),'+-', scores['test_recall_macro'].std())

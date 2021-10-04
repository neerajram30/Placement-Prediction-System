# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import sklearn 
import pickle

dataset = pd.read_csv('hirable.csv')

#Clean dataset

dataset = dataset.drop([
    "sl_no",
    "ssc_p",
    "ssc_b",
    "hsc_p",
    "hsc_b",
    "hsc_s",
    "specialisation",
    "salary",
    "degree_t"
], axis=1)
dataset = dataset.rename(columns = {'degree_p': 'bsc', 'mba_p': 'msc'})
dataset['gender'] = dataset.gender.replace(['M', 'F'], [1, 2])
dataset['workex'] = dataset.workex.replace(['Yes', 'No'], [1, 0])
dataset['status'] = dataset.status.replace(['Placed', 'Not Placed'], [1, 0])

#Downscaling Method

def downscale(score):
    return score/10/2
degrees = ['bsc', 'msc']
for col in degrees:
    dataset[col] = downscale(dataset[col])

#Saperating into dependent and independent variable

X = dataset.drop(['status'], axis=1)
y = dataset.status

#Splitting dataset into training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)



from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)


y_pred=model.predict(X_test)


from sklearn import metrics
from sklearn.metrics import classification_report
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Classification Report RF:\n",classification_report(y_test,y_pred))

#sample
sample = np.array([[0, 2.9, 1, 78.50, 3.7]])
model.predict(sample)


pickle.dump(model, open('hireable.pkl', 'wb'))

loaded_model = pickle.load(open('hireable.pkl', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
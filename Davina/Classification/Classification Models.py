#!/usr/bin/env python
# coding: utf-8

# # Classification Models

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv("train.csv")
X = data.drop(columns = ["diabetes"]).values
y = data['diabetes'].values
data.head()


# In[3]:


#Replace zeroes
zero_not_accepted = ['glucose_concentration', 'blood_pressure', 'skin_fold_thickness', 'serum_insulin', 'bmi']

for column in zero_not_accepted: 
    data[column] = data[column].replace(0, np.NaN) #replace all 0 with NaN (doesn't exist)
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.NaN, mean) #replace all NaN with the mean value


# # Decision Tree Classifier

# In[12]:


from sklearn.tree import DecisionTreeClassifier


X = data.iloc[:,0:9].values
y = data.iloc[:,9].values


XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state=1)

Dt = DecisionTreeClassifier(max_leaf_nodes=9, random_state=1)
Dt.fit(XTrain, yTrain)
yPred = Dt.predict(XTest)

accuracy = Dt.score(XTest, yTest) * 100
print('Accuracy Score:', round(accuracy, 2), '%')


# In[14]:


ConfusionM = confusion_matrix(yTest, yPred)
ConfusionM = sns.heatmap(ConfusionM, annot = True)

ConfusionM.set_title('Decision Tree Confusion Matrix')
ConfusionM.set_xlabel('Predicted')
ConfusionM.set_ylabel('Actual')


# # Stochastic Gradient Decent

# In[20]:


from sklearn.linear_model import SGDClassifier

XTrain, XTest, yTrain, yTest = train_test_split(X, y,test_size = 0.2, random_state=0)

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
SGDClassifier(max_iter=5)

clf.predict(XTest)

accuracy = clf.score(XTest, yTest) * 100
print('Accuracy Score:', round(accuracy, 2), '%')


# In[21]:


ConfusionM = confusion_matrix(yTest, yPred)
ConfusionM = sns.heatmap(ConfusionM, annot = True)

ConfusionM.set_title('Gaussian Mixture Model')
ConfusionM.set_xlabel('Predicted')
ConfusionM.set_ylabel('Actual')


# # Neural Networks

# In[22]:


from sklearn.neural_network import MLPClassifier


# In[23]:


XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=0)

NeuralNet = MLPClassifier(random_state = 3, max_iter = 9000)

NeuralNet.fit(XTrain, yTrain)
yPred = NeuralNet.predict(XTest)

accuracy = (accuracy_score(yTest, yPred)) * 100
print('Accuracy Score:', round(accuracy, 2), '%')

ConfusionM = confusion_matrix(yTest, yPred)
ConfusionM = sns.heatmap(ConfusionM, annot = True)
ConfusionM.set_title('Neural Net Confusion Matrix')
ConfusionM.set_xlabel('Predicted')
ConfusionM.set_ylabel('Actual')


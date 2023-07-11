import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


iris = datasets.load_iris()
x = iris.data
y = iris.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
gnb = GaussianNB()
gnb.fit(x_train,y_train)
gnb_pred = gnb.predict(x_test)

print("Accuracy of Gaussian Naive Bayes: ",accuracy_score(y_test,gnb_pred))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)

print("Accuracy of descision tree classifier: ",accuracy_score(y_test,dt_pred))

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(x_train,y_train)
svm_clf_pred = svm_clf.predict(x_test)


print("Accuracy of support vector machine:",accuracy_score(y_test,svm_clf_pred))




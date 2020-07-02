import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


trainSet = pd.read_csv('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/trainSet.csv')
testSet = pd.read_csv('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/testSet.csv')

TrainSet = trainSet[trainSet["label"] != 0]
TestSet = testSet[testSet["label"] != 0]

trainX2 = trainSet[["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]]
testX2 = testSet[["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]]

trainX = TrainSet[["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]]
trainY = TrainSet["label"]

testX = TestSet[["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]]
testY = TestSet["label"]


clf = RandomForestClassifier(n_estimators=50, verbose=1, min_samples_leaf=10,n_jobs=4)
clf.fit(trainX, trainY)


y_pred = clf.predict(testX)
y_prob = clf.predict_proba(testX)
trainY_pred = clf.predict(trainX)
testY_pred = clf.predict(testX)
err_rate = np.mean(y_pred != testY)

np.savetxt('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/y_pred.csv', y_pred, delimiter=',',header="label")
np.savetxt('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/trainY_pred.csv', trainY_pred, delimiter=',',header="label")
np.savetxt('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/testY_pred.csv', testY_pred, delimiter=',',header="label")
np.savetxt('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/y_test.csv', testY, delimiter=',',header="label")
np.savetxt('/Users/gofinge/Documents/PycharmProjects/STAT154_project_support/y_prob.csv', y_prob[:, 1], delimiter=',',header="label")
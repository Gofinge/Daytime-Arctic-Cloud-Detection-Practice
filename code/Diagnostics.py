import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

trainSetA = pd.read_csv('/Users/gofinge/Desktop/trainSetA.csv')
testSetA = pd.read_csv('/Users/gofinge/Desktop/testSetA.csv')

trainSetB = pd.read_csv('/Users/gofinge/Desktop/trainSetB.csv')
testSetB = pd.read_csv('/Users/gofinge/Desktop/testSetB.csv')

trainSetC = pd.read_csv('/Users/gofinge/Desktop/trainSetC.csv')
testSetC = pd.read_csv('/Users/gofinge/Desktop/testSetC.csv')

# cleaning data
trainSet = trainSetC[trainSetC["label"] != 0]
testSet = testSetC[testSetC["label"] != 0]

features_names = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
label_name = "label"

trainX = trainSet[features_names]
trainY = trainSet[label_name]
testX = testSet[features_names]
testY = testSet[label_name]

model = RandomForestClassifier(n_estimators=100, max_features=4, max_depth=50, min_samples_leaf=10, verbose=1, n_jobs=4)
model.fit(trainX, trainY)

testY_pred = model.predict(testX)
testY_pred2 = model.predict(testSetC[features_names])
trainY_pred = model.predict(trainX)
testY_prob = model.predict_proba(testX)

err_rate = np.mean(testY_pred != testY)

print(err_rate)

np.savetxt('/Users/gofinge/Desktop/trainSetC_pred.csv', trainY_pred, delimiter=',', header="label")
np.savetxt('/Users/gofinge/Desktop/testSetC_pred.csv', testY_pred, delimiter=',', header="label")
np.savetxt('/Users/gofinge/Desktop/testSetC_prob.csv', testY_prob[:, 1], delimiter=',', header="label")
np.savetxt('/Users/gofinge/Desktop/testSetC_pred2.csv', testY_pred2, delimiter=',', header="label")

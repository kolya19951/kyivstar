from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import numpy as np
trainY = [[], []]
trainX = []
testIDs = []
testX = []
base1 = {}
result = []
testHalfY = [[], []]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(4,), random_state=1)

def getSum(list1, list2):
    for i in range(len(list1)):
        list1[i] = list1[i] + list2[i]
    return list1


def init():
    global trainY, trainX, testIDs, testX, base1
    with open('train.txt', 'r+') as myFile:
        flag = True
        for line in myFile:
            if (flag):
                flag = False
                continue
            line = line.replace('\n', '')
            item = line.split('\t')
            item = [x if x != '' else 0 for x in item]
            item = [float(numeric_string) for numeric_string in item]
            trainY[0].append(item[0]);
            trainY[1].append(item[1]);
    with open('test.txt', 'r+') as myFile:
        flag = True
        for line in myFile:
            if (flag):
                flag = False
                continue
            line = line.replace('\n', '')
            testIDs.append(float(line));
            # trainY[1].append(line.split('\t')[1]);
    with open('Base1.txt', 'r+') as myFile:
        flag = True
        for line in myFile:
            if (flag):
                flag = False
                continue
            line = line.replace('\n', '')
            item = line.split('\t')
            item = [x if x != '' else 0 for x in item]
            item = [float(numeric_string) for numeric_string in item]
            if (not (item[0] in base1.keys())):
                base1[item[0]] = item[2:]
            else:
                base1[item[0]] = getSum(base1[item[0]], item[2:])
    trainX = [[] for i in range(len(trainY[0]))]
    testX = [[] for i in range(len(testIDs))]
    myFile.close()

    for i in range(0, len(testIDs)):
        testX[i] = base1[testIDs[i]]
        # print(len(testX[i]))

    for i in range(0, len(trainY[0])):
        # if(base1[trainY[i][0]] in trainY[0]):
        trainX[i] = base1[trainY[0][i]]
def study():
    global clf
    clf.fit(trainX, trainY[1])
    joblib.dump(clf, 'NN.pkl')
def load():
    global clf
    clf = joblib.load('NN.pkl')
def predict():
    global clf, result
    result = clf.predict(trainX)


def halfSetTest():
    global trainX, trainY, testX, base1, testHalfY
    j = int(len(trainY[0]) / 2)
    trainX = [[] for i in range(j)]
    for i in range(j):
        trainX[i] = base1[trainY[0][i]]
    testHalfY[0] = trainY[0][j:]
    testHalfY[1] = trainY[1][j:]
    trainY[0] = trainY[0][:j]
    trainY[1] = trainY[1][:j]
    testX = [[] for i in range(len(testHalfY[0]))]
    for i in range(len(testHalfY[0])):
        testX[i] = base1[testHalfY[0][i]]

init()
# load()
halfSetTest()
study()
predict()
count = 0;
for i in range(len(trainX)):
    if(result[i] == trainY[1][i]):
        count += 1
print((count/len(trainX))*100)

print(result)
thefile = open('result.txt', 'w')
for item in result:
    thefile.write("%s\n" % item)

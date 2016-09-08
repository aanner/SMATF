''' This is what it does '''
import numpy as np
from sklearn import svm

from tem_fun import nMovingAverage
from tem_fun import nCummulativeDerivative
from numpy import linalg as LA
import matplotlib.pyplot as plt


if __name__ == "__main__":
    '''
    Accessing data elements: data[i, j]
    Data order: DATE OPEN-0 HIGH-1 LOW-2 CLOSE-3 VOL-4 ADJCLOSE-5
    '''
    # GLOBAL PARAMETERS
    maxPast = 20
    daysDelta = 3
    datanumber = 5
    numFeatures = 1
    # IMPORTING GENERIC DATA
    data = genfromtxt('OMX.csv', delimiter=',', dtype=str)
    date = data[daysDelta:, 0]
    data = data[:-daysDelta, 1:].astype(np.float)
    ni = np.shape(data)[0]
    nj = np.shape(data)[1]
    print 'Datasize| (', ni, ', ', nj, ')'

    # CALCULATING LABELS
    labels = data[daysDelta:, datanumber]-data[:-daysDelta, datanumber]
    labels = labels[maxPast:]
    data = data[:-daysDelta, :]

    # Dividing in to testlabels and trainlabels
    numTestData = 100
    labelsTrain = labels[:-100]
    labelsPred = labels[-100:]

    for i in range(0, np.shape(labels)[0]):
        if labels[i] > 0:
            labels[i] = 1
        else:
            labels[i] = 0

    # CALCULATING FEATURES
    features = np.empty([ni-maxPast-daysDelta, numFeatures])
    # Feature 1: Moving average 10 days
    datanumber = 5
    numDays = 10
    mva = nMovingAverage(data[:, datanumber], numDays, maxPast)
    features[:, 0] = mva[:]
    plt.plot(features, labels, '.')
    plt.show()
    # # Feature 2:20 cummulative derivative
    # cud = nCummulativeDerivative(data[:, datanumber], numDays, maxPast)
    # features[:, 1:20] = cud[:, :]
    np.savetxt("features.csv", features, delimiter=",")
    # Dividing in to testlabels and trainlabels
    featuresTrain = features[:-100, :]
    featuresPred = features[-100:, :]
    # END CALCULATING FEATURES
    # print np.shape(featuresTrain), np.shape(featuresPred), np.shape(labelsTrain), np.shape(labelsPred)
    #

    clf = svm.SVC(gamma=0.01, C=100)
    clf.fit(featuresTrain, labelsTrain)
    prediction = clf.predict(featuresPred)

    percentRight = 0
    for i in range(0, np.shape(prediction)[0]):
        if prediction[i] == labelsPred[i]:
            percentRight = percentRight + 1
            print(percentRight)

    percentRight /= np.shape(prediction)[0]

    print 'Percent right: ', percentRight

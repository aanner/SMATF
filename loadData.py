from numpy import genfromtxt


def loadCSV(filename, delimiter):
    data = genfromtxt('OMX.csv', delimiter=',', dtype=str)
    data = data[:-daysDelta, 1:].astype(np.float)
    return data

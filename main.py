import matplotlib.image as img
import numpy as np
import os
import random

# weight matricies
w1 = np.ones((16,784)) # (neurons in first hidden layer, neurons in input layer)
w2 = np.ones((16,16))  # (neurons in second hidden layer, neurons in first hidden layer)
w3 = np.ones((10,16))  # (neurons in output, neurons in second hidden layer)

# bias matricies
b1 = np.ones((16,1)) # neurons in first hidden layer
b2 = np.ones((16,1)) # neurons in second hidden layer
b3 = np.ones((10,1)) # neurons in output layer

# input is the root folder to traverse
# returns an array of tuples of (
def traverseFolder(rootFolder):
    data = []
    for dirName, subdirList, fileList in os.walk(rootFolder):
        for fname in fileList:
            data.append((dirName[-1], dirName + '/' + fname))
    random.shuffle(data)
    return data

# input is training data from traverseFolder
# returns tuple of (b1, b2, b3, w1, w2, w3)
def trainNetwork(trains):
    curr = 0
    length = len(trains)
    for i in trains:
        print('Training: %' + str(100*curr/length))
        im = readData(i[1])
        res = calcFromNet(im)
        expect = np.zeros((10,1))
        expect[int(i[0])] = 1
        b1, b2, b3, w1, w2, w3 = backprop(res, expect, im)
        curr += 1
        del res
    return (b1, b2, b3, w1, w2, w3)

def testNetwork(tests):
    curr = 0
    length = len(tests)
    errors = [0,0,0,0,0,0,0,0,0,0]
    guesses = [0,0,0,0,0,0,0,0,0,0]
    for i in tests:
        im = readData(i[1])
        res = calcFromNet(im)
        guess = np.argmax(res[2])
        print(guess, res[2])
        guesses[guess] += 1
        if(guess != int(i[0])):
            errors[int(i[0])] += 1
    print(guesses)
    print(errors)

def readData(imPath = 'dataset/test/0/1001.png'):
    # im is a 1d array that is the image from left to right then top to bottom
    image = img.imread(imPath)
    return np.array(image).flatten().reshape(784,1)

# returns sigmoid of all values in array
def sigmoid(arr):
    return 1/(1 + np.exp(-arr))

# input is array with sigmoid applied
# returns derivative of sigmoid for all values in array
def sigmoidPrime(arr):
    return sigmoid(arr) * (np.ones((arr.size,1)) - sigmoid(arr))

# calculates the neural net
# returns a tuple of the activations for each layer
def calcFromNet(image):
    z1 = np.matmul(w1, image) + b1
    a1 = sigmoid(z1)
    z2 = np.matmul(w2, a1) + b2
    a2 = sigmoid(z2)
    z3 = np.matmul(w3, a2) + b3
    a3 = sigmoid(z3)
    return (a1, a2, a3)

# input is the return from calcFromNet, the expected output, and the original input
# adjusts all weight and bias values
def backprop(results, expected, inp):
    del3 = (results[2] - expected) * sigmoidPrime(results[2])
    del2 = np.matmul(w3.transpose(), del3) * sigmoidPrime(results[1])
    del1 = np.matmul(w2.transpose(), del2) * sigmoidPrime(results[0])
    # TODO: reduce loops done in python
    # adjust weights
    # adjust w3
    newW3 = np.ones((10,16))
    for j in range(10): # output layer
        for k in range(16): # second hidden layer
            newW3[j][k] = w3[j][k] + results[1][k] * del3[j]
    # adjust w2
    newW2 = np.ones((16,16))
    for j in range(16): # 
        for k in range(16): # 
            newW2[j][k] = w2[j][k] + results[0][k] * del2[j]
    # adjust w1
    newW1 = np.ones((16,784))
    for j in range(16): # 
        for k in range(784): # 
            newW1[j][k] = w1[j][k] + inp[k] * del1[j]
    # adjust bias
    return (b1 + del1, b2 + del2, b3 + del3, newW1, newW2, newW3)

testData = traverseFolder('dataset/test')
trainData = traverseFolder('dataset/training')
x = trainNetwork(trainData)
b1, b2, b3, w1, w2, w3 = x
testNetwork(testData)
print(trainData[0], trainData[-1])
print(testData[0], testData[-1])
print('b1',b1)
print('b2',b2)
print('b3',b3)
print('w1',w1)
print('w2',w2)
print('w3',w3)

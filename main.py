import matplotlib.image as img
import numpy as np
import os
import random
from tqdm import tqdm

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
def trainNetwork(trains, repeats):
    # define weight and bias matricies
    # weight matricies
    w1 = np.random.rand(16,784) # (neurons in first hidden layer, neurons in input layer)
    w2 = np.random.rand(16,16)  # (neurons in second hidden layer, neurons in first hidden layer)
    w3 = np.random.rand(10,16)  # (neurons in output, neurons in second hidden layer)

    # bias matricies
    b1 = np.random.rand(16,1) # neurons in first hidden layer
    b2 = np.random.rand(16,1) # neurons in second hidden layer
    b3 = np.random.rand(10,1) # neurons in output layer

    for i in tqdm(trains):
        im = readData(i[1])
        for j in range(repeats):
            res = calcFromNet(im, (b1, b2, b3, w1, w2, w3))
            expect = np.zeros((10,1))
            expect[int(i[0])] = 1
            b1, b2, b3, w1, w2, w3 = backprop(res, expect, im, (b1, b2, b3, w1, w2, w3))
            del res
        #break
    return (b1, b2, b3, w1, w2, w3)

def testNetwork(tests, matricies):
    curr = 0
    length = len(tests)
    errors = np.array([0,0,0,0,0,0,0,0,0,0])
    guesses = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in tests:
        im = readData(i[1])
        res = calcFromNet(im, (b1, b2, b3, w1, w2, w3))
        guess = np.argmax(res[2])
        guesses[guess] += 1
        if(guess != int(i[0])):
            errors[int(i[0])] += 1
    print("Guess distribution:", guesses)
    print("There were", np.sum(errors), "errors with the following distribution:", errors)
    return np.sum(errors)

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
def calcFromNet(image, matricies):
    b1, b2, b3, w1, w2, w3 = matricies
    z1 = np.matmul(w1, image) + b1
    a1 = sigmoid(z1)
    z2 = np.matmul(w2, a1) + b2
    a2 = sigmoid(z2)
    z3 = np.matmul(w3, a2) + b3
    a3 = sigmoid(z3)
    return (a1, a2, a3)

# input is the return from calcFromNet, the expected output, and the original input
# adjusts all weight and bias values
def backprop(results, expected, inp, matricies):
    b1, b2, b3, w1, w2, w3 = matricies
    del3 = (results[2] - expected) * sigmoidPrime(results[2])
    del2 = np.matmul(w3.transpose(), del3) * sigmoidPrime(results[1])
    del1 = np.matmul(w2.transpose(), del2) * sigmoidPrime(results[0])
    # adjust weights
    newW3 = w3 - np.transpose(np.repeat(results[1], 10, axis=1)) * np.repeat(del3, 16, axis=1)
    newW2 = w2 - np.transpose(np.repeat(results[0], 16, axis=1)) * np.repeat(del2, 16, axis=1)
    newW1 = w1 - np.transpose(np.repeat(inp, 16, axis=1)) * np.repeat(del1, 784, axis=1)
    # adjust bias
    # TODO: I think the del calculations aren't right
    return (b1 - del1, b2 - del2, b3 - del3, newW1, newW2, newW3)

testData = traverseFolder('dataset/test')
trainData = traverseFolder('dataset/training')

best = 0

for i in range(20):
    print("Running with 10 repeat in training")
    x = trainNetwork(trainData, 1)
    b1, b2, b3, w1, w2, w3 = x
    y = testNetwork(testData, x)
    if(best == 0 or best > y):
        best = y
    print()

print("Lowest errors was", best)

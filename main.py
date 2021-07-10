import matplotlib.image as img
import numpy as np

# weight matricies
w1 = np.ones((16,784))
w2 = np.ones((16,16))
w3 = np.ones((10,16))

# bias matricies
b1 = np.ones((16,1))
b2 = np.ones((16,1))
b3 = np.ones((10,1))

def readData(imPath = 'dataset/test/0/1001.png'):
    # im is a 1d array that is the image from left to right then top to bottom
    image = img.imread(imPath)
    return np.array(image).flatten().reshape(784,1)

def sigmoid(arr):
    return 1/(1 + np.exp(-arr))

def calcFromNet(image):
    a1 = sigmoid(np.matmul(w1, image) + b1)
    a2 = sigmoid(np.matmul(w2, a1) + b2)
    a3 = sigmoid(np.matmul(w3, a2) + b3)
    return a3

im = readData()
#print(im)
print(calcFromNet(im))

x = np.array([1,2,3])
x = x.reshape(3,1)
y = np.array([1,2,3])

# MLFromScratch
This project was a practice in understanding machine learning with backpropogation by creating a neural network that recognizes handwritten digits.  The main goal was to not use machine learning libraries like Scikit-learn.
# How it works
This project mostly follows [Michael Nielsen's book on neural networks](http://neuralnetworksanddeeplearning.com/) to implement the backpropogation and feed forward algorithms.  The only deviation from this book is the use of simulated annealing to achieve better results.  Finally, the dataset was gathered from [Kaggle's Handwritten Digits Images Dataset](https://www.kaggle.com/umegshamza/handwritten-digits-images-dataset?select=handwritten-digits-images-dataset).
# Results
The first revision of this program did not work well.  It had a runtime of about 25 minutes with an accuracy of only about 20%.  After adding in the simulated annealing, optimizations to the backpropogation algorithm, and random matrix initialization, this runtime was reduced to about 14 seconds with an accuracy of about 70%.
# Future work
In the future, there are two areas for improvement.  The first is to use a better cost function that can allow the network to get to the right answer faster.  The second is to generalize the network so that the network doesn't need 784 input nodes, 2 16 node hidden layers, and a 10 node output layer.

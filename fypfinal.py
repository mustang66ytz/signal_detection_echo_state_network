# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
import os
import random
from numpy import *
from scipy import linalg
from scipy.linalg import logm

m=500
n=1
 
def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-x))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=30000,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
args = vars(ap.parse_args())
#Initialize the teacher output y_train
y_train = np.array([[0.]*n]*m)
y_train[49, 0]=y_train[99, 0]=y_train[149, 0]=y_train[199, 0]=y_train[249, 0]=y_train[299, 0]=y_train[349, 0]=y_train[399, 0]=y_train[449, 0]=y_train[499, 0]=0.5
#Initialize the teacher input X
random.seed(24)
X = np.random.uniform(-0.5, 0.5, (500, 1))
for i in range(0, m):
    for j in range (1, 11):
        if ((i+j)%50==0):
            X[i, 0]+=np.random.uniform(1, 3)
#Initialize the test input X_test
random.seed(42)
X_test = np.random.uniform(-0.5, 0.5, (500, 1))
for i in range(0, m):
    for j in range (1, 11):
        if ((i+j)%50==0):
            X_test[i, 0]+=np.random.uniform(1, 3)
print ("training input", X)
#Print the test input and teacher output
print ("test input", X_test)
print ("teacher output", y_train)
 
# initialize our weight matrix, which is a square matrix to be trained, 
#its norm is scaled to acquire the echo state property.
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[0],X.shape[0]))
rhoW = max(abs(linalg.eig(W)[0]))
W *= 0.9/rhoW
# initialize a list to store the loss value for each epoch
lossHistory = []
#Training stage:
# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	# take the dot product between our features `X` and the
	# weight matrix `W`, then pass this value through the
	# sigmoid activation function, thereby giving us our
	# predictions on the dataset
	preds = sigmoid_activation(W.dot(X))
	# now that we have our predictions, we need to determine
	# our `error`, which is the difference between our predictions
	# and the true values
	error = preds - y_train
	# given our `error`, we can compute the total loss value as
	# the sum of squared loss -- ideally, our loss should
	# decrease as we continue training
	loss = np.sum(error ** 2)
	lossHistory.append(loss)
	print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))
	# the gradient update is the outer product between
	# the error and X, scaled by the total number of data points in `X`
	gradient = np.outer(error, X) / X.shape[0] 
	# in the update stage, all we need to do is nudge our weight
	# matrix in the negative direction of the gradient (hence the
	# term "gradient descent" by taking a small step towards a
	# set of "more optimal" parameters
	W += -args["alpha"] * gradient
#Test stage
print("[INFO] starting testing...")
for i in np.random.choice(500, 100):
	# compute the prediction by taking the dot product of the
	# weight matrix W and internal hidden state X, then
	# passing it through the sigmoid activation function
	activation = sigmoid_activation(W[i].dot(X_test))
	# the activation would bifurcate to approach 0 or 0.5 
	#to fit the desired teacher output
	label = 0 if activation < 0.02 else 0.5
	# show our outcome, the comparison between 
	#the predicted label and the true label
	print ("Activation", activation)
	print("predicted_label={}, true_label={}".format(
		label, y_train[i]))

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

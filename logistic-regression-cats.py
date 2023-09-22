import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

%matplotlib inline
%load_ext autoreload
%autoreload 2


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

# Standerdizing datasets
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# The Sigmoid function
def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    
    return s


# Function to intialize the arrays with zero

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    
    return w, b
# Propagate function that does forward propagation and backward propagation 
def propagate(w, b, X, Y):


    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    cost = (-1/m)*(np.sum((Y*np.log(A))+((1-Y)*np.log(1-A))))
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m) * (np.dot(X,(A-Y).T))
    db = (1/m) * (np.sum(A-Y))
    cost = np.squeeze(np.array(cost))

    #The gradients
    grads = {"dw": dw,
             "db": db}

    return grads, cost

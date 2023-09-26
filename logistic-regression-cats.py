import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
import os
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *


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

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
def predict(w, b, X):
    
    #creating the array that will store the predictions 
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # calculating the activations
    A = sigmoid(np.dot(w.T,X) + b)
    
    # looping through the data to check if the activation is greater than 0.5 or not to map it to its correct value
    for i in range(A.shape[1]):
        
       if(A[0,i] > 0.5):
            Y_prediction[0,i] = 1
       else:
            Y_prediction[0,i] = 0
        
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    
    # intializing the w and b 
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # retriving the values of params, grads, costs from the optimization function
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = params["w"]
    b = params["b"]
    
    # predicting the values of the training datasets
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)


    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

# running the model on the data set to check if it works with good effeciency or not
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()

# The path of the image
file_path = input("Enter the name of the image you want to check: ")
fname = "images/" + file_path

try:
    # We preprocess the image to fit your algorithm.
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

except FileNotFoundError:
    print(f"Error: The file '{fname}' was not found.")


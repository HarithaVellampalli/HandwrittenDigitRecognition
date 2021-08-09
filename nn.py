################################################################################
#
# LOGISTICS
#
#    <Haritha Vellampalli>
#
# FILE
#
#    <nn.py>
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. A summary of my nn.py code:
#
#       <Forward propagation implemented as - Matrix Multiplication, Activation, Matrix Multiplication, Activation, Matrix Multiplication & Softmax.>
#       <Cross Entropy is used as error.>
#       <Backpropagation is implemented for the above given forward propagation.>
#       <Optimization is used.>
#       <Implementation of Neural network includes additional features as - Batch input for training & Gradient descent with additional momentum>
#
#    2. Accuracy display
#
#       <Epoch 1: Training Loss = 0.4018508326307046
#       Epoch 2: Training Loss = 0.29070377131107483
#       Epoch 3: Training Loss = 0.23372695802049212
#       Epoch 4: Training Loss = 0.19985479357539634
#       Epoch 5: Training Loss = 0.15700619603510815
#       Epoch 6: Training Loss = 0.14140534245714478
#       Epoch 7: Training Loss = 0.12008778369237441
#       Epoch 8: Training Loss = 0.10405639461339879
#       Epoch 9: Training Loss = 0.09308675840315954
#       Epoch 10: Training Loss = 0.07973516033630064>
#       <Total Accuracy: 93.26>
#
#    3. Performance display
#
#       <6 mins>
#       <Input train size = 784, 60000; Input label size =  (10, 60000); Output train size = 784,        60000; Output label size =  (10, 60000); size_batch = 64; beta = 0.9; epochs = 10;  learning_rate = 0.5>
#
################################################################################

import os.path
import urllib.request
import gzip
import math
import numpy             as np
from sklearn.metrics import accuracy_score

################################################################################
#
# PARAMETERS
#
################################################################################

#Hyperparameters
size_batch = 64
beta = 0.9
epochs = 10
learning_rate = 0.5

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'
norm_const = 255

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

#One-hot Encoding of label data
y = train_labels.reshape(1, DATA_NUM_TRAIN)
train_labels = np.eye(10)[y.astype('int32')]
train_labels = train_labels.T.reshape(10, DATA_NUM_TRAIN)

y = test_labels.reshape(1, DATA_NUM_TEST)
test_labels = np.eye(10)[y.astype('int32')]
test_labels = test_labels.T.reshape(10, DATA_NUM_TEST)

#Normalizing to get values between 0 & 1
train_data = train_data/norm_const
test_data = test_data/norm_const

#Reshaping data
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1] * train_data.shape[2])
test_data = test_data.reshape(test_data.shape[0],test_data.shape[1] * test_data.shape[2])

#Transposing Data
train_data = train_data.T
test_data = test_data.T

# debug
print(train_data.shape)   # (784, 60000)
print(train_labels.shape) # (10, 60000)
print(test_data.shape)    # (784, 10000)
print(test_labels.shape)  # (10, 10000)

#Functions used in the training, forward and backward propagation

def reluAct(x):
  return np.maximum(x, 0)

def reluGrad(x):
  x[x > 0] = 1
  x[x <= 0]  = 0
  return x

def sigmoidAct(x):
  s = 1. / (1. + np.exp(-x))
  return s   

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

def crossentropy_loss(Y_star, Y):
  total = np.sum(np.multiply(Y_star, np.log(Y)))
  i = Y_star.shape[1]
  result = -(1./i) * total
  return result

# Initialization
# np.random.seed(123)

hidden_nodes = 1000
hidden_nodes1 = 100
digits = 10
input_nodes = 784
params = {"W1": np.random.randn(hidden_nodes, input_nodes),
          "b1": np.zeros((hidden_nodes, 1)) ,
          "W2": np.random.randn(hidden_nodes1, hidden_nodes),
          "b2": np.zeros((hidden_nodes1, 1)),
          "W3": np.random.randn(digits, hidden_nodes1),
          "b3": np.zeros((digits, 1)) }

#Feed Forward propagation function
def forward_prop(X, params):
    model = {}

    model["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    # model["A1"] = reluAct(model["Z1"]) 
    #Sigmoid used as it gives better results
    model["A1"] = sigmoidAct(model["Z1"])

    model["Z2"] = np.matmul(params["W2"], model["A1"]) + params["b2"]
    # model["A2"] = reluAct(model["Z2"])
    model["A2"] = sigmoidAct(model["Z2"])

    model["Z3"] = np.matmul(params["W3"], model["A2"]) + params["b3"]
    model["A3"] = softmax(model["Z3"])
    return model

#Back propagation function
def backward_prop(X, Y, params, model, batch_num):
    # error at last layer
    dZ3 = model["A3"] - Y
    dW3 = (1. / batch_num) * np.matmul(dZ3, model["A2"].T)
    db3 = (1. / batch_num) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.matmul(params["W3"].T, dZ3)
    # dZ2 = dA2 * reluGrad(model["Z2"])
    dZ2 = dA2 * sigmoidAct(model["Z2"]) * (1-sigmoidAct(model["Z2"]))
    dW2 = (1. / batch_num) * np.matmul(dZ2, model["A1"].T)
    db2 = (1. / batch_num) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    # dZ1 = dA1 * reluGrad(model["Z1"])
    dZ1 = dA1 * sigmoidAct(model["Z1"]) * (1-sigmoidAct(model["Z1"]))
    dW1 = (1. / batch_num) * np.matmul(dZ1, X.T)
    db1 = (1. / batch_num) * np.sum(dZ1, axis=1, keepdims=True)

    out_val = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return out_val

# Train

#Initialize
mom_dW1 = np.zeros(params["W1"].shape)
mom_db1 = np.zeros(params["b1"].shape)
mom_dW2 = np.zeros(params["W2"].shape)
mom_db2 = np.zeros(params["b2"].shape)
mom_dW3 = np.zeros(params["W3"].shape)
mom_db3 = np.zeros(params["b3"].shape)

for count_out in range(epochs):

    # Shuffling Dataset in batches
    # np.random.seed(456)
    random_rows = np.random.permutation(train_data.shape[1])
    train_data_shuf = train_data[:, random_rows]
    train_labels_shuf = train_labels[:, random_rows]
    batch_count = -(-DATA_NUM_TRAIN // size_batch)

    for count_in in range(batch_count):

        start_num = count_in * size_batch
        end_num = min(start_num + size_batch, train_data.shape[1] - 1)
        data_train = train_data_shuf[:, start_num:end_num]
        data_test = train_labels_shuf[:, start_num:end_num]
        batch_num = end_num - start_num

        # Performing Feed Forward Propagation & Backward propagation
        model = forward_prop(data_train, params)
        out_val = backward_prop(data_train, data_test, params, model, batch_num)

        # Optimization - Adding Momentum Term
        mom_dW1 = (beta * mom_dW1 + (1. - beta) * out_val["dW1"])
        mom_db1 = (beta * mom_db1 + (1. - beta) * out_val["db1"])
        mom_dW2 = (beta * mom_dW2 + (1. - beta) * out_val["dW2"])
        mom_db2 = (beta * mom_db2 + (1. - beta) * out_val["db2"])
        mom_dW3 = (beta * mom_dW3 + (1. - beta) * out_val["dW3"])
        mom_db3 = (beta * mom_db3 + (1. - beta) * out_val["db3"])

        # Optimization - Using the Gradient Descent
        params["W1"] = params["W1"] - learning_rate * mom_dW1
        params["b1"] = params["b1"] - learning_rate * mom_db1
        params["W2"] = params["W2"] - learning_rate * mom_dW2
        params["b2"] = params["b2"] - learning_rate * mom_db2
        params["W3"] = params["W3"] - learning_rate * mom_dW3
        params["b3"] = params["b3"] - learning_rate * mom_db3

    # Calculating Loss on Training Data
    model = forward_prop(train_data, params)
    loss_train = crossentropy_loss(train_labels, model["A3"])

    #Display Epoch and Loss
    print("Epoch {}: Training Loss = {}".format(count_out + 1, loss_train))

#Predicting accuracy of test data
model = forward_prop(test_data, params)
predicted = np.argmax(model["A3"], axis=0)
actual = np.argmax(test_labels, axis=0)

print(accuracy_score(actual, predicted))

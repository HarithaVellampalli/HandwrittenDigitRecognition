# Handwritten Digit Recognition

The handwriting recognition dataset used for the project is the MNIST dataset, which is constructed from normalized and centered, scanned images from several documents of numbers from 0-9. It is a fundamental dataset used for introductory image classification projects.

The aim of the project is to perform image classification using a traditional neural network created from scratch by only using python core modules as NumPy . The use of xNN packages as PyTorch or TensorFlow is not allowed. This enables us to understand the architecture and the fundamentals of the neural network better. 

The pseudo code is as follows:
```
#Cycle through the epochs

    #Set learning rate
    
    #Cycle through training data
        #Forward pass
        #Loss function
        #Back Propagation
        #Weight update

    #Cycle through testing data
        #Forward pass
        #Accuracy

    #Results displayed per epoch

#Final accuracy, plots, and performance
```

Each forward pass would have the following network structure with output sizes as given:
Layer  | Output
------------- | -------------
Data  | 1 x 28 x 28
Division | 1 x 28 x 28
Vectorization | 1 x 784
Matrix Multiplication | 1 x 1000
Addition | 1 x 1000
ReLU | 1 x 1000
Matrix Multiplication | 1 x 100
Addition | 1 x 100
ReLU | 1 x 100
Matrix Multiplication | 1 x 10
Addition | 1 x 10
Softmax | 1 x 10

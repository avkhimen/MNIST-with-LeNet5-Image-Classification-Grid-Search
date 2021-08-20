#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tensorflow and keras imports

from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

# Other imports

import numpy as np
import random

# from tqdm import tqdm

# Define the number of dataset classes, k-folds, and the dataset.

number_of_dataset_classes = 10
number_of_K_folds = 10
dataset = 'mnist'


# Wrapper for the mnist.load_data() functions.
# Returns (X_train, Y_train), (X_test, Y_test).

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        ((X_train, Y_train), (X_test, Y_test)) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        ((X_train, Y_train), (X_test, Y_test)) = \
            fashion_mnist.load_data()
    return ((X_train, Y_train), (X_test, Y_test))


# Separates (image, label) into k folds.
# Returns a tuple of (X_folds, Y_folds).

def separate_dataset_into_K_folds(X_train, Y_train, number_of_K_folds):
    if number_of_K_folds == 10:
        folds = get_10_folds(X_train, Y_train)
    elif number_of_K_folds == 5:
        folds = get_5_folds(X_train, Y_train)
    return folds


# Separates (image, label) into 10 folds.
# Returns a tuple of (X_folds, Y_folds) of 10 folds in each.

def get_10_folds(X_train, Y_train,
                 number_of_dataset_classes=number_of_dataset_classes):
    dataset_classes = get_dataset_classes(X_train, Y_train,
            number_of_dataset_classes)
    X_folds = [[],[],[],[],[],[],[],[],[],[],]
    Y_folds = [[],[],[],[],[],[],[],[],[],[],]
    for (jj, dataset_class) in enumerate(dataset_classes):
        image_index = 0
        while image_index < len(dataset_class):
            for (ii, fold) in enumerate(X_folds):
                try:
                    X_folds[ii].append(dataset_class[image_index])
                    Y_folds[ii].append(jj)
                    image_index += 1
                except Exception, e:
                    continue
    for (ii, fold) in enumerate(X_folds):
        X_folds[ii] = np.array(fold)
    for (ii, fold) in enumerate(Y_folds):
        Y_folds[ii] = np.array(fold)
    X_folds = np.array(X_folds)
    Y_folds = np.array(Y_folds)
    for (ii, X_fold) in enumerate(X_folds):
        Y_fold = Y_folds[ii]
        indices = np.arange(X_fold.shape[0])
        np.random.shuffle(indices)
        X_fold = X_fold[indices]
        Y_fold = Y_fold[indices]
        X_folds[ii] = X_fold
        Y_folds[ii] = Y_fold

    return (X_folds, Y_folds)


# Separates the (image, label) into 5 folds; not implemented.

def get_5_folds(X_train, Y_train):
    pass


# Separates the image training data into numpy arrays
# where every image in each numpy array is from the same class.
# Returns the numpy array with each index of the array corresponding
# to the label.

def get_dataset_classes(X_train, Y_train, number_of_dataset_classes):
    if number_of_dataset_classes == 10:
        dataset_classes = [[],[],[],[],[],[],[],[],[],[],]
    elif number_of_dataset_classes == 5:
        dataset_classes = [[], [], [], [], []]
    for dataset_class_index in range(number_of_dataset_classes):
        for item in range(X_train.shape[0]):
            if Y_train[item] == dataset_class_index:
                dataset_classes[dataset_class_index].append(X_train[item])
    return np.array(dataset_classes)


#Creates k sets of (x_train, y_train, x_test, y_test)
#constructed from the X_train and Y_train data
#Returns k sets of (x_train, y_train, x_test, y_test)
#which can be used to run k-fold cross-validation on

def create_fold_iterables(X_folds, Y_folds):
    iterables = []
    for (ii, val_fold) in enumerate(X_folds):
        X_stack = 0
        Y_stack = 0
        for (jj, train_fold) in enumerate(X_folds):
            if ii != jj:
                if type(X_stack) is int:
                    X_stack = train_fold
                    Y_stack = Y_folds[jj]
                else:
                    X_stack = np.vstack((X_stack, train_fold))
                    Y_stack = np.hstack((Y_stack, Y_folds[jj]))
        iterables.append([X_stack, Y_stack, val_fold, Y_folds[ii]])
    iterables = np.array(iterables)
    return iterables


# Shuffle the images and labels to ensure the results
# beteen different models configurations aren't biased.
# Returns (X_train, Y_train).

def shuffle_set(X_train, Y_train):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    return (X_train, Y_train)


# Creates the LeNet-5-like model defined by the kernel size,
# dropout (always set to False here), and the number of layers.
# Returns the defined model.

def create_lenet_model(kernel_size, dr, layers):

    model = Sequential()

    # Adding a Convolution Layer

    model.add(Conv2D(filters=6, kernel_size=(kernel_size, kernel_size),
              padding='valid', input_shape=(28, 28, 1),
              activation='relu'))

    # Adding a Pooling Layer

    model.add(MaxPool2D(pool_size=(2, 2)))

    # Adding a Convolution Layer

    if layers >= 2:
        model.add(Conv2D(filters=16, kernel_size=(kernel_size,
                  kernel_size), padding='valid', activation='relu'))

    # Adding a Pooling Layer

    if layers >= 2:
        model.add(MaxPool2D(pool_size=(2, 2)))

    if dr:
        model.add(Dropout(0.2))

    # Adding a Convolution Layer

    if layers >= 3 and kernel_size == 3:
        model.add(Conv2D(filters=120, kernel_size=(kernel_size,
                  kernel_size), padding='valid', activation='relu'))

    # Adding a Pooling Layer

    if layers >= 3 and kernel_size == 3:
        model.add(MaxPool2D(pool_size=(2, 2)))

    # Flattening the layer S

    model.add(Flatten())

    # Adding a Dense layer with `ReLU` activation

    model.add(Dense(120, activation='relu'))

    # Adding a Dense layer with `softmax` activation

    model.add(Dense(10, activation='softmax'))

    return model

#Compiles the model with training data using the given
#optimizer, learning rate, and loss function
#Returns the trained model

def compile_and_fit_model(
    model,
    train_x,
    train_y,
    opt,
    lr,
    loss_function,
    ):

    # Reshape data

    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)

    # Normalize data

    train_x = train_x / 255.0

    # One-hot encode the labels

    train_y = to_categorical(train_y, num_classes=10)

    if opt == 'Adam':
        opt = Adam(learning_rate=lr)
    elif opt == 'SGD':
        opt = SGD(learning_rate=lr)
    elif opt == 'RMSprop':
        opt = RMSprop(learning_rate=lr)

    if loss_function == 'categorical_crossentropy':
        loss = 'categorical_crossentropy'
    elif loss_function == 'kl_divergence':
        loss = 'kl_divergence'

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=128, epochs=20, verbose=1)
    return model

#Evaluates the model's accuracy
#Prints the model's accuracy

def evaluate_model(model, test_x, test_y):
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

    test_x = test_x / 255.0
    test_y = to_categorical(test_y, num_classes=10)

    score = model.evaluate(test_x, test_y, batch_size=128)
    print ('Test Loss:', score[0])
    print ('Test accuracy:', score[1])


# Get (X_train, Y_train), (X_test, Y_test) from the dataset

((X_train, Y_train), (X_test, Y_test)) = get_dataset(dataset)

# Define values in learning rates, optimizers, number of
# layers, kernel sizes, and loss functions.
# These will define the configurations for the model.

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
optimizers = ['Adam', 'SGD', 'RMSprop']
num_layers = [1, 2, 3]
kernel_sizes = [3, 5, 7]
add_dropout = [False]
loss_functions = ['categorical_crossentropy', 'kl_divergence']

# Perform grid search on the combinations of parameters that define the model

for lr in learning_rates:
    for layers in num_layers:
        for kernel_size in kernel_sizes:
            for dr in add_dropout:
                for loss_function in loss_functions:
                    for opt in optimizers:
                        scores = []

                        # Shuffle the (X_train, Y_train) to reduce bias

                        (X_train, Y_train) = shuffle_set(X_train,
                                Y_train)

                        # Get k folds for calculating the mean accuracy and standard deviation
                        # of each model

                        (X_folds, Y_folds) = \
                            separate_dataset_into_K_folds(X_train,
                                Y_train, 10)

                        # Create mini datasets to enable model evaluation

                        iterables = create_fold_iterables(X_folds,
                                Y_folds)

                        # for iterable in tqdm(iterables):

                        for iterable in iterables:
                            if kernel_size == 7 and layers == 3 \
                                or kernel_size == 5 and layers == 3:
                                continue
                            (K_fold_X_train, K_fold_Y_train,
                             K_fold_X_test, K_fold_Y_test) = iterable
                            val_x = \
                                K_fold_X_test.reshape(K_fold_X_test.shape[0],
                                    28, 28, 1)
                            val_y = to_categorical(K_fold_Y_test,
                                    num_classes=10)
                            model = create_lenet_model(kernel_size, dr,
                                    layers)
                            model = compile_and_fit_model(
                                model,
                                K_fold_X_train,
                                K_fold_Y_train,
                                opt,
                                lr,
                                loss_function,
                                )
                            score = model.evaluate(val_x, val_y,
                                    verbose=0)
                            scores.append(score[1])
                        f = open('fashion_mnist_run1_stats.txt', 'a')
                        s = '\n' + str(lr) + '|' + str(layers) + '|' \
                            + str(kernel_size) + '|' + str(dr) + '|' \
                            + str(loss_function) + '|' + str(opt) + '|' \
                            + str(np.mean(scores)) + '|' \
                            + str(np.std(scores))
                        print (s, np.mean(scores), '|', np.std(scores),
                               scores)
                        f.write(s)
                        f.close()


"""
# Define kernel size, number of layers, optimzier,
# learning rate, and loss function to train and
# evaluate the model

kernel_size = 5
layers = 2
dr = False
opt = 'SGD'
lr = 0.1
loss_function = 'kl_divergence'

# Create model

model = create_lenet_model(kernel_size, dr, layers)

# Train the model

model = compile_and_fit_model(
    model,
    X_train,
    Y_train,
    opt,
    lr,
    loss_function,
    )

# Evaluate the model on the X_test, Y_test data

evaluate_model(model, X_test, Y_test)
"""
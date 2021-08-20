# MNIST-with-LeNet5-Image-Classification-Grid-Search
MNIST with LeNet5 Image Classification Grid Search

The code performs grid search to determine the highest classification accuracy for MNIST and Fashion MNIST datasets using a modified LeNet5 architecture.

The following parameters were varied:

* learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
* optimizers = ['Adam', 'SGD', 'RMSprop']
* num_layers = [1, 2, 3]
* kernel_sizes = [3, 5, 7]
* add_dropout = [False]
* loss_functions = ['categorical_crossentropy', 'kl_divergence']

To run the code in terminal do:

python main_final.py

To change the dataset being worked on adjust the `dataset` variable between *mnist* and *fashion_mnist*

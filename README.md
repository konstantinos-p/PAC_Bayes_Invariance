# PAC_Bayesian_Generalization
Code from the paper "PAC-Bayesian Margin Bounds for convolutional neural networks".

<B>GE_estimation_dense.py</B>: Trains a dense neural network and estimates the network training and testing accuracy.

<B>GE_estimation_locally_connected.py</B>: Trains a locally connected neural network and estimates the network training and testing accuracy.

<B>GE_estimation_convolutional.py</B>: Trains a convolutional neural network and estimates the network training and testing accuracy.

<B>utils_spectral_norm.py</B>: Includes an implementation of Parseval regularization.

<B>test_spectral_convolutional_variable_ab.py</B>: Calculates the spectral norm for locally connected and convolutional 1d layers and computes empirical averages. Draws the theoretical average as well.

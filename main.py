import numpy as np
# import pandas as pd

# This is a neural network chess position evaluator.
# 7 encoding per square, 64 square -> 7 * 64 = 448


def initialize_parameters(layer_dimensions):
    parameters = {}
    for i in range(len(layer_dimensions)-1):
        parameters["W" + str(i+1)] = np.random.randn(layer_dimensions[i+1], layer_dimensions[i]) * 0.01
        parameters["b" + str(i+1)] = np.zeros((layer_dimensions[i+1], 1))
    
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(A, W) + b
    cache = (A, W, b)

    return Z, cache


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z # (A, cache = Z)

def relu(Z):
    return max(Z, 0), Z # (A, cache = Z)

layer_dimensions = (448, 20, 15, 1)
params = initialize_parameters(layer_dimensions)

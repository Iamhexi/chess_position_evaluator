import copy
import numpy as np
import pandas as pd
import chess

# fen - string in the FEN chess notation
def fen_to_input_vector(fen): 
    board = chess.Board(fen)
    board_array = []

    for i in range(8):
        for j in range(8):
            square = chess.square(j, i)

            piece = board.piece_at( square )

            if piece is None:
                indexOfOne = 0
            else:
                indexOfOne = (int) (piece.piece_type) # as piece type's numeric values correspond to position in the binary vector
            

            for k in range(7):
                if k == indexOfOne:
                    board_array.append(True)
                else:
                    board_array.append(False)




    if board.turn == chess.WHITE:
        board_array.append(True)
    else:
        board_array.append(False)

    vec = np.array(board_array)
    return vec.reshape((449, 1)) # check readme.md for a detalied explanation


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

def linear_activation_forward(A, W, b, activation):
    Z, linear_cache = linear_forward(A, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 # division by two (each layer has WL and bL) with trucate

    # layers using relu - it means all excepts the last one
    for i in range(1, L): # from 1 to L-1 as Python skips the last one
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], 'relu')
        caches.append(cache)

    linear_activation_forward(A, parameters[f'W{L}'], parameters[f'W{L}'], 'sigmoid')

    return A, caches

def log_cost(Y_hat, Y): # Y_hat is the activation value of the last layer
    number_of_training_examples = Y.shape[1] # number of training examples also called `m`
    cost = - np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 -Y_hat)) 
    return np.squeeze(cost) # convert one dimensional vector to a real number

def L2_cost(AL, Y): # AL = Y_hat, activation of the last layer of NN
    cost = np.sum( np.square(AL - Y) )
    return np.squeeze(cost)

# dz - gradient of cost of a current layer
# cache: (A_prev, W, b) from a forward pass
def linear_backward(dZ, cache): 
    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m # axis=1 -> columns, keepdims -> keeps the original shape
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z)) # not dot product as it would sum the values at the end

def sigmoid_backward(dA, activation_cache): # activation cache from the sigmoid function call
    Z = activation_cache
    dZ = dA * sigmoid_derivative(Z)
    return dZ

def relu_derivative(Z): # nice trick to calculate the derivative of relu using comparison
    return (Z >= 0).astype(float) # convert to float matrix, assumption: derivative of relu is 1 for Z = 0
    
def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = dA * relu_derivative(Z)
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation_cache == 'relu':
        dZ = relu_backward(dA, activation_cache)
        return linear_backward(dZ, linear_cache)

    elif activation_cache == 'sigmoid':
        dZ = relu_backward(dA, activation_cache)
        return linear_backward(dZ, linear_cache)
    
    # returns dA_prev, dW, db from linear_backward function

def model_backward(AL, Y, caches):
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)


    # last layer
    current_cache = caches[L-1] 
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # All previous layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(params, gradients, learning_rate):
    parameters = copy.deepcopy(params) # a copy instead of a reference
    L = len(parameters) // 2

    for i in range(1, L+1):
        parameters[f'W{i}'] = parameters[f'W{i}'] - learning_rate * gradients[f'dW{i}']
        parameters[f'b{i}'] = parameters[f'b{i}'] - learning_rate * gradients[f'db{i}']

    return parameters

def model(X, Y, layer_dimensions, learning_rate = 0.005, iterations = 3000, loss_function='L2'):
    parameters = initialize_parameters(layer_dimensions)
    
    for i in range(iterations):
        AL, caches = model_forward(X, parameters)
        
        if loss_function == 'L2':
            cost = L2_cost(AL, Y)
        elif loss_function == 'log':
            cost = log_cost(AL, Y)

        gradients = model_backward(AL, caches)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if iterations % 100 == 0 or i == iterations:
            cost = np.square(cost)
            print(f'Cost for {i} iteration: {cost}')



# preprocess data

# load X
# load Y
            
print(fen_to_input_vector("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
            
# divide into the training set and the test set

layer_dimensions = (448, 20, 15, 1)
# model(
#     X=,
#     Y=
#     layer_dimensions=layer_dimensions,
#     iterations=2000,
#     loss_function='L2' # or 'log'
# )

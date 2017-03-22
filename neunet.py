# -*- coding: utf-8 -*-
"""
Name :- Ravi Shekhar Singh
Date :- 20/03/2017

File :- neunet.py
dataset :- seeds_dataset
Description :- implement the neural network
               BackProportion algorithm

Reference :-
    http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

Version :- 0.2 
Licence :- GPLv3

"""
from random import seed
from random import random
from math import exp
from csv import reader
from random import randrange

# Initialize the neural network
#   It creates a new neural network ready for training. 
#   It accepts three parameters, the number of inputs, 
#   the number of neurons to have in the hidden layer 
#   and the number of outputs.
def init_network(num_of_input, num_of_hidden_layer, num_of_output):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(num_of_input+1)]} 
                    for i in range(num_of_hidden_layer)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(num_of_input+1)]} 
                    for i in range(num_of_output)]
    network.append(output_layer)
    return network

# test the initialization
'''    
seed(1)
network = init_network(2,1,2)
print('\n:::Testing Initialization of Weights:::')
for layer in network:
    print(layer)
'''
 
# forward propagate the network
#   1. Neuron Activation
#         activation = sum(weight_i * input_i) + bias
#   2. Neuron Transfer.
#         output = 1 / (1 + e^(-activation))
#   3. Forward Propagation.
#
 
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation
    
def transfer(activation):
    return 1.0/(1.0 + exp(-activation))
    
def forward_propagate(network, row):
    input1 = row
    for layer in network:
        input2 = []
        for neuron in layer:
            activation = activate(neuron['weights'], input1)
            neuron['output'] = transfer(activation)
            input2.append(neuron['output'])
        input1 = input2
    return input1
    
# test forward propagation (data is from above initiazation test)
'''
print('\n:::Testing Forward Propagation:::')
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, 
           {'weights': [0.4494910647887381, 0.651592972722763]}]]
           
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)
'''

# Back Propagate Error
#   1. Transfer Derivative
#           calculate its slope, sigmoid trasfer function
#           derivative = output * (1.0 - output)
#   2. Error Backpropagation
#           calculate the error for each output neuron   
#           error = (expected - output) * transfer_derivative(output) 
#           in hidden layer, calculated as the weighted error of each 
#           neuron in the output layer          
#           error = (weight_k * error_j) * transfer_derivative(output)

def transfer_derivative(output):
    return output*(1.0-output)
    
def back_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        error = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                err = 0.0
                for neuron in network[i+1]:
                    err += (neuron['weights'][j]*neuron['delta'])
                error.append(err)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                error.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = error[j]*transfer_derivative(neuron['output'])

# test backward propagation error
'''
print('\n:::Testing Backward Propagation Error:::')
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 
            0.8474337369372327, 0.763774618976614]}],[{'output': 0.6213859615555266,
            'weights': [0.2550690257394217, 0.49543508709194095]}, 
            {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]

expected = [0, 1]
back_propagate_error(network, expected)
for layer in network:
	print(layer)
'''

# Training the Network 
#   1. Update Weights
#           update networ weights with the calculated errors
#           the network is updated using stochastic gradient descent
#           weight = weight + learning_rate * error * input
#   2. Train Network
#           Train a network for a fixed number of epochs
#

def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inp = row[:-1]
        if i != 0:
            inp = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inp)):
                neuron['weights'][j] += learning_rate * neuron['delta']*inp[j]
            neuron['weights'][-1] += learning_rate*neuron['delta']

def train_network(network, train, lrate, num_epoch, num_output):
    for epoch in range(num_epoch):
        sum_err = 0
        for row in train:
            out = forward_propagate(network, row)
            expected = [0 for i in range(num_output)]
            expected[row[-1]] = 1
            sum_err += sum([(expected[i]-out[i])**2 for i in range(len(expected))])
            back_propagate_error(network, expected)
            update_weights(network, row, lrate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lrate, sum_err))
        
# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = init_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
print('\n')
for layer in network:
	print(layer)
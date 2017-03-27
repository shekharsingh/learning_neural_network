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

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

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
'''
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
'''
 
# Predict the output
#   arg-max function
def predict_output(network, row):
    output = forward_propagate(network,row)
    return output.index(max(output))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = init_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict_output(network, row)
		predictions.append(prediction)
	return(predictions)
 
# Test making predictions with the network
'''
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
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict_output(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
 '''



 
# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
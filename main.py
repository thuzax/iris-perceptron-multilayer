"""
SECTION 1 : Load and setup data for training
"""

import random
import math
random.seed(123)


# Load file
def read_data_set(data_set):
    with open(data_set) as data_file:
        data_set = data_file.read().split("\n")
        for i in range(len(data_set)):
            data_set[i] = data_set[i].split(",")

    return data_set

def get_classification_value(classification):
    if(classification == "Iris-setosa"):
        return 0
    if(classification == "Iris-versicolor"):
        return 1
    if(classification == "Iris-virginica"):
        return 2

def change_string_to_float(string_list):
    float_list = []
    for i in range(len(string_list)):
        float_list.append(float(string_list[i]))
    return float_list


data_set = read_data_set("iris.txt")

# Change string value to numeric
for line in data_set:
    line[4] = get_classification_value(line[4])
    line[:4] = change_string_to_float(line)

print(data_set)

# Split x and y (feature and target)
random.shuffle(data_set)

data_train = data_set[:int(len(data_set) * 0.5)]
data_test = data_set[int(len(data_set) * 0.5):]

train_set = []
train_result = []
for data in data_train:
    train_set.append(data[:4])
    train_result.append(data[4])

test_set = []
test_result = []
for data in data_test:
    test_set.append(data[:4])
    test_result.append(data[4])

# Define parameter
alpha = 0.005
epoch = 400
neuron = [4, 4, 3] # number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 96.67%
"""

def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = []
    for i in range(len(A)):
        C.append([])
        for j in range(len(B[0])):
            C[i].append(0)
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            
            C[i][j] += bias[j]
    
    return C


def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = []
    for i in range(len(B[0])):
        C.append(0)
    
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    
    return C


def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = []
    for i in range(len(A)):
        C.append(0)
    
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    
    return C


def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A


# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1


for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_set): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_result[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror +=  0.5 * (target[i] - X_2[i]) ** 2 
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alpha * (delta_2[j] * X_1[i])
                bias_2[j] -= alpha * delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        
        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -=  alpha * (delta_1[j] * x[i])
                bias[j] -= alpha * delta_1[j]
    
    cost_total /= len(train_set)
    if(e % 100 == 0):
        print(cost_total)

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_set, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print(preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_result[i]):
        acc += 1
print(acc / len(preds) * 100, "%")
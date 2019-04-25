import random
import math
import copy
random.seed(1)

# Matrix multiplication (for Testing)
def matrix_mul_bias(A, B, bias):
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


# Vector (A) x matrix (B) multiplication
def vec_mat_bias(A, B, bias):
    C = []
    for i in range(len(B[0])):
        C.append(0)
    
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
        C[j] += bias[j]
    
    return C


# Matrix (A) x vector (B) multipilicatoin (for backprop)
def mat_vec(A, B): 
    C = []
    for i in range(len(A)):
        C.append(0)
    
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    
    return C


# derivation of sigmoid (for backprop)
def sigmoid(A, deriv=False):
    if isinstance(A, list):
        if deriv: 
            for i in range(len(A)):
                A[i] = A[i] * (1 - A[i])
        else:
            for i in range(len(A)):
                A[i] = 1 / (1 + math.exp(-A[i]))
    else:
        if deriv: 
            A = A * (1 - A)
        else:
            A = 1 / (1 + math.exp(-A))
    return A

# Main funciton
if __name__=="__main__":

    # Define parameter
    alpha = 0.01
    epoch = 1000
    neurons = [2, 3, 2] # number of neurons each layer


    # Initiate weight and bias with 0 value
    weights = []
    for i in range(len(neurons) - 1):
        weights.append([])
        for j in range(neurons[i]):
            weights[i].append([])
            for k in range(neurons[i + 1]):
                weights[i][j].append(0)
    
    weight = []
    for i in range(len(weights[0])):
        weight.append([])
        for j in range(len(weights[0][i])):
            weight[i].append(weights[0][i][j])

    weight_2 = []
    for i in range(len(weights[1])):
        weight_2.append([])
        for j in range(len(weights[1][i])):
            weight_2[i].append(weights[1][i][j])

    
    bias_list = []
    for i in range(1, len(neurons)):
        bias_list.append([])
        for j in range(neurons[i]):
            bias_list[i-1].append(0)

    bias = []
    for i in range(len(bias_list[0])):
        bias.append(bias_list[0][i])

    bias_2 = []
    for i in range(len(bias_list[1])):
        bias_2.append(bias_list[1][i])

    # Initiate weight with random between -1.0 ... 1.0
    for i in range(neurons[0]):
        for j in range(neurons[1]):
            weight[i][j] = 2 * random.random() - 1

    for i in range(neurons[1]):
        for j in range(neurons[2]):
            weight_2[i][j] = 2 * random.random() - 1

    xor_test_set = [[0,0], [0,1], [1,0], [1,1]]
    xor_test_set_result = [0,1,1,0]

    for e in range(epoch):
        cost_total = 0
        for idx, data_list in enumerate(xor_test_set): # Update for each data; SGD
            
            
            # Forward propagation
            h_1 = vec_mat_bias(data_list, weight, bias)
            X_1 = sigmoid(h_1)
            h_2 = vec_mat_bias(X_1, weight_2, bias_2)
            X_2 = sigmoid(h_2)
            


            # Convert to One-hot target
            target = [0] * neurons[-1]
            target[int(xor_test_set_result[idx])] = 1


            # Cost function, Square Root Eror
            eror = 0
            for i in range(neurons[-1]):
                eror +=  0.5 * (target[i] - X_2[i]) ** 2 
            cost_total += eror

            # Backward propagation
            # Update weight_2 and bias_2 (layer 2)



            delta_2 = []
            for j in range(neurons[2]):
                delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

            # print(delta_2)
            

            for i in range(neurons[1]):
                for j in range(neurons[2]):
                    weight_2[i][j] -= alpha * (delta_2[j] * X_1[i])
                    bias_2[j] -= alpha * delta_2[j]
            
            delta_1 = mat_vec(weight_2, delta_2)
            for j in range(neurons[1]):
                delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
            
            
            # Update weight and bias (layer 1)
            delta_1 = mat_vec(weight_2, delta_2)
            for j in range(neurons[1]):
                delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
                # delta_1[j] = delta_1[j] * sigmoid(X_1[j], deriv=True)
            
            for i in range(neurons[0]):
                for j in range(neurons[1]):
                    weight[i][j] -=  alpha * (delta_1[j] * data_list[i])
                    bias[j] -= alpha * delta_1[j]
        

        cost_total /= len(xor_test_set)
        if(e % 100 == 0):
            print(cost_total)

    res = matrix_mul_bias(xor_test_set, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias_2)

    # Get prediction
    preds = []
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])

    for i in range(len(xor_test_set_result)):
        xor_test_set_result[i] = int(xor_test_set_result[i])
    # Print prediction
    print("Resultado esperado: ", xor_test_set_result)
    print("Predição:", preds)


    # Calculate accuration
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(xor_test_set_result[i]):
            acc += 1
    print(acc / len(preds) * 100, "%")
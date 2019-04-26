import math
import random
import copy
random.seed(20)

class Perceptron:
    def __init__(self, size_input, bias=1):
        self.size_input = size_input
        self.weights = []
        for i in range(size_input):
            self.weights.append(random.uniform(-1, 1))
        
        self.bias = random.uniform(-1, 1)
    
    def activation(self, x):
        print(x)
        s =  1.0 / ( 1 + math.pow(math.e, x) )
        return s
    
    # Makes weighted sum of the inputs
    def sum(self, x):
        if len(x) != len(self.weights):
            raise Exception(
                "Should have {} entries, got {}.".format(
                    len(self.weights) - 1, len(x) - 1
                )
            )
        
        s = 0
        for i in range(len(x)):
            s += x[i] * self.weights[i]
        s += self.bias
        return 1 if s > 1 else s
    
    # Predict the class for the inputs
    def predict(self, x):
        y = list(x) + [1]
        s = self.sum(y)
        a = self.activation(s)
        
        return a

    def reajust_weights(self, wc, bc):
        for i in range(len(self.weights)):
            self.weights[i] -= wc[i]
        self.bias -= bc

    def calculate_correction(self, alpha, delta, activation_values):
        r = []
        for i in range(len(self.weights)):
            r.append(alpha * delta * activation_values[i])
        b = alpha * delta
        return (r, b)

    def __str__(self):
        return str(self.size_input)

    def __repr__(self):
        return self.__str__()


class MLP:
    def __init__(self, size_input, size_hidden, size_output, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs

        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output

        self.hidden_layer = self.initialize_hidden_layer() 
        self.output_layer = self.initialize_output_layer()
        self.perceptron = self.hidden_layer + self.output_layer



    def initialize_hidden_layer(self):
        hidden_layer = []
        for i in range(self.size_hidden):
            hidden_layer.append(Perceptron(self.size_input))

        return hidden_layer

    def initialize_output_layer(self):
        output_layer = []
        for i in range(self.size_output):
            output_layer.append(Perceptron(self.size_hidden))

        return output_layer

    def train(self, data_set, result_set):
        for e in range(self.epochs):
            for id_set, train_set in enumerate(data_set):
                
                sum_list_hl = []
                activations_hl = []
                for perceptron in self.hidden_layer:
                    sum_list_hl.append(perceptron.sum(train_set))
                    activations_hl.append(perceptron.activation(sum_list_hl[-1]))
                
                sum_list_ol = []
                activations_ol = []
                for perceptron in self.output_layer:
                    sum_list_ol.append(perceptron.sum(activations_hl))
                    activations_ol.append(perceptron.activation(sum_list_ol[-1]))
                
                target = [0] * self.size_output
                # target = [result_set[id_set]]
                # print(result_set[id_set])
                target[result_set[id_set]] = 1

                
                deltas_output = []
                weights_correction_ol = []
                bias_correction_ol = []
                delta_in = [0] * self.size_hidden
                for p_id, perceptron in enumerate(self.output_layer):
                    deriv = activations_ol[p_id] * (1-activations_ol[p_id])
                    delta = (target[p_id] - activations_ol[p_id]) * deriv
                    deltas_output.append(delta)
                    
                    wc, bc = perceptron.calculate_correction(self.alpha, delta, activations_hl)

                    weights_correction_ol.append(wc)
                    bias_correction_ol.append(bc)

                    for w_id, w in enumerate(perceptron.weights):
                        delta_in[w_id] += w * delta
                
                deltas_hidden = []
                weights_correction_hl = []
                bias_correction_hl = []
                print("DEL TAA ",delta_in)
                print(train_set)
                for p_id, perceptron in enumerate(self.hidden_layer):
                    deriv = sum_list_hl[p_id] * (1-sum_list_hl[p_id])

                    delta = delta_in[p_id] * deriv
                    
                    deltas_hidden.append(delta)
                    
                    wc, bc = perceptron.calculate_correction(self.alpha, delta, train_set)

                    weights_correction_hl.append(wc)
                    bias_correction_hl.append(bc)
                
                for p_id, perceptron in enumerate(self.output_layer):
                    perceptron.reajust_weights(weights_correction_ol[p_id], bias_correction_ol[p_id])
                
                for p_id, perceptron in enumerate(self.hidden_layer):
                    perceptron.reajust_weights(weights_correction_hl[p_id], bias_correction_hl[p_id])
                

                print("))))000000000000000000000000000000000000")
    def run(self, data_set):
        results = []
        for id_set, test_set in enumerate(data_set):
            sum_list_hl = []
            activations_hl = []
            for perceptron in self.hidden_layer:
                sum_list_hl.append(perceptron.sum(test_set))
                activations_hl.append(perceptron.activation(sum_list_hl[-1]))
            
            sum_list_ol = []
            activations_ol = []
            for perceptron in self.output_layer:
                sum_list_ol.append(perceptron.sum(activations_hl))
                activations_ol.append(perceptron.activation(sum_list_ol[-1]))
            results.append(activations_ol)
        print(results)

mlp = MLP(2, 4, 2, 0.5, 1000)

mlp.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])

mlp.run([[1, 0], [1,1], [0,1], [0,0]])
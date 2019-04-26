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
        return s
    
    # Predict the class for the inputs
    def predict(self, x):
        y = list(x) + [1]
        s = self.sum(y)
        a = self.activation(s)
        
        return a


    def reajust_weights(self, alpha, delta, activation_values):
        for i in range(len(self.weights)):
            self.weights[i] -= alpha * delta * activation_values[i]
        self.bias -= alpha * delta

    def __str__(self):
        return str(self.size_input)

    def __repr__(self):
        return self.__str__()

class MLP:
    
    def __init__(self, size_input, sizes_hidden, size_output, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        
        self.size_input = size_input
        self.sizes_hidden = sizes_hidden
        self.size_output = size_output
        
        ########
        self.sizes_layers = []
        self.sizes_layers.append(self.size_input)
        for size in self.sizes_hidden:
            self.sizes_layers.append(size)
        self.sizes_layers.append(size_output)
        #######
        
        self.input_layer = self.initialize_input_layer(self.size_input)
        self.hiddens_layers = self.initialize_hidden_layers(self.sizes_hidden)
        self.output_layer = self.initialize_output_layer(self.size_output)
        
        self.layers = [self.input_layer]
        for h in self.hiddens_layers:
            self.layers += [h]
        self.layers += [self.output_layer]

        #######
        self.perceptrons = []
        self.perceptrons += self.input_layer
        for layer in self.hiddens_layers:
            self.perceptrons += layer
        self.perceptrons += self.output_layer
        #######

        self.weights = []
        self.bias = []


    def initialize_input_layer(self, size_input):
        input_layer = []
        for i in range(size_input):
            input_layer.append(None)

        return input_layer


    def initialize_hidden_layers(self, sizes_hidden):
        hidden_layers = []
        k = 0
        for i in range(len(sizes_hidden)):
            hidden_layers.append([])
            for j in range(sizes_hidden[i]):
                hidden_layers[i].append(Perceptron(self.sizes_layers[i]))
            k += 1

        return hidden_layers

    def initialize_output_layer(self, size_output):
        output_layer = []
        for i in range(size_output):
            output_layer.append(Perceptron(self.sizes_layers[-2]))

        return output_layer

    def predict(self, x):
        pass


    def calculate_output_correction_term(self, target, all_activation_values):
        delta_output = []

        for i in range(self.size_output):
            derivate = all_activation_values[-1][i] * (1 - all_activation_values[-1][i])
            d = (target[i] - all_activation_values[-1][i]) * derivate
            delta_output.append(d)

        return delta_output

    def calculate_correction_term(self, layer_id, layer, next_layer_delta , all_activation_values):
        deltas = []
        next_layer = self.layers[layer_id + 1]

        for p_idx in enumerate(layer):
            delta_p = 0
            for nld_id, nl_delta in enumerate(next_layer_delta):
                delta_p += next_layer[nld_id].weights[p_idx] * nl_delta

            derivate = all_activation_values[layer_id - 1][p_idx] * (1 - all_activation_values[layer_id - 1][p_idx])
            deltas.append(delta_p * derivate)

        return deltas
        

    def reajust_weights(self, all_activation_values, delta_output):
        for p_idx, perceptron in enumerate(self.output_layer):
            perceptron.reajust_weights(self.alpha, delta_output[p_idx], all_activation_values[-2])

    
    def train(self, data_set, result_set):

        for e in range(self.epochs):
            for id_set, train_set in enumerate(data_set):
                # Foward Propagation
                input_list = copy.deepcopy(train_set)
                all_activation_values = []
                for hidden_layer in self.hiddens_layers:
                    activation_values = []
                    for perceptron in hidden_layer:
                        sum_value = perceptron.sum(input_list)
                        activation_value = perceptron.activation(sum_value)
                        activation_values.append(activation_value)

                    all_activation_values.append(activation_values)
                    input_list = copy.deepcopy(activation_values)

                activation_values = []
                for perceptron in self.output_layer:
                    sum_value = perceptron.sum(input_list)
                    activation_value = perceptron.activation(sum_value)
                    activation_values.append(activation_value)

                all_activation_values.append(activation_values)
                

                target = [0] * self.size_output
                target[result_set[id_set]] = 1
                delta_output = self.calculate_output_correction_term(target, all_activation_values)
                
                delta_before_layer = self.calculate_correction_term(len(self.layers) - 2, self.layers[-2], delta_output, all_activation_values)
                for i in range(len(self.hiddens_layers) - 1, -1, -1):
                    self.reajust_weights(all_activation_values, delta_output)
                    delta_output = delta_before_layer
                    delta_before_layer = self.calculate_correction_term(i, self.layers[i+1], delta_output, all_activation_values)
                


    def test(self, test_set):
        input_list = copy.deepcopy(test_set)
        all_activation_values = []
        for layer in self.layers[self.size_input-1:]:
            activation_values = []
            for perceptron in layer:
                sum_value = perceptron.sum(input_list)
                activation_value = perceptron.activation(sum_value)

                activation_values.append(activation_value)

            all_activation_values.append(activation_values)
            input_list = copy.deepcopy(activation_values)
        
        return input_list


mlp = MLP(2, [4], 1, 1, 1000)

mlp.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
probabilities = mlp.test([0, 1])
print(probabilities)

max_pos = 0
for i in range(1, len(probabilities)):
    if(probabilities[max_pos] < probabilities[i]):
        max_pos = i

print(max_pos)

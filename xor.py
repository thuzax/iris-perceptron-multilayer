#!/usr/bin/python

import numpy as np
import math

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivada(x):
  return x*(1.0 - x)


class MLP:
  def __init__(self, inputs):
    self.inputs = inputs

    self.tamanho = len(self.inputs)
    self.tamanho_input = len(self.inputs[0])

    self.pesos_i = np.random.random((self.tamanho_input, self.tamanho))
    self.pesos_h = np.random.random((self.tamanho, 1))

  # Predição
  def run(self, input_run):
    l1_result = sigmoid(np.dot(input_run, self.pesos_i))
    l2_result = sigmoid(np.dot(l1_result, self.pesos_h))
    
    return l2_result


  def train(self, inputs,outputs, it):
    for i in range(it):
      # Foward propagation para todos os testes
      l_inputs = inputs
      l_hidden = sigmoid(np.dot(l_inputs, self.pesos_i))
      l_outputs = sigmoid(np.dot(l_hidden, self.pesos_h))

      # Backward propagation para todos os testes
      l_outputs_err = outputs - l_outputs
      l_outputs_delta  =  np.multiply(l_outputs_err, sigmoid_derivada(l_outputs))

      l_hidden_err = np.dot(l_outputs_delta, self.pesos_h.T)
      l_hidden_delta = np.multiply(l_hidden_err, sigmoid_derivada(l_hidden))

      # Correção dos vaores
      self.pesos_h += np.dot(l_hidden.T, l_outputs_delta)
      self.pesos_i += np.dot(l_inputs.T, l_hidden_delta)

inputs_treino = np.array([[0,0], [0,1], [1,0], [1,1] ])
outputs = np.array([ [0], [1],[1],[0] ])

n = MLP(inputs_treino)
print(n.run(inputs_treino))
n.train(inputs_treino, outputs, 10000)

inputs_testes = np.array([[0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1] ])
outputs_esperados = np.array([0, 1, 1, 0, 0, 1, 1, 0])
results = n.run(inputs_testes)

acc = 0.0
for idx, result in enumerate(results):
  if result < 0.5:
    print(inputs_testes[idx][0], " XOR ", inputs_testes[idx][1], " = ", math.floor(result))
    if math.floor(result) == int(outputs_esperados[idx]):
      acc += 1
  else:
    print(inputs_testes[idx][0], " XOR ", inputs_testes[idx][1], " = ", math.ceil(result))
    if math.ceil(result) == int(outputs_esperados[idx]):
      acc += 1

print(acc / len(inputs_testes) * 100, "%")

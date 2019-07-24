import numpy as np


class NeuralNetwork(object):

    def __init__(self, training_inputs=None, training_outputs=None):
        np.random.seed(1)
        self.synaptic_weights = None

    # sigmoid function to normalize inputs
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + pow(np.e, -x))


    # sigmoid derivatives to adjust synaptic weights
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def initSynapticWeights(self, training_inputs):
        rows, columns = training_inputs.shape
        self.synaptic_weights = 2 * np.random.random((columns, 1)) - 1

    def train(self, training_inputs, training_outputs, training_iterations):
        if self.synaptic_weights is None:
            self.initSynapticWeights(training_inputs)

        for iterention in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return outputs


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    # input dataset
    training_inputs = np.array([
        [0, 0, 0, 0, 1],  # 0
        [0, 0, 0, 1, 1],  # 1
        [0, 0, 1, 0, 0],  # 0
        [0, 1, 0, 1, 0],  # 1
        [1, 0, 0, 0, 0],  # 0
        [0, 0, 0, 1, 1],  # 1
        [0, 0, 1, 1, 1],  # 1
        [0, 1, 1, 1, 1],  # 1
    ])

    # output dataset
    training_outputs = np.array([
        [0, 1, 0, 1, 0, 1, 1, 1]
    ]).T

    neural_network.train(training_inputs, training_outputs, 100000)

    print neural_network.think(np.array([1, 1, 1, 1, 0]))
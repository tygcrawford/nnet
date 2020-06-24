import os
import numpy as np
import matplotlib.pyplot as plt

curr_path = os.path.dirname(os.path.realpath(__file__))
train_path = f'{curr_path}/data/train'
i_path = f'{train_path}/train-images'
l_path = f'{train_path}/train-labels'


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.a = np.zeros((neurons, 1))


class InputLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)

    def calculate(self, data):
        self.a = data
        return self.a


class DenseLayer(Layer):
    def __init__(self, neurons, input_neurons, activation, weights=None, biases=None):
        super().__init__(neurons)

        self.input_neurons = input_neurons

        self.z = np.zeros((neurons, 1))

        # TODO: Weight initialization techniques
        # he_init = np.sqrt(2 / self.input_neurons)
        if weights is None:
            self.weights = np.random.randn(self.neurons, self.input_neurons)
        else:
            self.weights = weights
        if biases is None:
            self.biases = np.zeros((neurons, 1))
        else:
            self.biases = biases

        self.activation = activation

    def calculate(self, data):
        self.z = np.add(np.dot(self.weights, data), self.biases)
        self.a = self.activation(self.z)
        return self.a



class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward_prop(self, data):
        output = data
        for layer in self.layers:
            output = layer.calculate(output)
        return output

    def cost(self, expected, derivative=False):
        if derivative:
            return 2 * (self.layers[-1].a - expected)

        return np.sum(np.square(self.layers[-1].a - expected))

    def back_prop(self, expected):
        error = [None] * len(self.layers)
        error[-1] = np.multiply(self.cost(expected, derivative=True), self.sigmoid(self.layers[-1].z, derivative=True))

        for l in range(len(self.layers)-2, 0, -1):
            error[l] = np.multiply(np.dot(np.transpose(self.layers[l + 1].weights), error[l + 1]),
                                   self.sigmoid(self.layers[l].z, derivative=True))

        bias_gradient = error
        weight_gradient = [None] * len(self.layers)
        for l in range(len(self.layers)-1, 0, -1):
            weight_gradient[l] = np.dot(error[l], np.transpose(self.layers[l-1].a)) # np.zeros((error[l], self.layers[l-1].shape[0]))

        return weight_gradient[1:], bias_gradient[1:]

    def batch_train(self, data, expected):
        wg, bg = None, None
        for i in range(len(data)):
            self.forward_prop(data[i])
            if wg is None and bg is None:
                wg, bg = self.back_prop(expected[i])
            else:
                w, b = self.back_prop(expected[i])
                for j in range(len(wg)):
                    wg[j] += w[j]
                for j in range(len(bg)):
                    bg[j] += b[j]

        for i in range(len(wg)):
            wg[i] /= len(data)
        for i in range(len(bg)):
            bg[i] /= len(data)

        l_rate = 1
        for i in range(1, len(self.layers)):
            self.layers[i].weights -= wg[i-1] * l_rate
            self.layers[i].biases += bg[i-1] * l_rate


    # TODO: Test save and load
    def save(self, path):
        with open(path, "wb") as f:
            l = [l.neurons for l in self.layers]
            np.save(f, np.array(l))
            for i in range(1,len(self.layers)):
                np.save(f, self.layers[i].weights)
            for i in range(1,len(self.layers)):
                np.save(f, self.layers[i].biases)

    @staticmethod
    def load(path, activation):
        with open(path, "rb") as f:
            l = np.load(f)

            weights = []
            for weight_layer in range(len(l) - 1):
                weights.append(np.load(f))

            biases = []
            for bias_layer in range(len(l) - 1):
                biases.append(np.load(f))

        layers = [InputLayer(l[0])]
        for i in range(1, len(l)):
            layers.append(DenseLayer(i, i-1, activation, weights=weights[i-1], biases=biases[i-1]))

        return NeuralNetwork(layers)

    @staticmethod
    def mean_squared_error(output, expected):
        return np.sum(np.square(output - expected))

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return np.exp(-x) / np.square(1 + np.exp(-x))

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def display_output(output, labels=None):
        plt.imshow(output, cmap="gist_gray")
        plt.xticks([])
        plt.yticks(np.arange(len(output)), labels)
        plt.show()


class Data:
    def __init__(self, image_path, label_path):
        self.image_f = open(image_path, "rb")
        self.label_f = open(label_path, "rb")

        self.image_f.seek(4)
        self.label_f.seek(4)

        self.i_num = self.byte_as_int(self.image_f, 4)
        self.l_num = self.byte_as_int(self.label_f, 4)

        self.i_width = self.byte_as_int(self.image_f, 4)
        self.i_height = self.byte_as_int(self.image_f, 4)
        self.i_area = self.i_width * self.i_height

    def get_next(self):
        image = []
        for pixel in range(self.i_area):
            image.append(self.byte_as_int(self.image_f, 1)/255)

        label = self.byte_as_int(self.label_f, 1)

        expected = np.zeros((10, 1))
        expected[label][0] = 1

        image = np.reshape(image, (len(image), 1))

        return image, expected

    def get_index(self, index):
        self.image_f.seek(16 + index * 784)
        self.label_f.seek(8 + index)

        return self.get_next()

    def close(self):
        self.image_f.close()
        self.label_f.close()

    @staticmethod
    def byte_as_int(file, length):
        return int.from_bytes(file.read(length), byteorder='big')

    @staticmethod
    def display_image(img):
        img = np.reshape(img, (28, 28))
        plt.imshow(img, cmap='hot', interpolation='nearest')
        plt.show()


d = Data(i_path, l_path)

NN = NeuralNetwork([
    InputLayer(784),
    DenseLayer(16, 784, NeuralNetwork.sigmoid),
    DenseLayer(16, 16, NeuralNetwork.sigmoid),
    DenseLayer(10, 16, NeuralNetwork.sigmoid)
])

data = [d.get_next() for i in range(60000)]

img = [i[0] for i in data]
lbl = [i[1] for i in data]

NN.batch_train(img, lbl)

NN.save("save.npy")


NN = NeuralNetwork.load("save.npy", NeuralNetwork.sigmoid)

training_tests = [d.get_index(i)[0] for i in range(500, 510)]

afs = []

for test in training_tests:
    afs.append(NN.forward_prop(test))

for i in range(len(training_tests)):
    Data.display_image(training_tests[i])
    NeuralNetwork.display_output(afs[i])

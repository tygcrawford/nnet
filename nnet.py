import os
import numpy as np
import matplotlib.pyplot as plt

curr_path = os.path.dirname(__file__)
train_path = f'{curr_path}data/train'
i_path = f'{train_path}/train-images'
l_path = f'{train_path}/train-labels'


class NNet:
    def __init__(self, layers, activation, weights=None, biases=None):
        self.layers = np.array(layers)
        self.activation = activation
        self.weights = [] if weights is None else weights
        self.biases = [] if biases is None else biases

    def randomize(self):
        for layer in range(1, len(self.layers)):
            he_init = np.sqrt(2 / self.layers[layer - 1])
            self.weights.append(np.random.randn(self.layers[layer], self.layers[layer - 1]) * he_init)

        for layer in range(1, len(self.layers)):
            self.biases.append(np.zeros((self.layers[layer], 1)))

    def run(self, data):
        output = data
        for layer in range(len(self.layers) - 1):
            output = self.activation(np.add(np.dot(self.weights[layer], output), self.biases[layer]))
        return output

    def save(self, path):
        with open(path, "wb") as f:
            np.save(f, self.layers)
            for weight_layer in self.weights:
                np.save(f, weight_layer)
            for bias_layer in self.biases:
                np.save(f, bias_layer)

    @staticmethod
    def load(path, squishify):
        with open(path, "rb") as f:
            layers = np.load(f)

            weights = []
            for weight_layer in range(len(layers) - 1):
                weights.append(np.load(f))

            biases = []
            for bias_layer in range(len(layers) - 1):
                biases.append(np.load(f))

        return NNet(layers, squishify, weights=weights, biases=biases)

    @staticmethod
    def sigmoid(x):
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

        image = np.reshape(image, (len(image), 1))

        return image, label

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
i, label = d.get_index(18)
d.close()

l = [784, 16, 16, 10]

nnet_a = NNet(l, NNet.sigmoid)
nnet_a.randomize()
nnet_a.save("save.npy")

nnet_b = NNet.load("save.npy", NNet.sigmoid)

a_out = nnet_a.run(i)
b_out = nnet_b.run(i)

NNet.display_output(a_out)

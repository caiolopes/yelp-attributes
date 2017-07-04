import pandas as pd
import numpy as np

class MLP(object):
    def __init__(self, input_size, hidden_size, output_size, output_func='sigmoid'):
        self.hidden_size = hidden_size
        self.hidden = np.random.uniform(-0.5, 0.5,
                                        (hidden_size, input_size + 1))

        self.output = np.random.uniform(-0.5, 0.5,
                                        (output_size, hidden_size + 1))

        self.output_func = output_func


    def sigmoid(self, net):
        return 1 / (1 + np.exp(-net))


    def d_sigmoid(self, net):
        return self.sigmoid(net) * (1 - self.sigmoid(net))


    def softmax(self, w):
        e = np.exp(w - np.amax(w))
        dist = e / np.sum(e)
        return dist


    def tanh(self, net):
        return np.tanh(net)


    def d_tanh(self, net):
        return 1 - np.multiply(net, net)


    def forward(self, x):
        self.f_h_net = np.zeros(self.hidden.shape[0])
        self.df_h_dnet = np.zeros(self.hidden.shape[0])
        for j in range(0, len(self.hidden)):
            net_h = np.matmul(np.concatenate((x, [1])), self.hidden[j, 0:])
            self.f_h_net[j] = self.sigmoid(net_h)
            self.df_h_dnet[j] = self.d_sigmoid(net_h)

        self.f_o_net = np.zeros(self.output.shape[0])
        self.df_o_dnet = np.zeros(self.output.shape[0])
        for j in range(0, len(self.output)):
            net_o = np.matmul(np.concatenate((self.f_h_net, [1])), self.output[j, 0:])
            self.df_o_dnet[j] = self.d_sigmoid(net_o)

            if self.output_func == 'sigmoid':
                self.f_o_net[j] = self.sigmoid(net_o)
            elif self.output_func == 'softmax':
                self.f_o_net[j] = self.sigmoid(net_o)

    def backpropagation(self, X, Y, eta=0.01, rate_decay=0.0001, threshold=1e-2, num_iter=-1):
        squared_error = 2 * threshold

        print(f"Running for {self.hidden_size} hidden nodes, eta = {eta}, and threshold = {threshold}, num_iter = {num_iter}")
        i = 0
        while squared_error > threshold or i < num_iter:
            squared_error = 0

            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

            for p in range(0, len(X)):
                x_p = X[p, 0:]
                y_p = Y[p, 0:]

                self.forward(x_p)
                o_p = self.f_o_net

                delta_p = y_p - o_p

                squared_error = squared_error + sum(delta_p ** 2)

                delta_o_p = delta_p * self.df_o_dnet

                delta_h_p = self.df_h_dnet * (np.matmul(delta_o_p,
                        self.output[0:, 0:len(self.output[0]) - 1]))

                self.output = self.output + eta * np.matmul(delta_o_p[:,np.newaxis],
                        np.array(np.concatenate((self.f_h_net, [1])))[np.newaxis])

                self.hidden = self.hidden + eta * np.matmul(delta_h_p[:,np.newaxis],
                        np.array(np.concatenate((x_p, [1])))[np.newaxis])

            eta = eta * (eta / (eta + (eta * rate_decay)))
            squared_error = squared_error / len(X)
            print("Iter {}: Average squared error: {}".format(i, squared_error))
            i = i + 1
            if num_iter != -1 and num_iter == i:
                break

import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size

        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.activation = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # nullify layers gradients
        self.params()['W1'].grad = np.zeros((self.n_input, self.hidden_layer_size))
        self.params()['B1'].grad = np.zeros((1, self.hidden_layer_size))
        self.params()['W2'].grad = np.zeros((self.hidden_layer_size, self.n_output))
        self.params()['B2'].grad = np.zeros((1, self.n_output))
        # forward layer 1
        layer_forward1 = self.layer1.forward(X)
        # # forward activation funtcion
        activation_forward = self.activation.forward(layer_forward1)
        # # forward layer 2
        layer_forward2 = self.layer2.forward(activation_forward)
        # calculate loss
        loss, grad = softmax_with_cross_entropy(layer_forward2, y)
        # # backward layer 2
        layer_backward2 = self.layer2.backward(grad)
        # # backward activation funtcion
        activation_backward = self.activation.backward(layer_backward2)
        # backward layer 1
        layer_backward1 = self.layer1.backward(activation_backward)
        # l2 regularization on all params
        W1_reg_loss, W1_reg_grad = l2_regularization(self.params()['W1'].value, self.reg)
        B1_reg_loss, B1_reg_grad = l2_regularization(self.params()['B1'].value, self.reg)
        W2_reg_loss, W2_reg_grad = l2_regularization(self.params()['W2'].value, self.reg)
        B2_reg_loss, B2_reg_grad = l2_regularization(self.params()['B2'].value, self.reg)
        # update gradients
        self.params()['W1'].grad += W1_reg_grad
        self.params()['B1'].grad += B1_reg_grad
        self.params()['W2'].grad += W2_reg_grad
        self.params()['B2'].grad += B2_reg_grad
        # update loss
        loss += (W1_reg_loss + W2_reg_loss + B1_reg_loss + B2_reg_loss)
        # update layers weights
        # self.params()['W1'].value -= self.params()['W1'].grad
        # self.params()['B1'].value -= self.params()['B1'].grad
        # self.params()['W2'].value -= self.params()['W2'].grad
        # self.params()['B2'].value -= self.params()['B2'].grad
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'W1': self.layer1.params()['W'], 'B1': self.layer1.params()['B'],
                  'W2': self.layer2.params()['W'], 'B2': self.layer2.params()['B']}
        return result

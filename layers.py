import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength*np.sum(W**2)
    grad = 2*reg_strength*W
    return loss, grad


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    norm_predictions = predictions - np.amax(predictions, axis=1)[:, None]
    exp_array = np.exp(norm_predictions)
    return exp_array/np.sum(exp_array, axis=1)[:, None]


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
      batch_size is a number of batches in probs array
    Returns:
      loss: single value
    """
    mask_array = np.zeros(probs.shape, dtype=int)
    ce_loss = np.zeros(probs.shape[0], dtype=np.float)
    for i in range(probs.shape[0]):
        mask_array[i, target_index[i]] = 1
    for i in range(probs.shape[0]):
        ce_loss[i] = -np.sum(mask_array[i] * np.log(probs[i]))
    return np.average(ce_loss)


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    target_array = np.zeros(probs.shape, dtype=int)
    for i in range(target_array.shape[0]):
        target_array[i, target_index[i]] = 1
    dprediction = (probs - target_array)/probs.shape[0]
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = np.multiply(d_out, np.int64(self.X > 0))
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        W = self.params()['W'].value
        B = self.params()['B'].value
        predictions = np.dot(X, W) + B
        return predictions

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        # n = d_out.shape[0]
        X = self.X
        W = self.params()['W'].value
        B = self.params()['B'].value
        d_result = np.dot(d_out, W.T)
        dW = np.dot(X.T, d_out)
        self.params()['W'].grad = dW
        dB = np.sum(d_out, axis=0, keepdims=True)
        print(f'd_out\n{d_out}\n')
        print(f'dB\n{dB}\n')
        print(f'np.sum(d_out, axis=0)\n{np.sum(d_out, axis=0)}\n')
        self.params()['B'].grad = dB
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}

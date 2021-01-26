import numpy as np


class LinearSVM(object):
    """ A subclass that uses the Multiclass SVM loss function """
    def __init__(self):
        self.W = None

    def loss_vectorized(self, X, y, reg):
        """
        Structured SVM loss function, naive implementation (with loops).
        Inputs:
        - X: A numpy array of shape (num_train, D) contain the training data
          consisting of num_train samples each of dimension D
        - y: A numpy array of shape (num_train,) contain the training labels,
          where y[i] is the label of X[i]
        - reg: (float) regularization strength
        Outputs:
        - loss: the loss value between predict value and ground truth
        - dW: gradient of W
        """

        # Initialize loss and dW
        loss = 0.0
        dW = np.zeros(self.W.shape)
        num_train = X.shape[0]
        scores = np.dot(X, self.W)
        correct_score = scores[range(num_train), list(y)].reshape(-1, 1)
        # delta = 1
        margin = np.maximum(0, scores - correct_score + 1)
        # 分对的损失为0
        margin[range(num_train), list(y)] = 0
        # reg就是权重lamda
        loss = np.sum(margin) / num_train + 0.5 * reg * np.sum(self.W * self.W)
        num_classes = self.W.shape[1]
        mask = np.zeros((num_train, num_classes))
        mask[margin > 0] = 1
        mask[range(num_train), list(y)] = 0
        mask[range(num_train), list(y)] = -np.sum(mask, axis=1)
        dW = np.dot(X.T, mask)
        dW = dW / num_train + reg * self.W
        return loss, dW

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, print_flag=False):
        """
        Train linear SVM classifier using SGD
        Inputs:
        - X: A numpy array of shape (num_train, D) contain the training data
          consisting of num_train samples each of dimension D
        - y: A numpy array of shape (num_train,) contain the training labels,
          where y[i] is the label of X[i], y[i] = c, 0 <= c <= C
        - learning rate: (float) learning rate for optimization
        - reg: (float) regularization strength
        - num_iters: (integer) numbers of steps to take when optimization
        - batch_size: (integer) number of training examples to use at each step
        - print_flag: (boolean) If true, print the progress during optimization
        Outputs:
        - loss_history: A list containing the loss at each training iteration
        """
        loss_history = []
        num_train = X.shape[0]
        dim = X.shape[1]
        num_classes = np.max(y) + 1

        # Initialize W
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # iteration and optimization
        for t in range(num_iters):
            idx_batch = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            loss, dW = self.loss_vectorized(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W += -learning_rate * dW
            if print_flag and t % 100 == 0:
                print('iteration %d / %d: loss %f' % (t, num_iters, loss))
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of linear SVM to predict data labels
        Inputs:
        - X: A numpy array of shape (num_train, D) contain the training data
        Outputs:
        - y_pred: A numpy array, predicted labels for the data in X
        """
        y_pred = np.zeros(X.shape[0])
        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_class = W.shape[1]
    N = X.shape[0]
    f = X.dot(W) #N,C
    f -= f.max()
    expf = np.exp(f)
    p = expf / expf.sum(axis=1, keepdims=True)
    loss = -np.log(p[range(N), y.astype(int)]).sum() / N
    loss += 0.5 * reg * np.sum(W * W)

    df = p
    df[range(N), y.astype(int)] -= 1.0

    dpen = reg * W
    dW = X.T.dot(df) / N + dpen


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W) # NxD * DxC = NxC
    scores -= np.max(scores,axis=1,keepdims=True)
    probabilities = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
    correct_class_probabilities = probabilities[range(num_train),y.astype(int)]

    loss = np.sum(-np.log(correct_class_probabilities)) / num_train
    loss += 0.5 * reg * np.sum(W*W) 

    probabilities[range(num_train),y.astype(int)] -= 1
    dW = X.T.dot(probabilities) / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

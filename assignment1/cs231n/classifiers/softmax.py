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

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W)
      true_label_score = scores[y[i]]
      loss += -np.log(np.exp(true_label_score)/np.sum([np.exp(scores[j])for j in range(num_classes)]))
      for j in range(num_classes):
        if j == y[i]:
          dW[:,j] += -1*(X[i]*(1-np.exp(scores[j])/np.sum([np.exp(scores[m]) for m in range(num_classes)])))
        else:
          dW[:,j] += X[i]*np.exp(scores[j])/np.sum([np.exp(scores[m]) for m in range(num_classes)])
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += reg*2*W

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
    num_classes = W.shape[1]
    scores_mat = X.dot(W)
    true_label_score_per_sample = np.choose(y, np.transpose(scores_mat))
    exp_scores_mat = np.exp(scores_mat)
    exp_true_label_score_per_sample = np.exp(true_label_score_per_sample)
    loss = np.sum(-np.log(exp_true_label_score_per_sample/np.sum(exp_scores_mat, axis=1)))
    exp_ratio = exp_scores_mat/np.expand_dims(np.sum(exp_scores_mat, axis=1), axis=1)
    true_class_mask = np.eye(num_classes)[y]
    dW = np.matmul(np.transpose(X),(exp_ratio-true_class_mask))
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

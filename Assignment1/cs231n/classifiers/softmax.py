import numpy as np
from random import shuffle

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
  num_classes = W.shape[1] # D
  num_train = X.shape[0] # N
  losses = np.zeros(num_train)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    dotOutput=X[i].dot(W)
    dotOutput -= np.max(dotOutput) # 연산 안정성을 위해 shift 
    correct_predict = dotOutput[y[i]]
    exp_dot = np.exp(dotOutput) # 1*D s=Wx
    exp_sum = np.sum(exp_dot) # ∑e^s_k
    exp_cor = np.exp(correct_predict) # e^s_y[i]
    estimation = exp_cor/ exp_sum # exp and normalize # e^s_y[i]/∑e^s_k
    losses[i]=-np.log(estimation) # loss = ∑-log(est_i)/N

  #gradient
    dW[:,y[i]] += (-1) * (exp_sum - exp_cor)/exp_sum * X[i] #only for label column
    for j in range(num_classes):
      if(y[i]!=j):
        dW[:, j] += exp_dot[j] / exp_sum * X[i]
  # dW[:,y[i]] += (∑ - e^s_yi )/∑ * x[i]
  # dW[:, j ] += e^s_xj / ∑ * x[i] (*y[i]!=j)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss = losses.sum() /num_train
  dW /=num_train

  loss +=reg* np.sum(W*W)
  dW += 2*reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0] # N

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  dotOutput=X.dot(W) # N*C
  dotOutput -= np.max(dotOutput) # 연산 안정성을 위해 shift 
  exp_Out = np.exp(dotOutput)
  exp_sum = np.sum(exp_Out,axis=1)
  exp_cor = exp_Out[range(num_train),y]

  estimation = (exp_cor/exp_sum)
  
  loss = -(np.sum(np.log(estimation))/num_train) + reg*np.sum(W*W) # data loss + reg loss
  
  #gradient 
  s = exp_Out / exp_sum.reshape(num_train, 1)
  s[range(num_train), y] = - (exp_sum - exp_cor) / exp_sum
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


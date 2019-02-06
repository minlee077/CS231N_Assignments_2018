import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] # label socre
    for j in range(num_classes):
      if j == y[i]: # 올바르게 분류하지 않은 클래스들에 대해서만 loss를 누적한다.
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # score와 correct class score와의 차이가 delta(1)보다 클때
        loss += margin
        dW[:, j] += X[i] # 올바르지 않은 클래스에 대해서는 기존에 X[i]만큼 gradient를 증가
        dW[:, y[i]] -= X[i] # 올바르게 분류한 경우에 대해서는 X[i]만큼 gradient감소  loss = ∑max(W[:,j]x[j]- W[:,y[i]]x[i] + delta)이므로, dL/dw는 이와 같이 된다.

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  N = X.shape[0]


  delta = 1.0
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores =np.array(X.dot(W))
  correct_scores = scores[np.arange(N), y]  # (N, ) label to score
  margins = np.maximum(0,scores-correct_scores.reshape(N,1) + delta) # 모든 row(batch size = N)에 대해 correct score를 차감
  margins[range(N),y]=0 # 올바른 label에 대한 margin은 0으로 수정 
  loss = np.sum(margins) / N
  loss += reg*np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dS = np.zeros(scores.shape)
  dS[margins > 0] = 1 
  dS[np.arange(N), y] -= np.sum(dS, axis=1)   #  (N, 1) = (N, 1)

  
  dW = np.dot(X.T,dS)
  dW /=N
  dW +=2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

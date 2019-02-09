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
    scores = X[i].dot(W) # score = X[i]W (1*D * D*C) W의 각 column이 label template이다. (1xClasses 의 벡터)
    correct_class_score = scores[y[i]] # label socre
    for j in range(num_classes): #loss = ∑max(W[:,j]x[i]- W[:,y[i]]x[i] + delta)
      if j == y[i]: #label과 다른 클래스들에 대해서만 loss를 계산하여 누적한다. (*label과 같은 경우에는 delta만큼의 loss가 추가될 것이다.)
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # score와 correct class score와의 차이가 delta(1)보다 클때 loss += max(0,margin)
        loss += margin# loss += (W[:,j]x[i] - W[:,y[i]]x[i] + delta)
        dW[:, j] += X[i] # j템플렛에 상응하는 W의 열에 대해서, 대해서 X[i]만큼 gradient를 증가 (W[:,j]*x[i]) # dL/dW =[:,j]X[i] - [:,y[i]]X[i]
        dW[:, y[i]] -= X[i] #y[i] 템플렛(라벨)에 상응하는 W의 열에 대해 X[i]만큼 

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

  scores =np.array(X.dot(W)) # N*D * D*C = N*C score matrix
  correct_scores = scores[np.arange(N), y]  # (N, )label to score
  margins = np.maximum(0,scores-correct_scores.reshape(N,1) + delta) # 모든 data(batch size = N rows)에 대해 correct score(벡터 N*1 column)를 차감
  margins[range(N),y]=0 # 올바른 label에 대한 margin은 0으로 수정 
  loss = np.sum(margins) / N
  loss += reg*np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #목표 dL/dW
  # S = XW (N*C)
  # dS/dW =X (D*N)
  # L = ∑∑max(0,S(j!=y[i]) - S(correct score) +delta)
  # dW = ( D*C )
  dS = np.zeros(scores.shape) # N*C 
  dS[margins > 0] = 1 # loss에 영향을 줬던 데이터에 곱해진 템플릿 요소들을 1로 초기화
  dS[np.arange(N), y] -= np.sum(dS, axis=1)#  (특정 data, label)  correct label에 대해서 1로 초기화 했던 값을 label dS에 차감해줌 (-correct score때문)

  dW = np.dot(X.T,dS) # D*N * N*C  margin이 0보다 큰 것들이 X^T에 곱해지며, correct score가 사용된 만큼 sum up한 것이 곱해짐
  dW /= N
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
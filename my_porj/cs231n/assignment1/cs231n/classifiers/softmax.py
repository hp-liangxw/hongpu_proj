import numpy as np


def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):

        scores = X[i].dot(W)
        scores -= np.max(scores)

        loss += -np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores)))

        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += X[i].T * np.exp(scores[j]) / np.sum(np.exp(scores)) - X[i].T
            else:
                dW[:, j] += X[i].T * np.exp(scores[j]) / np.sum(np.exp(scores))

    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)

    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]

    # 求解loss
    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape(-1, 1)

    loss = - np.sum(np.log(np.exp(scores[range(num_train), y]) / np.sum(np.exp(scores), axis=1)))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)

    # 求解grad
    p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    p[range(num_train), y] -= 1

    dW = X.T.dot(p)
    dW /= num_train
    dW += reg * W

    return loss, dW
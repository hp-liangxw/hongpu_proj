import numpy as np


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            # 参考公式(1)，这里的y[i]即正确分类k
            # 而正确分类对loss是没有贡献的
            if j == y[i]:
                continue

            # 即M_j
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            # margin>0才需要计算损失
            if margin > 0:
                # 这里可以参考下公式(1)
                # 两层循环，其实是Li中的每一行分别对w_j和w_k求偏导
                # 由于Li中每一行都与w_k有关，因此dW[:, y[i]]每一次都需要更新
                dW[:, y[i]] += -X[i].T
                dW[:, j] += X[i].T
                loss += margin

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]

    # 得到所有的分数
    scores = X.dot(W)
    # 正确类别对应的分数
    correct_class_score = scores[range(num_train), y].reshape(-1, 1)
    # 计算loss
    margins = scores - correct_class_score + 1
    margins[margins < 0] = 0
    # 所有的loss
    data_loss = np.sum(margins) / num_train
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # 计算梯度
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)  # 1 by N
    margins[range(num_train), y] -= row_sum
    dW = np.dot(X.T, margins) / num_train + reg * W  # D by C

    return loss, dW


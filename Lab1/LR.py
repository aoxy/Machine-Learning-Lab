# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def main():
    # 载入数据
    x_train = np.load("./data/train_data.npy")  # shape: (70, 3)
    y_train = np.load("./data/train_target.npy")  # shape: (70, )
    x_test = np.load("./data/test_data.npy")  # shape: (30, 3)
    y_test = np.load("./data/test_target.npy")  # shape: (30, )

    # 转换形状为(n,1)
    Y_train = y_train.reshape(y_train.shape[0], 1)
    Y_test = y_test.reshape(y_test.shape[0], 1)

    # 求出W
    W, W_save = gradient_descent(x_train, Y_train, 0.001, 100000)
    print("W = ", W)

    # 训练集精度
    p_train = predict(x_train, W)
    print(u'训练集精度为 %f%%' % np.mean(np.float64(p_train == Y_train)*100))

    # 训练集精度
    p_test = predict(x_test, W)
    print(u'测试集精度为 %f%%' % np.mean(np.float64(p_test == Y_test)*100))

    # 动态展示梯度下降法优化LR模型的过程
    plt.figure(num=0, figsize=(18, 14))
    show_dynamic(x_train, y_train, W_save)

    # 作训练集结果图(包括决策边界)
    plt.figure(num=1, figsize=(18, 14))
    show_result(x_train, y_train, W, "Train")

    # 作测试集结果图(包括决策边界)
    plt.figure(num=2, figsize=(18, 14))
    show_result(x_test, y_test, W, "Test")


def gradient_descent(X, Y, alpha=0.001, max_iter=100000):
    """
    返回使用梯度下降法求得的 W ，X形状为(n,3),Y形状为(n,1)
    """
    W = np.random.randn(X.shape[1], 1)  # 随机初始化 W ,维度(3,1)
    W_save = []  # 记录迭代过程中的 W,用于动态展示迭代过程
    save_step = int(max_iter/100)  # 记下100组W
    Xt = np.transpose(X)  # Xt 维度(3,70)
    for i in range(max_iter):
        H = sigmoid(np.dot(X, W))  # H 维度(70,1)
        dW = np.dot(Xt, H-Y)  # dw 维度(3,1)
        W = W-alpha * dW  # 更新 W
        if i % save_step == 0:
            W_save.append([W.copy(), i])
    return W, W_save


def sigmoid(z):
    h = np.zeros((len(z), 1))  # 初始化，与z的长度一置
    h = 1.0/(1.0+np.exp(-z))
    return h


def predict(X, W):
    m = X.shape[0]  # m 组数据
    p = np.zeros((m, 1))
    p = sigmoid(np.dot(X, W))    # 预测的结果z，是个概率值
    # 概率大于0.5预测为1，否则预测为0
    for i in range(m):
        p[i] = 1 if p[i] > 0.5 else 0
    return p


def show_result(X, y, W, title):
    w0 = W[0][0]
    w1 = W[1][0]
    w2 = W[2][0]
    x1_low = min(X[:, 1])
    x1_high = max(X[:, 1])
    plotx1 = np.arange(x1_low, x1_high, 0.01)
    plotx2 = -w0/w2-w1/w2*plotx1
    plt.plot(plotx1, plotx2, c='r', label='decision boundary')
    plt.title(title)
    plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], s=90, label='label = 0')
    plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1],
                marker='s', s=90, label='label = 1')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid()
    plt.legend()
    plt.show()


def show_dynamic(X, y, W_save):
    x1_low = min(X[:, 1])
    x1_high = max(X[:, 1])
    for w in W_save:
        plt.clf()
        w0 = w[0][0][0]
        w1 = w[0][1][0]
        w2 = w[0][2][0]
        plotx1 = np.arange(x1_low, x1_high, 0.01)
        plotx2 = -w0/w2-w1/w2*plotx1
        plt.plot(plotx1, plotx2, c='r', label='decision boundary')
        plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], s=90, label='label = 0')
        plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1],
                    marker='s', s=90, label='label = 1')
        plt.grid()
        plt.legend()
        plt.title('iter:%s' % np.str(w[1]))
        plt.pause(0.001)
    plt.show()


if __name__ == "__main__":
    main()

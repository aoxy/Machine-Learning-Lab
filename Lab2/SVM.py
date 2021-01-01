import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        self._w = self._b = self.Wb_save = None

    def fit(self, x, y, c=1, lr=0.01, batch_size=32, epoch=10000):
        n = len(x)
        batch_size = min(batch_size, n)
        self._w = np.zeros(x.shape[1])  # 用0初始化w，b
        self._b = 0
        save_step = int(epoch/100)  # 记下100组Wb
        self.Wb_save = []
        for i in range(epoch):
            if i % save_step == 0:
                self.Wb_save.append([self._w.copy(), self._b, i])
            self._w *= 1 - lr  # w的模的平方要尽量小
            # 随机选取 batch_size 个样本
            batch = np.random.choice(n, batch_size)
            x_batch = x[batch]
            y_batch = y[batch]
            err = 1 - y_batch * self.predict(x_batch, True)
            if np.max(err) <= 0:  # 最小化的函数第二项不能再优化
                continue
            mask = err > 0  # 分类错误的样本
            delta_v = lr * c * y_batch[mask]
            delta = delta_v.reshape(delta_v.shape[0], 1)
            self._w += np.mean(delta * x_batch[mask], axis=0)
            self._b += np.mean(delta)
        return self._w, self._b

    def predict(self, x, raw=False):
        y_pred = np.dot(x, self._w) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred)

    def show_result(self, X, y, title):
        W = self._w
        b = self._b
        w1, w2 = W[0], W[1]
        x1_low = min(X[:, 0])
        x1_high = max(X[:, 0])
        plt.title(title)
        plotx1 = np.arange(x1_low, x1_high, 0.01)
        plotx2 = -b/w2-w1/w2*plotx1
        plt.plot(plotx1, plotx2, c='r', label='decision boundary')
        plt.scatter(X[:, 0][y == -1], X[:, 1][y == -1],
                    s=90, label='label = -1')
        plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1],
                    marker='s', s=90, label='label = 1')
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.grid()
        plt.legend()
        plt.show()

    def show_dynamic(self, X, y):
        x1_low = min(X[:, 0])
        x1_high = max(X[:, 0])
        for Wb in self.Wb_save:
            plt.clf()
            w1, w2 = Wb[0][0], Wb[0][1]
            b = Wb[1]
            plt.title('iter:%s' % np.str(Wb[2]))
            plotx1 = np.arange(x1_low, x1_high, 0.01)
            if w2 == 0:
                plotx2 = [b]*len(plotx1)
            else:
                plotx2 = -b/w2-w1/w2*plotx1
            plt.plot(plotx1, plotx2, c='r', label='decision boundary')
            plt.scatter(X[:, 0][y == -1], X[:, 1][y == -1],
                        s=90, label='label = -1')
            plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1],
                        marker='s', s=90, label='label = 1')
            plt.xlabel("$X_1$")
            plt.ylabel("$X_2$")
            plt.grid()
            plt.legend()

            plt.pause(0.001)
        plt.show()


def main():
    # 载入数据集1
    x_train1 = np.load("./data/s-svm/train_data.npy")  # shape: (70, 2)
    y_train1 = np.load("./data/s-svm//train_target.npy")  # shape: (70, )
    x_test1 = np.load("./data/s-svm//test_data.npy")  # shape: (30, 2)
    y_test1 = np.load("./data/s-svm//test_target.npy")  # shape: (30, )

    # 训练模型，求出 W,b
    svm = SVM()
    W, b = svm.fit(x_train1, y_train1)  # 训练模型
    print("W=", W)
    print("b=", b)

    # 训练集和测试集精度
    print("数据集1的训练集精度为：{:.2f} %".format(
        (svm.predict(x_train1) == y_train1).mean() * 100))
    print("数据集1的测试集精度为：{:.2f} %".format(
        (svm.predict(x_test1) == y_test1).mean() * 100))

    # 动态展示梯度下降法优化SVM模型的过程
    svm.show_dynamic(x_train1, y_train1)

    # 作训练集结果图(包括决策边界)
    svm.show_result(x_train1, y_train1, "D1 Train")

    # 作测试集结果图(包括决策边界)
    svm.show_result(x_test1, y_test1, "D1 Test")


if __name__ == "__main__":
    main()

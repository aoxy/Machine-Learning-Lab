import numpy as np
import random
import copy
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k):
        self.k = k
        self.mu = None
        self.label = None
        self.saved = []

    def fit(self, raw_X, show=False, normalize=True):
        X = copy.deepcopy(raw_X)
        m = X.shape[0]  # 样本数
        d = X.shape[1]  # 维度

        # 归一化数据
        if normalize:
            for i in range(d):
                max_Xi = np.max(np.abs(X[:, i]))
                X[:, i] = X[:, i]/max_Xi

        # 初始向量设置
        sample = random.sample(range(m), self.k)
        self.mu = copy.deepcopy(X[sample])
        raw_mu = copy.deepcopy(raw_X[sample])

        # 迭代优化
        self.saved = []
        self.label = np.zeros(m, dtype=int)
        while True:
            old_label = copy.deepcopy(self.label)
            for i in range(m):
                self.label[i] = np.argmin(self.dist(X[i], self.mu, 2))
            if show:
                label = copy.deepcopy(self.label)
                self.saved.append((label, raw_mu))
                raw_mu = self.calculate_mu(raw_X, self.label)
            self.mu = self.calculate_mu(X, self.label)
            if (old_label == self.label).all():
                # 样本分类类别标签不再发生变化时终止迭代
                break
        self.mu = self.calculate_mu(raw_X, self.label)
        if show:
            self.show_dynamic(raw_X, self.saved)
        return self.label, self.mu

    def calculate_mu(self, X, label):
        mu = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            class_k = X[np.where(label == k)]
            if len(class_k) != 0:
                mu[k] = np.mean(class_k, axis=0)
            else:
                mu[k] = copy.deepcopy(X[random.randint(0, X.shape[0]-1)])
        return mu

    def dist(self, x, y, dim=1):
        if dim == 1:
            return np.sqrt(np.sum(np.square(x-y)))
        else:
            return np.sqrt(np.sum(np.square(x-y), axis=1))

    def dbi(self, X, label, mu):
        avg_C = np.zeros(self.k)
        for k in range(self.k):
            X_k = X[np.where(label == k)]
            if len(X_k) == 0:
                avg_C[k] = 0
            else:
                avg_C[k] = np.sum(self.dist(mu[k], X_k, 2))/len(X_k)
        kdbi = 0
        for i in range(self.k):
            max_val = -float("inf")
            for j in range(self.k):
                if i == j:
                    continue
                max_val = max(
                    max_val, (avg_C[i]+avg_C[j])/self.dist(mu[i], mu[j]))
            kdbi += max_val
        return kdbi/self.k

    def plot_result(self, X, label, mu):
        plt.figure(figsize=(15, 12))
        plt.title("Clustering Result")
        color = ['r', 'b', 'g', 'k', 'm', 'w', 'y', 'c']
        for k in range(self.k):
            x = X[np.where(self.label == k)][:, 0]
            y = X[np.where(self.label == k)][:, 1]
            plt.scatter(x, y, s=90, label='class_{}'.format(k),
                        c=color[k % len(color)], alpha=0.3)
            plt.scatter(mu[k, 0], mu[k, 1], s=200, label='center_{}'.format(
                k), marker='+', c=color[k % len(color)], linewidths=3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def show_dynamic(self, X, saved):
        color = ['r', 'b', 'g', 'k', 'm', 'w', 'y', 'c']
        for i in range(len(saved)):
            plt.figure(figsize=(15, 12))
            plt.clf()
            label = saved[i][0]
            mu = saved[i][1]
            for k in range(self.k):
                x = X[np.where(label == k)][:, 0]
                y = X[np.where(label == k)][:, 1]
                plt.scatter(x, y, s=90, label='class_{}'.format(k),
                            c=color[k % len(color)], alpha=0.3)
                plt.scatter(mu[k, 0], mu[k, 1], s=200, label='center_{}'.format(
                    k), marker='+', c=color[k % len(color)], linewidths=3)
            plt.legend()
            plt.title('Iteration: #{}'.format(i))
            plt.show()


def plot_raw(data):
    plt.figure(figsize=(15, 12))
    plt.title("Raw Data")
    color = ['r', 'b', 'g', 'k', 'm', 'w', 'y', 'c']
    for index, class_n in enumerate(data):
        xy = np.array(data[class_n])
        plt.scatter(xy[:, 0], xy[:, 1], s=90, label='label = {}'.format(index),
                    c=color[index % len(color)])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def main():
    # 载入数据集
    data = np.load("./data/k-means/k-means.npy", allow_pickle=True).item()
    plot_raw(data)  # 展示理想的聚类结果
    class_0 = data['class_0']
    class_1 = data['class_1']
    class_2 = data['class_2']
    raw_data = class_0+class_1+class_2
    X = np.array(raw_data)  # shape: (150, 2)

    # 训练模型
    kmeans = KMeans(k=3)
    label, mu = kmeans.fit(X, show=True)

    # 分类结果
    print("簇中心坐标：\n", mu)
    dbi = kmeans.dbi(X, label, mu)
    print("DBI = {:.4f}".format(dbi))
    kmeans.plot_result(X, label, mu)


if __name__ == "__main__":
    main()

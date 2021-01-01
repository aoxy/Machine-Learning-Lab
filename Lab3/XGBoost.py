import numpy as np


class XGBoostTree:
    class Node:
        def __init__(self, feature=None, threshold=None,
                     left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, split_size=2, min_gain=1e-5,
                 max_depth=3, loss=None, la=0, gamma=0, eta=0.1):
        self.root = None
        self.split_size = split_size
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.loss = loss
        self.la = la
        self.gamma = gamma
        self.eta = eta

    def fit(self, X, y, p):
        G = self.loss.grad(y, p)
        H = self.loss.hess(y, p)
        XypGH = np.c_[X, y, p, G, H]
        self.root = self.build_tree(XypGH)

    def build_tree(self, XypGH, cur_depth=1):
        max_gain = 0
        LXyp = None
        RXyp = None
        feature = None
        threshold = None
        split_point = None
        size = np.shape(XypGH)[0]
        d = np.shape(XypGH)[1] - 4  # 属性个数
        if size >= self.split_size and cur_depth <= self.max_depth:
            for f in range(d):
                XypGH = XypGH[np.lexsort(XypGH.T[f, None])]
                unique_values = np.unique(XypGH[:, f])
                for fv in unique_values:
                    split_i = np.searchsorted(XypGH[:, f], fv)
                    g_l = np.sum(XypGH[:split_i, -2])
                    g_r = np.sum(XypGH[split_i:, -2])
                    h_l = np.sum(XypGH[:split_i, -1])
                    h_r = np.sum(XypGH[split_i:, -1])
                    gain = 0.5*(np.square(g_l)/(h_l+self.la)
                                + np.square(g_r)/(h_r+self.la)
                                - np.square(g_l+g_r)/(h_l+h_r+self.la))
                    - self.gamma
                    if gain >= max_gain and gain > self.min_gain:
                        max_gain = gain
                        feature = f
                        threshold = fv
                        split_point = split_i
            if max_gain > self.min_gain:
                XypGH = XypGH[np.lexsort(XypGH.T[feature, None])]
                LXypGH = XypGH[:split_point]
                RXypGH = XypGH[split_point:]
                left = self.build_tree(LXypGH, cur_depth + 1)
                right = self.build_tree(RXypGH, cur_depth + 1)
                return self.Node(feature, threshold, left, right)
        g = np.sum(XypGH[:, -2])
        h = np.sum(XypGH[:, -1])
        leaf_value = self.eta*(-g / (h + self.la))
        return self.Node(value=leaf_value)

    def predict_value(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature]
        if feature_value < tree.threshold:
            return self.predict_value(x, tree.left)
        else:
            return self.predict_value(x, tree.right)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x, self.root))
        p = np.array(y_pred)
        return p


class LogisticLoss():
    def grad(self, y, p):
        p = 1 / (1 + np.exp(-p))
        return p - y

    def hess(self, y, p):
        p = 1 / (1 + np.exp(-p))
        return p * (1 - p)


class XGBoost:
    def __init__(self, n_estimators=3, split_size=2, min_gain=1e-5, max_depth=4, la=1, gamma=0, eta=0.5):
        self.n_estimators = n_estimators
        self.split_size = split_size
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.la = la
        self.gamma = gamma
        self.eta = eta
        self.loss = LogisticLoss()
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostTree(
                split_size=self.split_size,
                min_gain=self.min_gain,
                max_depth=self.max_depth,
                loss=self.loss,
                la=self.la,
                gamma=self.gamma,
                eta=self.eta)
            self.trees.append(tree)

    def fit(self, X, y):
        p = np.zeros(np.shape(y))
        for tree in self.trees:
            tree.fit(X, y, p)
            p += tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += tree.predict(X)
        h = 1.0/(1.0+np.exp(-y_pred))
        return np.round(h)

    def plot_tree_aux(self, g, branch, name):
        left = branch.left
        right = branch.right
        if left.value is None:
            g.node(name+'l', "f{} < {:.7g}".format(left.feature, left.threshold))
            g.edge(name, name+'l', color='blue', label='yes')
            self.plot_tree_aux(g, left, name+'l')
        else:
            g.node(name+'l', "leaf = {:.7g}".format(left.value), shape='box')
            g.edge(name, name+'l', color='blue', label='yes')
        if right.value is None:
            g.node(name+'r', "f{} < {:.7g}".format(right.feature, right.threshold))
            g.edge(name, name+'r', color='red', label='no')
            self.plot_tree_aux(g, right, name+'r')
        else:
            g.node(name+'r', "leaf = {:.7g}".format(right.value), shape='box')
            g.edge(name, name+'r', color='red', label='no')

    def plot_tree(self, num_trees=0):
        from graphviz import Digraph
        tree = self.trees[num_trees]
        root = tree.root
        g = Digraph()
        if root.value is None:
            g.node('root', "f{} < {:.7g}".format(root.feature, root.threshold))
            self.plot_tree_aux(g, root, 'root')
        else:
            g.node('root', "leaf = {:.7g}".format(root.value), shape='box')
        g.render('XGBoostTree%s' % (num_trees + 1), format='svg', cleanup=True)


def main():
    # 载入数据集
    x_train = np.load("./data/train_data.npy")  # shape: (615, 8)
    y_train = np.load("./data/train_target.npy")  # shape: (615,)
    x_test = np.load("./data/test_data.npy")  # shape: (153, 8)
    y_test = np.load("./data/test_target.npy")  # shape: (153,)

    # 训练模型
    xgbt = XGBoost()
    xgbt.fit(x_train, y_train)

    # 训练集和测试集精度
    p_train = xgbt.predict(x_train)
    p_test = xgbt.predict(x_test)
    print("训练集精度为：{:.2f} %".format(
        (p_train == y_train).mean() * 100))
    print("测试集精度为：{:.2f} %".format(
        (p_test == y_test).mean() * 100))

    # 画出决策子树
    xgbt.plot_tree(0)
    xgbt.plot_tree(1)
    xgbt.plot_tree(2)


if __name__ == "__main__":
    main()

import numpy as np
import string
import tqdm
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


class LDA:
    def __init__(self, texts, K, cold_count):
        self.K = K  # 主题个数
        self.V = 0  # 字典单词个数
        self.Vset = None  # 字典
        self.D = len(texts)  # 文档篇数
        self.Dset = []  # 文档列表
        self.preprocess(texts, cold_count)  # 文本清洗预处理
        self.Vset_to_index = {}  # 单词在字典中的序号
        for v in range(self.V):
            self.Vset_to_index[self.Vset[v]] = v
        self.delta = np.zeros((self.K, self.V))  # delta[k][v]是第k个主题的第v个单词个数
        self.n_k = np.zeros(self.K)  # 第k个主题单词总数
        self.sigma = np.zeros((self.D, self.K))  # sigma[m][k]第m个文档第k个主题的单词个数
        self.n_m = np.zeros(self.D)  # 第m个文档有多少个单词
        self.alpha = np.ones(self.K)
        self.theta = np.ones((self.D, self.K))
        self.beta = np.ones(self.V)
        self.varphi = np.ones((self.K, self.V))
        self.z = []  # z[m][n]是文档m的第n个单词的话题
        for m in range(self.D):
            self.z.append(np.zeros_like(self.Dset[m]))

    def get_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess(self, texts, cold_count):
        Vset = np.array([], dtype='<U112')
        print("分词处理......")
        for doc in tqdm.tqdm(texts):
            words = word_tokenize(doc.lower())
            words_tag = pos_tag(words)
            lematizer = WordNetLemmatizer()
            words_lemmatized = [lematizer.lemmatize(
                word, self.get_pos(tag)) for word, tag in words_tag]
            words_cleaned = [word for word in words_lemmatized if (len(word) >= 3) and (
                word not in stopwords.words('english')) and (word not in string.punctuation) and word.isalpha()]
            Vset = np.concatenate((Vset, words_cleaned))
            self.Dset.append(np.array(words_cleaned))
        self.Vset = np.unique(Vset)
        wcount = {}
        for m in range(len(self.Dset)):
            for n in range(len(self.Dset[m])):
                word = self.Dset[m][n]
                wcount[word] = wcount.get(word, 0) + 1
        print("去除低频词......")
        for m in tqdm.tqdm(range(self.D)):
            i = 0
            while i < len(self.Dset[m]):
                w = self.Dset[m][i]
                if wcount[w] < cold_count:
                    self.Vset = np.delete(self.Vset, np.where(self.Vset == w))
                    self.Dset[m] = np.delete(self.Dset[m], i)
                else:
                    i += 1
        self.V = len(self.Vset)

    def gibbs_sampling(self, epoch=100):
        print("吉布斯采样......")
        for _ in tqdm.tqdm(range(epoch)):
            for m in range(self.D):
                for v in range(len(self.Dset[m])):
                    self.z[m][v] = self.topic_updated(m, v)

    def topic_updated(self, m, v):
        topic_old = int(self.z[m][v])
        self.delta[topic_old][self.Vset_to_index[self.Dset[m][v]]] -= 1
        self.n_k[topic_old] -= 1
        self.sigma[m][topic_old] -= 1
        self.n_m[m] -= 1
        p = np.zeros(self.K)
        for k in range(self.K):
            p[k] = (self.sigma[m][k] + self.alpha[k]) / \
                (self.n_m[m] + np.sum(self.alpha)) * \
                (self.delta[k][self.Vset_to_index[self.Dset[m][v]]] +
                 self.beta[self.Vset_to_index[self.Dset[m][v]]]) / \
                (self.n_k[k] + np.sum(self.beta))
        p = p / np.sum(p)
        topic_new = np.argmax(np.random.multinomial(1, p))
        self.delta[topic_new][self.Vset_to_index[self.Dset[m][v]]] += 1
        self.n_k[topic_new] += 1
        self.sigma[m][topic_new] += 1
        self.n_m[m] += 1
        return topic_new

    def cal_theta_varphi(self):
        for j in range(self.D):
            for k in range(self.K):
                self.theta[j][k] = (
                    self.sigma[j][k] + self.alpha[k]) / \
                    (self.n_m[j] + np.sum(self.alpha))
        for k in range(self.K):
            for v in range(self.V):
                self.varphi[k][v] = (
                    self.delta[k][v] + self.beta[v]) / \
                    (self.n_k[k] + np.sum(self.beta))

    def train(self, epoch):
        for m in range(self.D):
            self.n_m[m] = len(self.Dset[m])
            for v in range(len(self.Dset[m])):
                topic = int(np.random.randint(0, self.K))
                self.z[m][v] = topic
                self.delta[topic][self.Vset_to_index[self.Dset[m][v]]] += 1
                self.n_k[topic] += 1
                self.sigma[m][topic] += 1
        self.gibbs_sampling(epoch)
        self.cal_theta_varphi()

    def top_words(self, limit):
        top_words = []
        for k in range(self.K):
            top_idx = self.varphi[k].argsort()[::-1][0:limit]
            top_words.append(list(self.Vset[top_idx]))
        return top_words


def main():
    # 载入数据集
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    texts = np.load("./data/text.npy")
    lda = LDA(texts, K=20, cold_count=8)

    # 训练模型
    lda.train(60)

    # 输出20个主题的top10的词
    top10words = lda.top_words(10)
    for i in range(lda.K):
        print(top10words[i])
    np.savetxt("top10words.txt", top10words, '%s', delimiter=',')


if __name__ == '__main__':
    main()

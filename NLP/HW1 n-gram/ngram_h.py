#!/usr/bin/python3

import re
import itertools
import functools
import math

from typing import List, Dict, Tuple
from typing import Iterator

import pickle as pkl

Sentence = List[str]
IntSentence = List[int]

Corpus = List[Sentence]
IntCorpus = List[IntSentence]

Gram = Tuple[int]

_splitor_pattern = re.compile(r"[^a-zA-Z']+|(?=')")
_digit_pattern = re.compile(r"\d+")


def normaltokenize(corpus: List[str]) -> Corpus:
    """
    Normalizes and tokenizes the sentences in `corpus`. Turns the letters into
    lower case and removes all the non-alphadigit characters and splits the
    sentence into words and added BOS and EOS marks.

    Args:
        corpus - list of str

    Return:
        list of list of str where each inner list of str represents the word
          sequence in a sentence from the original sentence list
    """

    tokeneds = [["<s>"]
                + list(
        filter(lambda tkn: len(tkn) > 0,
               _splitor_pattern.split(
                   _digit_pattern.sub("N", stc.lower()))))
                + ["</s>"]
                for stc in corpus
                ]
    return tokeneds


def extract_vocabulary(corpus: Corpus) -> Dict[str, int]:
    """
    Extracts the vocabulary from `corpus` and returns it as a mapping from the
    word to index. The words will be sorted by the codepoint value.

    Args:
        corpus - list of list of str

    Return:
        dict like {str: int}
    """

    vocabulary = set(itertools.chain.from_iterable(corpus))
    vocabulary = dict(
        map(lambda itm: (itm[1], itm[0]),
            enumerate(
                sorted(vocabulary))))
    return vocabulary


def words_to_indices(vocabulary: Dict[str, int], sentence: Sentence) -> IntSentence:
    """
    Convert sentence in words to sentence in word indices.

    Args:
        vocabulary - dict like {str: int}
        sentence - list of str

    Return:
        list of int
    """

    return list(map(lambda tkn: vocabulary.get(tkn, len(vocabulary)), sentence))


# OPTIONAL: implement a DictTree instead of using a flattern dict
# class DictTree:
# def __init__(self):
# pass
#
# def __len__(self) -> int:
# pass
#
# def __iter__(self) -> Iterator[Gram]:
# pass
#
# def __contains__(self, key: Gram):
# pass
#
# def __getitem__(self, key: Gram) -> int:
# pass
#
# def __setitem__(self, key: Gram, frequency: int):
# pass
#
# def __delitem__(self, key: Gram):
# pass

import math


class NGramModel:
    def __init__(self, vocab_size: int, n: int = 4):
        """
        Constructs `n`-gram model with a `vocab_size`-size vocabulary.

        Args:
            vocab_size - int
            n - int
        """

        self.vocab_size: int = vocab_size
        self.n: int = n

        self.frequencies: List[Dict[Gram, int]] \
            = [{} for _ in range(n)]

        # 我认为应该更改为float 10/20 11:17
        self.disfrequencies: List[Dict[Gram, float]] \
            = [{} for _ in range(n)]

        self.ncounts: Dict[Gram
        , Dict[int, int]
        ] = {}

        self.discount_threshold: int = 7
        self._d: Dict[Gram, Tuple[float, float]] = {}
        self._alpha: List[Dict[Gram, float]] \
            = [{} for _ in range(n)]

        self.eps = 1e-10

    def learn(self, corpus: IntCorpus):
        """
        Learns the parameters of the n-gram model.

        Args:
            corpus - list of list of int
        """

        # self.n: lenth of n-gram
        for stc in corpus:
            # print(stc)
            for i in range(1, len(stc) + 1):  # 查看到i为止的序列
                for j in range(min(i, self.n)):  # 以j作为gram的迭代
                    # TODO: count the frequencies of the grams
                    tmp_gram = stc[i - j - 1:i]
                    tmp_gram = tuple(tmp_gram)
                    if self.frequencies[j].get(tmp_gram, 0):
                        self.frequencies[j][tmp_gram] = self.frequencies[j][tmp_gram] + 1

                        # print(self.frequencies[j][tmp_gram])
                    else:
                        self.frequencies[j][tmp_gram] = 1

                    # print(self.frequencies[j][tmp_gram])
        # print(self.frequencies[1])

        for i in range(1, self.n):  # 按照gram长度划分
            grams = itertools.groupby(
                sorted(
                    sorted(
                        map(lambda itm: (itm[0][:-1], itm[1]),  # 去除最后一项后的键值对
                            self.frequencies[i].items()),
                        key=(lambda itm: itm[1])),
                    key=(lambda itm: itm[0])))  # 再按照第一位元素排序， 后面的值为频率
            # TODO: calculates the value of $N_r$

            for key, group in grams:  # key[0]: 表示上一项的值, key[1]: 下一项出现频次

                cnt = 0
                for _ in group:
                    cnt = cnt + 1

                if key[0] in self.ncounts:
                    self.ncounts[key[0]][key[1]] = cnt  # 上一长度gram: (多一个元素所有可能的频次: 该频次的出现次数)
                else:
                    self.ncounts[key[0]] = {}
                    self.ncounts[key[0]][key[1]] = cnt

                # 记录了: 对本项(key)，可能的下一项的(r , N_r)

        return self.frequencies

    # 输入gram,计算其对应插值参数d
    # func 'd' not debugged yet
    def d(self, gram: Gram) -> float:
        """
        Calculates the interpolation coefficient.

        Args:
            gram - tuple of int

        Return:
            float
        """
        # r即本gram的出现频次
        length = len(gram)
        r = self.frequencies[length].get(gram, self.eps)

        # 调用self.ncounts， 查看下一项的不同频次的出现次数
        if gram not in self._d:
            # print(gram)
            # print(gram[:-1])
            # TODO: calculates the value of $d'$
            ncounts = self.ncounts[gram[:-1]]  # 查看前缀, counts是存储频次的字典
            # print(ncounts)

            def counts(num):
                if num in ncounts:
                    return float(ncounts[num])
                else:
                    return float(self.eps)  # 如有报错,改为0

            lamda = float(counts(1) / (counts(1) - 8 * counts(8)))
            N_r = counts(r)
            N_r_plus1 = counts(r + 1)
            d_dot = lamda * ((r + 1) * N_r_plus1) / (r * N_r) + 1 - lamda  # 求出各项参数 及d'

            self._d[gram] = (d_dot, 1.0)

        # self._d[gram] = (numerator1 / denominator, - numerator2 / denominator)  # 10/19 ques ?

        if r > 7:
            return self._d[gram][1]
        else:
            return self._d[gram][0]

    def alpha(self, gram: Gram) -> float:  # '回退' 意为降低gram大小要求
        """
        Calculates the back-off weight alpha(`gram`)

        Args:
            gram - tuple of int

        Return:
            float
        """

        n = len(gram)
        # 概率存放在 disfrequencies 内
        if gram not in self._alpha[n]:
            if gram in self.frequencies[n - 1]:
                # TODO: calculates the value of $\alpha$

                # V+, V- Accumulated, sum add
                V_plus = []
                V_minus = []

                sum_pls = 0
                sum_minus = 0
                for i in range(1, self.vocab_size):
                    # 添加元素:查询V+和V-
                    index = len(gram)  # 减一是原长度 checked in test_funcs 10/19 21:35
                    gram_add = list(gram)
                    gram_add.append(i)
                    gram_add = tuple(gram_add)

                    if gram_add in self.frequencies[index]:
                        # print(gram_add)
                        # print(self.frequencies[index].get(gram_add, 0))
                        V_plus.append(i)
                        sum_pls = sum_pls + self.disfrequencies[index].get(gram_add, 0)  # checked in test
                        # sum_pls = sum_pls + self.disfrequencies[index].get(gram_add, 0)
                    else:
                        V_minus.append(i)
                        gram_minus = gram_add[1:]
                        index = index - 1

                        sum_minus = sum_minus + self.disfrequencies[index - 1].get(gram_minus, self.eps)

                numerator = 1 - sum_pls
                denominator = 1 - sum_minus

                self._alpha[n][gram] = numerator / denominator
            else:
                self._alpha[n][gram] = 1.
        return self._alpha[n][gram]

    def __getitem__(self, gram: Gram) -> float:  # 计算回退概率(调用之前实现的函数)
        """
        Calculates smoothed conditional probability P(`gram[-1]`|`gram[:-1]`).

        Args:
            gram - tuple of int

        Return:
            float
        """

        n = len(gram) - 1

        if gram not in self.disfrequencies[n]:
            if n > 0:
                # TODO: calculates the smoothed probability value according to the formulate

                # 1:C>0, use param_d
                if self.disfrequencies[n].get(gram, 0) > 0:
                    param_d = self.d(gram)
                    # print('param_d = ', param_d)
                    Ck = self.frequencies[n][gram]
                    Ck_1 = self.frequencies[n - 1][gram[:-1]]
                    P = param_d * Ck / Ck_1
                    self.disfrequencies[n][gram] = P

                # 2:C=0, use param_a
                else:
                    param_a = self.alpha(gram[:-1])
                    # print('param_a = ', param_a)
                    # P = self.disfrequencies[n-1].get(gram[1:], self.eps) * param_a
                    P = self.__getitem__(gram[1:]) * param_a
                    self.disfrequencies[n][gram] = P

            # 第一项
            else:
                self.disfrequencies[n][gram] = self.frequencies[n].get(gram, self.eps) / float(len(self.frequencies[0]))
        return self.disfrequencies[n][gram]

    # 交叉熵
    def log_prob(self, sentence: IntSentence) -> float:
        """
        Calculates the log probability of the given sentence. Assumes that the
        first token is always "<s>".

        Args:
            sentence: list of int

        Return:
            float
        """

        log_prob = 0.
        cnt = 0
        n = self.n
        for i in range(2, len(sentence) + 1):
            # TODO: calculates the log probability
            # 思路是递进地对每一个ngram求概率
            cnt = float(cnt+1)
            length = min(n, i)
            gram = tuple(sentence[i-length:i])  # 可能超出范围,有待测试
            # print(len(gram))
            # print(gram)
            log_prob = log_prob + math.log2(self.__getitem__(gram))

        log_prob = - log_prob / cnt
        return log_prob


    def ppl(self, sentence: IntSentence) -> float:
        """
        Calculates the PPL of the given sentence. Assumes that the first token
        is always "<s>".

        Args:
            sentence: list of int

        Return:
            float
        """

        # calculates the PPL

        PPL = 1.
        cnt = 0
        n = self.n
        for i in range(2, len(sentence) + 1):
            # calculates the log probability
            cnt = cnt + 1
            length = min(n, i)
            gram = tuple(sentence[i - length + 1:i])  # 可能超出范围,有待测试
            PPL = PPL / self.__getitem__(gram)

        # print(PPL)
        # print(cnt)

        PPL = math.pow(PPL, 1/cnt)
        return PPL


if __name__ == "__main__":


    action = "eval"
    eval_type = "prob"

    if action == "train":
        with open("data/news.2007.en.shuffled.deduped.train", encoding='UTF-8') as f:
            texts = list(map(lambda l: l.strip(), f.readlines()))

        # 下面是用部分训练数据测试代码准确性
        # with open("data/train_part.txt", encoding='UTF-8') as f:
        #     texts = list(map(lambda l: l.strip(), f.readlines()))

        print("Loaded training set.")

        corpus = normaltokenize(texts)
        vocabulary = extract_vocabulary(corpus)
        # print(vocabulary)
        corpus = list(
            map(functools.partial(words_to_indices, vocabulary),
                corpus))

        print("Preprocessed training set.")

        model = NGramModel(len(vocabulary))
        model.learn(corpus)
        with open("model.pkl", "wb") as f:
            pkl.dump(vocabulary, f)
            pkl.dump(model, f)

        print("Dumped model.")

    elif action == "eval":
        with open("model.pkl", "rb") as f:
            vocabulary = pkl.load(f)
            model = pkl.load(f)
        print("Loaded model.")

        with open("data/news.2007.en.shuffled.deduped.test") as f:
            test_set = list(map(lambda l: l.strip(), f.readlines()))
        test_corpus = normaltokenize(test_set)
        test_corpus = list(
            map(functools.partial(words_to_indices, vocabulary),
                test_corpus))

        # 输出两种评估方式的结果
        if eval_type == "ppl":
            Left = len(test_corpus)
            Sum = 0
            ppls = []
            for t in test_corpus:
                # print(t)
                # 计数
                Sum = Sum + 1
                Left = Left - 1
                print(Left, " Sentences left")
                print(Sum, " Sentences done")

                ppls.append(model.ppl(t))
                print(ppls[-1])
            print("Avg of ppls: ", sum(ppls) / len(ppls))

        if eval_type == "prob":

            Left = len(test_corpus)
            Sum = 0
            probs = []
            for t in test_corpus:
                Sum = Sum + 1
                Left = Left - 1
                print(Left, " Sentences left")
                print(Sum, " Sentences done")
                # print(t)
                probs.append(model.log_prob(t))
                print(probs[-1])
            print("Avg of probs: ", sum(probs) / len(probs))

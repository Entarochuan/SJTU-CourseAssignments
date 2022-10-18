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

    tokeneds = [ ["<s>"]
               + list(
                   filter(lambda tkn: len(tkn)>0,
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
#class DictTree:
    #def __init__(self):
        #pass
#
    #def __len__(self) -> int:
        #pass
#
    #def __iter__(self) -> Iterator[Gram]:
        #pass
#
    #def __contains__(self, key: Gram):
        #pass
#
    #def __getitem__(self, key: Gram) -> int:
        #pass
#
    #def __setitem__(self, key: Gram, frequency: int):
        #pass
#
    #def __delitem__(self, key: Gram):
        #pass

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

        # 用self.disfrequencies存储词频统计后结果
        self.frequencies: List[Dict[Gram, int]] \
            = [{} for _ in range(n)]
        self.disfrequencies: List[Dict[Gram, int]] \
            = [{} for _ in range(n)]

        self.Nr_cnt = {}  # 以出现次数为索引记录不同出现次数gram的个数

        self.ncounts: Dict[Gram
        , (int, int, int)  # 按照gram: r, N_r, N_(r+1)的格式来记录
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
            for i in range(1, len(stc) + 1):  # start from point i, 0th not counted
                for j in range(min(i, self.n)):
                    # TODO: count the frequencies of the grams
                    # len of gram = j
                    #                     if (i+j)<=len(stc) :
                    tmp_gram = stc[i:i + j + 1]
                    tmp_gram = tuple(tmp_gram)
                    if self.frequencies[j].get(tmp_gram, 0):
                        self.frequencies[j][tmp_gram] = self.frequencies[j][tmp_gram] + 1
                    else:
                        self.frequencies[j][tmp_gram] = 1
                    # print(self.frequencies[j][tmp_gram])
        # print(self.frequencies[1])

        for i in range(1, self.n):  # 按照gram长度划分
            # for key, value in self.frequencies[i].items():
            #     print(key[:-1])
            #     print(value)

            grams = itertools.groupby(
                sorted(
                    sorted(
                        map(lambda itm: (itm[0][:-1], itm[1]),  # 长度为i的gram内(最后一项，)的键值对， 按照频率从小到大排序
                            self.frequencies[i].items()),
                        key=(lambda itm: itm[1])),
                    key=(lambda itm: itm[0])))  # 再按照第一位元素排序， 后面的值为频率
            # TODO: calculates the value of $N_r$

            flag = False
            for key, group in grams:  # key[0]表示值, key[1]表示出现次数
                # print(key[0], key[1])

                cnt = 0
                for item in group:
                    print(item)
                    cnt = cnt + 1

                self.Nr_cnt[key[1]] = cnt


        # 为每个gram赋值, 23:30思路：直接在上面的函数改写
        for i in range(1, self.n):
            tmp_list = [0, 0, 0]  # (r, Nr, Nr+1)
            # print(self.frequencies[i])
            for (key, value) in zip(self.frequencies[i].keys(), self.frequencies[i].values()):  # key:gram , value:词频
                value = int(value)
                print(key, value, 'dealing')
                tmp_list[0] = value
                tmp_list[1] = self.Nr_cnt[value]

                tmp_list[2] = 0
                print(key[1])
                tmp_list[2] = self.Nr_cnt[key[1]+1]


                tmp_tuple = tuple(tmp_list)
                self.ncounts[key] = tmp_tuple

    def d(self, gram: Gram) -> float:
        """
        Calculates the interpolation coefficient.

        Args:
            gram - tuple of int

        Return:
            float
        """

        if gram not in self._d:
            # TODO: calculates the value of $d'$
            pass
            self._d[gram] = (numerator1 / denominator, - numerator2 / denominator)
        return self._d[gram]

    def alpha(self, gram: Gram) -> float:
        """
        Calculates the back-off weight alpha(`gram`)

        Args:
            gram - tuple of int

        Return:
            float
        """

        n = len(gram)
        if gram not in self._alpha[n]:
            if gram in self.frequencies[n - 1]:
                # TODO: calculates the value of $\alpha$
                pass
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
                # TODO: calculates the smoothed probability value according to the formulae
                pass
            else:
                self.disfrequencies[n][gram] = self.frequencies[n].get(gram, self.eps) / float(len(self.frequencies[0]))
        return self.disfrequencies[n][gram]

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
        for i in range(2, len(sentence) + 1):
            # TODO: calculates the log probability
            pass
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

        # TODO: calculates the PPL
        pass

if __name__ == "__main__":
    action = "train"

    if action=="train":
        with open("data/news.2007.en.shuffled.deduped.train") as f:
            texts = list(map(lambda l: l.strip(), f.readlines()))

        print("Loaded training set.")

        corpus = normaltokenize(texts)
        vocabulary = extract_vocabulary(corpus)
        #print(vocabulary)
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

    elif action=="eval":
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
        ppls = []
        for t in test_corpus:
            #print(t)
            ppls.append(model.ppl(t))
            print(ppls[-1])
        print("Avg: ", sum(ppls)/len(ppls))

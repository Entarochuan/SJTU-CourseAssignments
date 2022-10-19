import re
import itertools
import functools
import math

from typing import List, Dict, Tuple
from typing import Iterator
from typing import List, Dict, Tuple
import pickle as pkl
import functools
from ngram_h import *


if __name__ == "__main__":

    Sentence = List[str]
    IntSentence = List[int]

    Corpus = List[Sentence]
    IntCorpus = List[IntSentence]

    Gram = Tuple[int]

    with open("data/train_part.txt", encoding='UTF-8') as f:
        texts = list(map(lambda l: l.strip(), f.readlines()))

    print("Loaded training set.")

    corpus = normaltokenize(texts)
    vocabulary = extract_vocabulary(corpus)
    corpus = list(
            map(functools.partial(words_to_indices, vocabulary),
                corpus))

    print("Preprocessed training set.")

    model = NGramModel(len(vocabulary))
    frequencies = model.learn(corpus)

    i = 1
    # print(frequencies[0])
    map1 = map(lambda itm: (itm[0][:-1], itm[1]),  # 去除最后一项后的键值
                            frequencies[i].items())

    sort1 = sorted(
                        map(lambda itm: (itm[0][:-1], itm[1]),  # 长度为i-1的gram内的键值对， 按照频率从小到大排序
                            frequencies[i].items()),
                        key=(lambda itm: itm[1]))

    sort2 = sorted(
                    sort1,
                    key=(lambda itm: itm[0]))


    # print(sort1)
    # print(sort2)

    grams = itertools.groupby(
        sorted( sort2,
            key=(lambda itm: itm[0])))

    print('testing')
    cnt = 0
    # print(model.vocab_size)
    for key, group in grams:
        cnt = cnt + 1
        print(key[0])
        tmp_gram = key[0]
        # print('start')
        # print(tmp_gram)
        # print(tmp_gram[:-1])
        # print(tmp_gram[1:])
        # print('len=', len(tmp_gram[:-1]))
        # print(model.ncounts[tmp_gram])
        # print('end')


        # tmp_gram = list(tmp_gram)
        # # print(tmp_gram_add)
        # tmp_gram.append(1)
        # tmp_gram = tuple(tmp_gram)
        index = len(tmp_gram) - 1
        print(tmp_gram, index)
        if tmp_gram in model.frequencies[index]:
            print(tmp_gram)
            print('yes')
        else:
            print('No')
        # print(model.frequencies[index][tmp_gram])


        # print(tmp_gram_add)

        if cnt == 1000:
            break

    print(model.vocab_size)

    # for i in range(model.n):
    #     print(model.frequencies[i])

    # print(model.frequencies[i])
import re
import itertools
import functools
import math

from typing import List, Dict, Tuple
from typing import Iterator

import pickle as pkl
from ngram_h import *


if __name__ == "__main__":

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
    model.learn(corpus)

    print(model.ncounts)

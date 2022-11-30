### BPE英文分词

下面简要介绍实现的各项函数及其功能。



#### 1. `build_bpe_vocab`

按照要求，对每个单词字母间加上空格操作，作为字典的键值。



#### 2.`get_bigram_freq`

统计词典中所有`bigram`的出现频次。



#### 3. `refresh_bpe_vocab_by_merging_bigram`

对字典中的所有键值，判断是否符合输入的bigram，有则将合并后的bigram输入作为替代。



#### 4. `get_bpe_tokens`

将字典中键值按空格切分，得到分词状态的新词典。



#### 5. `print_bpe_tokenize`

按照长度从词表中依次提取分词与句子对比，如匹配则递归地将剩下的句子按照此方法继续查询匹配，直到整个句子都被匹配完毕。



#### 6. `结果`

```python
naturallanguageprocessing 的分词结果为：
n a tur all an g u ag e pro c es s ing</w> 
```


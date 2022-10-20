# Ngram Report



### 一、模型参数

模型主要的参数包括以下几项:

```python
self.frequencies: List[Dict[Gram, int]] \
            = [{} for _ in range(n)]
            
self.disfrequencies: List[Dict[Gram, float]] \
            = [{} for _ in range(n)]
            
self.ncounts: Dict[Gram
        , Dict[int, int]] = {}

self._d: Dict[Gram, Tuple[float, float]] = {}

self._alpha: List[Dict[Gram, float]] \
            = [{} for _ in range(n)]
            
```

下面逐一解析各项参数的含义：

`self.frequencies` : 统计并存放所有gram的出现频次。

`self.disfrequencies` : 存储所有gram的出现概率(在`__getitem__`中被计算)

`self.ncounts` : 以gram作为键，对应值表示以此gram为前缀的下一个gram的频次，及不同频次对应的出现次数。

`self._d`: 回退算法中的d。

`self._alpha`: 回退算法中的alpha

注意到，`self.ncounts`参数的计算是以不同前缀gram作为前提的。（感谢助教老师的提醒）



### 二、代码实现

具体的实现见以上函数，下面仅简要概述各个函数的实现思路。



##### 1. learn

`learn`函数第一部分实现的是词频统计。

在第二部分中，首先以下的函数段实现了按前缀和频次的排序。

```python
grams = itertools.groupby(
                   sorted(
                   sorted(
                   map(lambda itm: (itm[0][:-1], itm[1]),  # 去除最后一项后的键值对
                       self.frequencies[i].items()),
                       key=(lambda itm: itm[1])),
                       key=(lambda itm: itm[0])))  # 再按照第一位元素排序， 后面的值为频率
```

基于此，以gram作为前缀索引，即可统计不同频次的gram出现次数的值。



##### 2. _d

`_d`函数实现了Katz回退平滑算法中参数d的计算。调用`learn`函数中统计的`self.ncounts`，按公式计算即可。



##### 3. _alpha

`_alpha`函数实现了Katz回退平滑算法中参数α的计算。在实际的参数计算中，需要调用`__get_item__`函数以得到所取gram的对应概率。实际实现参照公式即可。



##### 4.  __get_item__

`__get_item__`函数实现了回退概率的计算。按照算法要求对输入gram进行分类，选择调用参数d和参数α计算概率，记录在`self.disfrequencies`中。



##### 5. log_prob

`log_prob`函数计算输入句子的概率对数。按照公式计算**log2(P(gram))**之和，取负除以cnt即可。



##### 6. ppl

`ppl`函数计算困惑度。同样按照公式计算即可。



### 三、结果

在测试集上，模型表现如下:

```python
Avg_of_probs = 4.00465082149174
Avg_of_ppls  = 74.96187670395679
```


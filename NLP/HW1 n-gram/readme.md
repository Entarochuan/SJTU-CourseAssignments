## N-Gram homework



报告在`ngram.ipynb`内。



#### Environment

No requirements



#### Run

`ngram.ipynb` 和 `ngram_h,py`都可以运行。在`saved_model`文件夹下存放了`.h`版本输出的模型，主文件夹下的`model.pkl`是运行`.ipynb`版本得到的模型。

在`ngram_h,py`中，可调的参数有action和eval_type，`eval_type='prob'`输出交叉熵结果，eval_type='ppl'输出PPL计算结果。



#### Result

```python
Avg_of_probs = 4.00465082149174
Avg_of_ppls  = 74.96187670395679
```


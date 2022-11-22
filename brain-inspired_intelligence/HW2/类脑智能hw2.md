## 类脑智能 

#### 520030910393 马逸川

### 

### 3.0 目录结构





### 3.1 模型描述

使用`Spikingjelly`工具包提供的函数搭建模型。下面将具体从神经元、拓扑结构、输入输出和所使用的学习方法四个方面介绍模型的结构。



#### 3.1.1 神经元

使用`spikingjelly.activation_based`中提供的`neuron.IFNode()`作为脉冲神经元。其中，定义神经元的参数为: `v_threshold=1.0, v_reset=0.0`。

参考文档，脉冲神经元的充放电、重置两类方程分别如下:

![image-20221121204743033](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221121204743033.png)

![image-20221121204809516](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221121204809516.png)

由于定义了`v_reset=0`，故实际中使用的是`hard`重置方式。

在文档中，神经元重置方式的实现如下:

```python
def neuronal_reset(self):
    if self.v_reset is None:
        self.v = self.v - self.spike * self.v_threshold
    else:
        self.v = (1. - self.spike) * self.v + self.spike * self.v_reset
```

在实际的网络结构中，在每一层卷积层后加上`neuron.IFNode()`层。



#### 3.1.2 拓扑结构

参考网络中主要处理特征的`self.conv`层说明，代码如下：

```python
self.conv = nn.Sequential(
    neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, 		surrogate_function=surrogate.ATan()),
    nn.MaxPool2d(2, 2),  # 14 * 14

    nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(8),
    neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
    nn.MaxPool2d(2, 2)  # 7 * 7
)
```

在模型中，主要用到了
import jittor
from pylab import *
# 中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import matplotlib.pyplot as plt


n = 1000

class Model(Module):

    def __init__(self, *args, **kw):
        self.layer1 = nn.Linear(1, 64)
        self.relu = nn.Relu()
        self.layer2 = nn.Linear(64, 1)

    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def get_data(n, batch_size): # generate random data for training test.
    for i in range(n):
        x = -1 + 2 * np.random.random((batch_size, 1))
        y = x * x
        yield jt.float32(x), jt.float32(y)

if __name__ =='__main__':

    model = Model()
    learning_rate = 0.1
    optim = nn.SGD(model.parameters(), learning_rate)

    for i, (x, y) in enumerate(get_data(1000, batch_size=50)):
        # print('tmp num = ', x, y)
        pred_y = model(x)
        # print(pred_y)
        loss = jt.sqr(pred_y - y)
        loss_mean = loss.mean()
        optim.step(loss_mean)

    print('train done, start testing')

    plt.figure()


    def get_test_data():  # generate random data for training test.
        test_feature = np.arange(-1, 1, 0.001).reshape(-1, 1)
        test_label = test_feature * test_feature
        test_feature = jt.float32(test_feature)
        test_label = jt.float32(test_label)
        yield jt.float32(test_feature), jt.float32(test_label)

    for i, (x, y) in enumerate(get_test_data()):
        y_pred = model(x)
        plt.plot(x.data, y.data, label='原曲线')
        plt.plot(x.data, y_pred.data, label='预测曲线')
        plt.legend(loc=0)
        plt.show()



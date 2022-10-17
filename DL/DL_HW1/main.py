import numpy as np
import paddle
from paddle import nn
from paddle.io import Dataset, BatchSampler, DataLoader
import matplotlib.pyplot as plt
import math


def my_Model():
    return nn.Sequential(
        nn.Linear(1, 2),
        nn.Sigmoid(),
        nn.Linear(2, 16),
        nn.Sigmoid(),
        nn.Linear(16, 1)
    )


class dataset(Dataset):

    def __init__(self, labels, features):
        super(dataset, self).__init__()
        self.labels = paddle.to_tensor(labels, dtype='float64')
        self.features = paddle.to_tensor(features, dtype='float64')
        self.num_samples = len(labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]
        return feature, label

    def __len__(self):
        return self.num_samples


def f(x):
    return x**3 / 1000


# 好像没有必要自己写，不过写了就不删了
def train(model, loss, updater, train_iter, test_iter, epochs=100, lr=0.1):

    for epoch in range(epochs):
        for X, y in train_iter:
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            updater.step()

    y_pred = []
    for X, y in test_iter:
        with paddle.no_grad():
            y_hat = model(X)
            y_pred.append(y_hat)

    return y_pred


def main():

    paddle.device.set_device('cpu')

    data = np.arange(-900, 100 , 1)
    data_train = paddle.to_tensor(data[:800])
    data_test = paddle.to_tensor(data[800:])

    label_train = paddle.to_tensor([f(x) for x in data_train])
    print(label_train)
    label_test = paddle.to_tensor([f(x) for x in data_test])
    print(type(label_test))

    model = my_Model()

    model = paddle.Model(model)

    model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.1, parameters=model.parameters()),
                  loss=paddle.nn.CrossEntropyLoss(reduction='mean'),
                  metrics=paddle.metric.Accuracy())

    # loss = paddle.nn.CrossEntropyLoss()
    # updater = paddle.optimizer.SGD(model.parameters(), 0.1)

    train_dataset = dataset(label_train, data_train)
    test_dataset = dataset(label_test, data_test)

    # 训练
    model.fit(train_dataset,
              epochs=10,
              batch_size=1,
              verbose=1)

    # 推理
    result_test = model.predict(test_dataset)
    data_test = np.array(data_test)
    shape = data_test.shape
    # print(shape)
    result_test = np.array([result_test[0][i][0][0] for i in range(len(result_test[0]))]).reshape(shape)
    # print(result_test.shape)
    label_test = np.array(label_test).reshape(shape)
    # print(label_test.shape)

    # train_iter = DataLoader(train_dataset, batch_size=64)
    # test_iter = DataLoader(test_dataset, batch_size=64)

    # y_pred = train(model, loss, updater, train_iter, test_iter)

    plt.figure()
    plt.plot(data_test, label_test)
    plt.plot(data_test, result_test)
    plt.show()

if __name__ == '__main__':
    main()

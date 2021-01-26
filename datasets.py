import pickle
import os
import numpy as np

lableName = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

def load_CIFAR_batch(filename):
    """
    cifar-10数据集是分batch存储的，这是载入单个batch
    @参数 filename: cifar文件名
    @r返回值: X, Y: cifar batch中的 data 和 labels
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """
    读取载入整个 CIFAR-10 数据集
    @参数 ROOT: 根目录名
    @return: X_train, Y_train: 训练集 data 和 labels
             X_test, Y_test: 测试集 data 和 labels
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return X_train, Y_train, X_test, Y_test


if __name__=='__main__':
    x_train, y_train, x_test, y_test = load_CIFAR10('data/cifar10/')
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)



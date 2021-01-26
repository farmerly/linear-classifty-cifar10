import numpy as np
import datasets
from linear_svm import LinearSVM


if __name__=='__main__':
    """
    随机梯度下降法
    Stochastic Gradient Descent
    """
    x_train, y_train, x_test, y_test = datasets.load_CIFAR10('data/cifar10/')
    ## 划分训练集，验证集，测试集
    num_train = 49000
    num_val = 1000
    num_test = 1000
    mask = range(num_train, num_train + num_val)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_train)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]
    # 去均值归一化
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    x_train = np.resize(x_train, (num_train, x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
    x_val = np.resize(x_val, (num_val, x_val.shape[1] * x_val.shape[2] * x_val.shape[3]))
    x_test = np.resize(x_test, (num_test, x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

    # 堆叠数组
    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

    svm = LinearSVM()
    loss_history = svm.train(x_train, y_train, learning_rate = 1e-7, reg = 2.5e4, num_iters = 2000, batch_size = 200, print_flag = True)

    y_train_pred = svm.predict(x_train)
    num_correct = np.sum(y_train_pred == y_train)
    accuracy = np.mean(y_train_pred == y_train)
    print('Training correct %d/%d: The accuracy is %f' % (num_correct.real, x_train.shape[0], accuracy.real))

    y_test_pred = svm.predict(x_test)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = np.mean(y_test_pred == y_test)
    print('Test correct %d/%d: The accuracy is %f' % (num_correct.real, x_test.shape[0], accuracy.real))

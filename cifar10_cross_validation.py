import numpy as np
import datasets
from linear_svm import LinearSVM

if __name__=='__main__':
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


    learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
    regularization_strengths = [8000.0, 9000.0, 10000.0, 11000.0, 18000.0, 19000.0, 20000.0, 21000.0]

    results = {}
    best_lr = None
    best_reg = None
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_svm = None  # The LinearSVM object that achieved the highest validation rate.

    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = LinearSVM()
            loss_history = svm.train(x_train, y_train, learning_rate=lr, reg=reg, num_iters=2000)
            y_train_pred = svm.predict(x_train)
            accuracy_train = np.mean(y_train_pred == y_train)
            y_val_pred = svm.predict(x_val)
            accuracy_val = np.mean(y_val_pred == y_val)
            if accuracy_val > best_val:
                best_lr = lr
                best_reg = reg
                best_val = accuracy_val
                best_svm = svm
            results[(lr, reg)] = accuracy_train, accuracy_val
            print('lr: %e reg: %e train accuracy: %f val accuracy: %f' %
                  (lr, reg, results[(lr, reg)][0].real, results[(lr, reg)][1].real))
    print('Best validation accuracy during cross-validation:\nlr = %e, reg = %e, best_val = %f' %
          (best_lr, best_reg, best_val))

    y_test_pred = best_svm.predict(x_test)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = np.mean(y_test_pred == y_test)
    print('Test correct %d/%d: The accuracy is %f' % (num_correct.real, x_test.shape[0], accuracy.real))
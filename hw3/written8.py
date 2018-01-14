import sys
import numpy as np
import matplotlib.pyplot as plt

import models
import utils

TRAIN_DATA = './hw3_train.dat.txt'
TEST_DATA = './hw3_test.dat.txt'
plot_mode = None

LR = 0.01
try:
    plot_mode = sys.argv[1]
    LR = float(sys.argv[2])
except:
    pass

def main():
    x, y = utils.load_data(TRAIN_DATA)
    y[y < 0] = 0

    x_out, y_out = utils.load_data(TEST_DATA)
    y_out[y_out < 0] = 0

    # GD, Q19
    
    gd = models.LogisticRegression()
    gd.fit(x, y, lr = LR, batch = -1, epoch = 2000, out_data=(x_out, y_out))
    
    y_pred = gd.predict(x_out)
    # print(count_err(y_out, y_pred))

    # SGD, Q20

    sgd = models.LogisticRegression()
    sgd.fit(x, y, lr = LR, batch = 1, epoch = 2000, out_data=(x_out, y_out))

    if plot_mode == 'q8':
        plt.title('8. Learning Rate: ' + str(LR))
        epoch = np.arange(2000)
        font ={'size': 16}
        plt.ylabel('E_in', **font)
        plt.xlabel('Iteration', **font)
        plt.plot(epoch, gd.log_in, label = 'GD')
        plt.plot(epoch, sgd.log_in, label = 'SGD')
        plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.show()
    elif plot_mode == 'q9':
        plt.title('9. Learning Rate: ' + str(LR))        
        epoch = np.arange(2000)
        font ={'size': 16}
        plt.ylabel('E_out', **font)
        plt.xlabel('Iteration', **font)
        plt.plot(epoch, gd.log_out, label = 'GD')
        plt.plot(epoch, sgd.log_out, label = 'SGD')
        plt.legend(loc='lower rigdht', shadow=True, fontsize='x-large')
        plt.show()

def count_err(y, y_pred):
    return np.sum(y != y_pred) / len(y)

if __name__ == '__main__':
    main()
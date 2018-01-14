import numpy as np
import utils

import models
import utils

TRAIN_PATH = './hw4_train.dat.txt'
TEST_PATH = './hw4_test.dat.txt'

def main():
    # coursera 13
    x, y = utils.load_data(TRAIN_PATH)
    model = models.LinearRegression()
    model.fit(x, y, l = 10)
    pred = model.predict_binary(x)

    print('cour 13 err rate:', utils.count_err(y, pred))

    # coursera 14
    for i in range(-10, 3):
        print(i)
        model.fit(x, y, l = 10**i)
        pred = model.predict_binary(x)
        
if __name__=='__main__':
    main()
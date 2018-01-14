import numpy as np
import re

def load_data(path):
    ls =[]
    with open(path) as raw:
        for line in raw:
            line = line.strip()
            data_list = re.split('[ ,\t, \n]', line)
            # data_list = line.split(' ')
            ls.append(data_list)
        data = np.array(ls)
        print(data[0])
        data = data[:, :3].astype(np.float)
    return data[:, :-1], data[:, -1]

def turn_neg_to_0(y):
    y[y<0] = -1
    return y

def shuffle(x, y):
    ran_ind = np.arange(len(x))
    np.random.shuffle(ran_ind)
    x = x[ran_ind]
    y = y[ran_ind]
    return x, y

def count_err(y, y_pred):
    return np.sum(y != y_pred) / len(y)

import numpy as np

def pla(x, y, epoch = 2000):
    x = np.insert(x, 0, 1, axis = 1)
    w = np.zeros(x.shape[1])
    count_update = 0
    # for run in range(epoch):
    for i in range(epoch):
        x, y = _shuffle(x, y)
        for j in range(x.shape[0]):
            hypothesis = sign(w, x[j])
            if  hypothesis != y[j]:
                w = w + y[j] * x[j]
                count_update += 1
        print('epoch', i+1, '\t | PLA update:', count_update)
        count_update = 0
    return w[0], w[1:]

def sign(w, x):
    # print('dot:', np.dot(w, x))
    hypothesis = float(np.dot(w, x))
    if hypothesis > 0:
        return 1
    else:
        return 0

def _shuffle(x, y):
    data = np.insert(x, x.shape[1], y, axis = 1)
    np.random.shuffle(data)
    return data[:, :x.shape[1]], data[:,-1]
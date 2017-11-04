import numpy as np

def pla(x, y, lr = 1, epoch = 2000, shuffle = True):
    x = np.insert(x, 0, 1, axis = 1)
    update_ls = []
    count_update = 0
    data_num = x.shape[0]
    for i in range(epoch):
        if shuffle:
            x, y = _shuffle(x, y)
        w = np.zeros(x.shape[1])
        count_correct = 0
        j = 0
        while count_correct < data_num:
            hypothesis = sign(w, x[j])
            if  hypothesis != y[j]:
                w = w + y[j] * x[j]
                count_update += 1
                count_correct = 0
            else:
                count_correct +=1
            j = (j+1) % data_num
        print('epoch', i+1, '\t | PLA update:', count_update)
        update_ls.append(count_update)
        count_update = 0
    update_avg = sum(update_ls) / epoch
    print(update_avg)
    return w[0], w[1:]

def pocket(x, y, epoch = 2000):
    x = np.insert(x, 0, 1, axis = 1)
    w = np.zeros(x.shape[1])
    num_data = x.shape[0]
    w_bst = w
    num_wrong_bst = num_data
    for i in range(epoch):
        k = i % num_data
        if y[k] != sign(w, x[k]):
            w = w + y[k] * x[k]
            num_wrong = 0
            for j in range(num_data):
                if y[j] != sign(w, x[j]):
                    num_wrong += 1
            if num_wrong <= num_wrong_bst:
                w_bst = w
                num_wrong_bst = num_wrong
    print(num_wrong_bst)
    return w_bst[0], w_bst[1:]
        
        
def sign(w, x):
    # print('w:', w, 'x', x)
    # print('dot:', np.dot(w, x))
    hypothesis = float(np.dot(w, x))
    if hypothesis > 0:
        return 1
    else:
        return -1

def _shuffle(x, y):
    data = np.insert(x, x.shape[1], y, axis = 1)
    np.random.shuffle(data)
    return data[:, :x.shape[1]], data[:,-1]

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

def pocket(x, y, update = 50, epoch = 2000, shuffle = True):
    # define value
    x = np.insert(x, 0, 1, axis = 1)
    w = np.zeros(x.shape[1])
    num_data = x.shape[0]

    # initilize
    w_bst = w
    w_bst_ls = []
    num_wrong_bst_ls = []
    for i in range(epoch):
        count_update = 0
        num_wrong_bst = num_data
        nxt = 0
        if shuffle:
            x, y = _shuffle(x, y)
        while count_update < update:
            k = nxt % num_data
            if y[k] != sign(w, x[k]):
                count_update += 1
                w = w + y[k] * x[k]
                num_wrong = 0
                for j in range(num_data):
                    if y[j] != sign(w, x[j]):
                        num_wrong += 1
                if num_wrong <= num_wrong_bst:
                    w_bst = w
                    num_wrong_bst = num_wrong
            nxt += 1
        print('epoch:', i+1, '\t | num. of wrong:', num_wrong_bst)        
        w_bst_ls.append(w_bst)
        num_wrong_bst_ls.append(num_wrong_bst)
    wrong_rate = sum(num_wrong_bst_ls) / (num_data * epoch)
    print('traing error rate:', wrong_rate)
    return w_bst[0], w_bst[1:]

def predict(x, b, w):
    result = np.dot(x, w)
    result = _sign_arr(result)
    return result
        
def sign(w, x):
    # print('w:', w, 'x', x)
    # print('dot:', np.dot(w, x))
    hypothesis = float(np.dot(w, x))
    if hypothesis > 0:
        return 1
    else:
        return -1

def _sign_arr(arr):
    arr = arr > 0
    arr = arr.astype(int)
    arr[arr == 0 ] = -1
    return arr


def _shuffle(x, y):
    data = np.insert(x, x.shape[1], y, axis = 1)
    np.random.shuffle(data)
    return data[:, :x.shape[1]], data[:,-1]

def err_rate(y, y_pred):
    num_data = y.shape[0]
    wrong_rate = sum(np.absolute(y - y_pred)) / (2 * num_data)    
    return wrong_rate
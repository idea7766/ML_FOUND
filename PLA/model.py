import numpy as np

def pla(x, y, update = 10000 , lr = 1, epoch = 2000, shuffle = True, x_val = None, y_val = None):
    x = np.insert(x, 0, 1, axis = 1)
    update_ls = []
    count_update = 0
    data_num = x.shape[0]
    val_err_ls = []
    for i in range(epoch):
        if shuffle:
            x, y = _shuffle(x, y)
        w = np.zeros(x.shape[1])
        count_correct = 0
        # up = 0
        j = 0
        while (count_correct < data_num) and (count_update < update):
            hypothesis = sign(w, x[j])
            if  hypothesis != y[j]:
                w = w + lr * y[j] * x[j]
                count_update += 1
                count_correct = 0
            else:
                count_correct +=1
            j = (j+1) % data_num
            # up += 1
        if x_val != None and y_val != None:
            y_val_pred = predict(x_val, w[0], w[1:]) 
            val_err = err_rate(y_val, y_val_pred)
            val_err_ls.append(val_err)
            print('epoch:', i+1, '\t | PLA update:', count_update, '\t | val_error rate:', val_err)
        else:
            print('epoch:', i+1, '\t | PLA update:', count_update)
        update_ls.append(count_update)
        count_update = 0
    if x_val != None and y_val != None:
        val_err_avg = sum(val_err_ls) / epoch
        print('val error:', val_err_avg)
    update_avg = sum(update_ls) / epoch
    print(update_avg)
    return w[0], w[1:]

def pocket(x, y, update = 50, epoch = 2000, shuffle = True, x_val = None, y_val = None):
    # define value
    x = np.insert(x, 0, 1, axis = 1)
    w0 = np.zeros(x.shape[1])
    num_data = x.shape[0]

    # initilize
    w_bst_ls = []
    err_ls = []
    val_err_ls =[]
    for i in range(epoch):
        w = w0
        w_bst = w
        err_bst = 1
        count_update = 0
        nxt = 0
        if shuffle:
            x, y = _shuffle(x, y)
        while count_update < update:
            k = nxt % num_data
            # w = w_bst
            if y[k] != sign(w, x[k]):
                count_update += 1
                w = w + y[k] * x[k]
                y_pred = predict(x[:, 1:], w[0], w[1:])
                err = err_rate(y, y_pred)
                if err <= err_bst:
                    w_bst = w
                    err_bst = err
            nxt += 1
        w_bst_ls.append(w_bst)
        err_ls.append(err_bst)

        if x_val != None and y_val != None:
            y_val_pred = predict(x_val, w_bst[0], w_bst[1:]) 
            val_err = err_rate(y_val, y_val_pred)
            val_err_ls.append(val_err)
            print('epoch:', i+1, '\t | error rate:', 
                    err_bst, '\t | val_error rate:', val_err)        
        else:
            print('epoch:', i+1, '\t | error rate:', err_bst)


    train_err_rate = sum(err_ls) / epoch
    val_err_rate = sum(val_err_ls) / epoch
    print('traing error rate:', train_err_rate)
    print('validation error rate:', val_err_rate)    
    return w_bst[0], w_bst[1:]

def predict(x, b, w):
    result = np.dot(x, w) + b
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
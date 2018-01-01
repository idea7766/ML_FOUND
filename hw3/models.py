import numpy as np
import utils

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, x, y):
        '''
        # 用於驗證的 close form
        ## Return
        b: bias
        w: weight array
        '''
        x = np.insert(x, 0, values = 1, axis = 1)
        x_trans = np.transpose(x)
        
        y_mat = np.transpose(np.mat(y))
        # y_mat = np.mat(y)  
        # print('y_mat.shape', y_mat.shape)
        x_mat = np.mat(x)
        # print('x_mat.shape', x_mat.shape)
        x_trans_mat = np.mat(x_trans)
        # print('x_trans_mat.shape', x_trans_mat.shape)

        w = np.linalg.inv(x_trans_mat * x_mat)  * x_trans_mat * y_mat
        w = np.array(w)
        w = w.flatten()
        # print(w)
        self.b = w[0]
        self.w = w[1:]
        return w[0], w[1:] #return b, w

    def predict_binary(self, x):
        b = self.b
        w = self.w
        # print(b,w)
        z = np.dot(x, w) + b
        y = np.sign(z)
        return y.flatten()

class LogisticRegression():
    def __init__(self):
        pass

    def fit(self, x, y, lr = 0.1, batch = 32, epoch = 10000, optimizer = 'static', shuffle = False, out_data=None):
        self.log_in = []
        self.log_out = []

        if out_data != None:
            x_out = out_data[0]
            y_out = out_data[1]
            
        num_data = x.shape[0]
        num_fea = x.shape[1]

        w = np.zeros(num_fea + 1)
        x = np.insert(x, 0, 1, axis = 1)

        sum_squ_grad = 0

        if shuffle == True:
            x, y = utils.shuffle(x, y)

        for i in range(epoch):
            if batch == -1:
                grad = self.gradient(x, y, w)
            elif batch >= 1:
                batch_ind = i * batch % num_data
                batch_end = (batch_ind + batch) % num_data
                if batch_ind >= batch_end:
                    x_bat = np.append(x[batch_ind : ], x[:batch_end], axis=0)
                    y_bat = np.append(y[batch_ind : ], y[:batch_end], axis=0)                    
                else:
                    x_bat = x[batch_ind : batch_end]
                    y_bat = y[batch_ind : batch_end]
                    grad = self.gradient(x_bat, y_bat, w)

            if optimizer == 'adagrad':
                ada, sum_squ_grad = self.adagrad(grad, sum_squ_grad)
                w = w - (lr / ada) * grad
            elif optimizer == 'static':
                w = w - lr * grad
            
            y_pred = self._predict(x[:, 1:], w[0], w[1:])            
            train_acc = self.acc(y, y_pred)
            self.log_in.append(self.err(y, y_pred))

            if out_data != None:
                y_out_pred = self._predict(x_out, w[0], w[1:])
                self.log_out.append(self.err(y_out, y_out_pred))
            print('>>> epoch : %d \t  | acc: %f' %(i + 1, train_acc)) # 改
        
        self.b = w[0]
        self.w = w[1:]
        return w[0], w[1:]

    def sigmoid(self, z):
        sig = 1 / (1 + np.exp(-z))
        return np.clip(sig, 1e-8, 1-(1e-8))

    def gradient(self, x, y, w):
        '''
        x: [1, x1, x2, ...]
        w: [bias, w1, w2, ...]
        '''
        num_data = x.shape[0]
        num_fea = x.shape[1]
        gradient_w = np.zeros(num_fea)

        z = np.dot(x, w)
        hypothesis = self.sigmoid(z)

        loss = hypothesis - y
        
        gradient_w = np.dot(np.transpose(x), loss) / num_data

        return gradient_w

    def adagrad(self, gradient, sum_squ_grad):
        sum_squ_grad += gradient ** 2
        adagrad = sum_squ_grad ** 0.5
        return adagrad, sum_squ_grad 

    def acc(self, y, y_pred):
        return np.sum(y == y_pred) / len(y)

    def err(self, y, y_pred):
        return np.sum(y != y_pred) / len(y)

    def _predict(self, x, b, w):
        z = np.dot(x, w) + b
        y = np.around(self.sigmoid(z))
        return y.flatten()

    def predict(self, x):
        w = self.w
        b = self.b
        z = np.dot(x, w) + b
        y = np.around(self.sigmoid(z))
        return y.flatten()
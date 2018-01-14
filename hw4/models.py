import numpy as np
import utils

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, x, y, l=0):
        '''
        # closed form
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

        w = np.linalg.inv(x_trans_mat * x_mat + l * np.identity(x_mat.shape[1]))  * x_trans_mat * y_mat
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
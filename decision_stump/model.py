import numpy as np

import utils as ut

def decision_stump(generator = ut.stocastic_data_gen):
    '''
    This function include gnerate stochastic data.
    '''
    run = 0
    e_in_ls = []
    e_out_ls = []
    err_ls = []

    for x, y in generator():
        num_data = len(x)
        theta = 0
        s = 0
        bst_err = num_data
        
        run += 1
        for x_theta in x:
            s = 1
            pred = np.sign(x - x_theta)
            pred[pred == 0] = 1
            num_err = np.sum(pred != y)
            if num_err >= num_data - num_err:
                s = -1
                num_err = num_data - num_err
            print(num_err)
            if num_err <= bst_err:
                bst_err = num_err
                theta = x_theta
        print('run', run, bst_err, s, theta)
        e_in = E_in(s, theta)

        e_in_ls.append(e_in)
        e_out_ls.append

def E_in(s, theta):
    return (s * np.absolute(theta) - s + 1) / 2

def E_out(lam = )
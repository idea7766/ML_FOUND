import numpy as np
import matplotlib.pyplot as plt

import utils as ut

def decision_stump(generator = ut.stocastic_data_gen):
    '''
    This function include gnerate stochastic data.
    '''
    NUM_RUN = 1000

    run = 0
    e_in_ls = []
    e_out_ls = []
    err_ls = []

    for x, y in generator(run = NUM_RUN):
        num_data = len(x)
        theta = 0
        bst_s = 0
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
            # print(num_err)
            if num_err <= bst_err:
                bst_s = s
                bst_err = num_err
                theta = x_theta
        e_in_ls.append(bst_err / num_data)
        e_out_ls.append(E_out(bst_s, theta))
        print('run', run, '\tE in:', bst_err / num_data, '\tE out', E_out(s, theta))
    print('Result', '\tE in:', sum(e_in_ls) / NUM_RUN, '\tE out', sum(e_out_ls) / NUM_RUN)

    plt.ylabel('E_out')
    plt.xlabel('E_in')
    plt.scatter(e_in_ls, e_out_ls, color = 'blue')
    # plt.plot([0, 0], [1, 1])
    plt.show()

def E_in(s, theta):
    return (s * np.absolute(theta) - s + 1) / 2

def E_out(s, theta):
    return (0.5 + 0.3 * s * (np.absolute(theta) - 1))
import numpy as np
import re

def load_data(path):
    ls =[]
    with open(path) as raw:
        for line in raw:
            data_list = re.split('[ ,\t, \n]', line)
            # data_list = line.split(' ')
            ls.append(data_list)
        data = np.array(ls)
        data = data[:, :5].astype(np.float)
    return data
    
# class record:
#     def __init__(self):
#         self.num_frequency = {}
#     def count(self, number):
        
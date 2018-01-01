import numpy as np

def stocastic_data_gen(min = -1, max = 1, num_data = 20, run = 1000, noise_rate = 0.2):
    for i in range(run):
        x = np.random.uniform(min, max, num_data)
        # x = np.sort(x)
        y = np.sign(x)
        y[y == 0] = 1
        prob = np.random.uniform(0 , 1 , num_data)
        y[prob >= 1-noise_rate ] *= -1
        yield x, y
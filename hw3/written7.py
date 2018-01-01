import numpy as np
import models
import matplotlib.pyplot as plt

def main():
    x1, x2, y = generate_data()
    x = fea_trans(x1, x2)
    model = models.LinearRegression()
    model.fit(x, y)

    err_ls = []
    b_total = 0
    w_total =np.zeros(5)
    for _ in range(1000):
        x1_out, x2_out ,y_out = generate_data()
        x_out = fea_trans(x1_out, x2_out)        
        y_pred = model.predict_binary(x_out)

        err_ls.append(count_err(y_out, y_pred))
    print('error rate:',sum(err_ls) / len(err_ls))
    
    plt.xlabel('Eout')
    plt.ylabel('Frequency')
    plt.title('7. Error Rate')
    plt.hist(err_ls)
    plt.grid(True) 
    plt.show()
    

def generate_data(min=-1, max =1, num_data =1000, noise_rate = 0.1):
    x1 = np.random.uniform(min, max, num_data)
    x2 = np.random.uniform(min, max, num_data)
    y = def_ans(x1, x2, noise_rate)
    return x1, x2, y

def def_ans(x1, x2, noise_rate):
    y = np.sign(np.square(x1) + np.square(x2) - 0.6)
    y[y == 0 ] = 1
    if noise_rate > 0:
        prob = np.random.uniform(0 , 1 , len(y))
        y[prob >= 1 - noise_rate] *= -1
    return y

def fea_trans(x1, x2):
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    new_x = np.concatenate((x1, x2, x1 * x2, np.square(x1), np.square(x2)), axis=1)
    # print(new_x[0])
    return new_x

def count_err(y, y_pred):
    return np.sum(y != y_pred) / len(y)

if __name__ == '__main__':
    main()
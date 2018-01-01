import utils
import model

# define PATH
DATA_PATH_18 = './hw1_18_train.dat'
DATA_PATH_18_test = './hw1_18_test.dat'

# load data
data = utils.load_data(DATA_PATH_18)
data_test = utils.load_data(DATA_PATH_18_test)

# distinguish x and y
x_train, y_train = data[:, :4], data[:, -1]
x_test, y_test = data_test[:, :4], data_test[:, -1]
# print(x_train)
# print(y_train)

b, w = model.pocket(x_train, y_train, epoch = 2000, shuffle = True, 
                    x_val = x_test, y_val = y_test)
                    
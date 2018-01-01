import utils
import model

# define PATH
DATA_PATH_8 = './hw1_8_train.dat'
DATA_PATH_18 = './hw1_18_train.dat'
DATA_PATH_18_test = './hw1_18_test.dat'

# load data
data_8 = utils.load_data(DATA_PATH_8)
data_18 = utils.load_data(DATA_PATH_18)

# distinguish x and y
x_8, y_8 = data_8[:, :4], data_8[:, -1]
x_18, y_18 = data_18[:, :4], data_18[:, -1]
print(x_18)
print(y_18)

# b, w = model.pla(x_train, y_train, epoch = 2000)
b, w = model.pocket(x_18, y_18, epoch = 2000)


print('w', w)
print('b', b)
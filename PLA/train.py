import utils
import model

DATA_PATH = './hw1_8_train.dat'

data = utils.load_data(DATA_PATH)
x_train, y_train = data[:, :4], data[:, -1]
# b, w = model.pla(x_train, y_train, epoch = 2000)
b, w = model.pocket(x_train, y_train, epoch = 2000)

# b, w = model.pla_train_到對(x_train, y_train)

print('w', w)
print('b', b)


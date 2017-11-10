import utils
import model

# define PATH
DATA_PATH_8 = './hw1_8_train.dat'

# load data
data_8 = utils.load_data(DATA_PATH_8)

# distinguish x and y
x_train, y_train = data_8[:, :4], data_8[:, -1]

b, w = model.pla(x_train, y_train, epoch = 2000, shuffle = True, hist_graph = True)
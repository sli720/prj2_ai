"""
This script is used for neuronal network hyperparameter search.

We assume that a good neuronal network architecture was already found (using 3.Find_architecture_neuronal_network.py)
After that the hyperparameters can be "bruteforced" using a the gridsearch approach.

The parameters which should be tested can be configured in this script in the variable "param_grid".
It's possible to specify value ranges for different parameters at the same time, however, gridsearch
will try every possible combination ==> If too much candidates are specified the runtime of this script can become
very very long.

We bruteforced just 2 differerent parameters at the same time, sometimes only one parameter when lots of possible candidates were available.

Please note: The results can further be optimized by using a RandomSearch algorithm after the GridSearch. However,
we didn't had enough computation power during the course so we just used the grid search results.

This script uses cross-validation with 3 foldes to detect the best settings. (input data is split into 3 sets; every parameter combination is tested 3 times everytime with another validation set from the 3)
"""


from keras.utils import to_categorical, np_utils
from keras.callbacks import *       # for TensorBoard
from keras.activations import *
from keras.models import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import *

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


from my_logging import get_logger
from dataset import *
from config import *
from visualization import *


logger = get_logger("Hyperparameter Search Neuronal_Network")


# =========== START CONFIG ================

min_max_normalization = True   # True... minMax ; False... Standard derivation normalization

# Batch size is closely connected to the number of epochs
# The bigger the batch size, the more memory the system must have and the faster one epoch can be calculated
# Example, batch_size of 128 => one epoch takes 74 seconds, batch size of 256 => 26 seconds one epoch; 
# batch size 512 => 20 seconds one epoch
# At the same time the val_acc after the first and second epoch for different batch sizes:
# batch size => first epoch val_acc | second epoch val_acc | third epoch val_acc | .. | epoch 30 val_acc (seconds per epoch)
# 128  => 0.8152 | 0.7354 | 0.8152 | .. | 0.8892 (73sec)
# 256  => 0.6598 | 0.7573 | 0.7720 | .. | 0.9770 (36sec)
# 256  => 0.7218 | 0.7765 | 0.8247 | .. | 0.9784 (36sec)    ... afeer epoch 50: 0.9837
# 512  => 0.6527 | 0.6562 | 0.7503 | .. | 0.9740 (20sec)
# 1024 => 0.5446 | 0.6571 | 0.7300 | .. | 0.9546 (11sec)    ... after epoch 50: 0.9742
# "It has been observed in practice that when using a larger 
# batch there is a significant degradation in the quality of 
# the model, as measured by its ability to generalize."
# source: https://arxiv.org/abs/1609.04836
# "In general, batch size of 32 is a good starting point, and you should also try with 64, 128, and 256."
# "Generally, the batch sizes of 32, 64, 128, 256, 512 and 1024 are used while training a neural network"
# Source: https://mydeeplearningnb.wordpress.com/2019/02/23/convnet-for-classification-of-cifar-10/
# "batch size should be a power of 2 to be most effective"
# source: https://arxiv.org/abs/1303.2314
batch_size = 512  # 256 seems to be a very good value, but it takes time; currently I use 512 to save time during param search (because it's just a little bit worse than 256 but double as fast; the final model uses 256 as batch size)
epochs = 10


# =========== END CONFIG ================


# Read the data
logger.info("Going to read train and test dataset...")
data_train = load_dataset(file_path_trainset, skip_rows = None, already_preprocessed = True)
data_test = load_dataset(file_path_testset, skip_rows = None, already_preprocessed = True)
logger.info("Finished reading train and test set")


# Split the train dataset into a validation and a train dataset
#data_train, data_validation = train_test_split(data_train, test_size=train_validationn_set_ration_for_cross_validation, stratify = data_train["Label"])


# Split data into network input (data_x) and network output (data_y)
data_train_y = data_train["Label"]
data_train_x = data_train.drop(columns=["Label",])

data_test_y = data_test["Label"]
data_test_x = data_test.drop(columns=["Label",])

#data_validation_y = data_validation["Label"]
#data_validation_x = data_validation.drop(columns=["Label",])


# Extract required information 
num_classes_train = data_train_y.nunique()  # number of distinct output (data_train_y) values; output values can be e.g.: BENIGN, Infilteration, sqli, ...
num_features_train = len(data_train_x.columns)

num_classes_test = data_test_y.nunique() 
num_features_test = len(data_test_x.columns)

#num_classes_validation = data_validation_y.nunique() 
#num_features_validation = len(data_validation_x.columns)

trainset_size = len(data_train)
#validationset_size = len(data_validation)
testset_size = len(data_test)


# Print data associated with the dataset
logger.info("Total number output classes  : {}".format(num_classes))
logger.info("Number output classes (train): {}".format(num_classes_train))
#logger.info("Number output classes (vali): {}".format(num_classes_validation))
logger.info("Number output classes (test): {}".format(num_classes_test))

logger.info("Total number input features  : {}".format(num_features))
logger.info("Number input features (train): {}".format(num_features_train))
#logger.info("Number input features (vali): {}".format(num_features_validation))
logger.info("Number input features (test): {}".format(num_features_test))

logger.info("Size of train dataset: {}".format(trainset_size))
#logger.info("Size of validation dataset: {}".format(validationset_size))
logger.info("Size of test dataset: {}".format(testset_size))




# One hot encoding
data_train_y = np_utils.to_categorical(data_train_y, num_classes)    # Replace output label number with one hot encoding
data_test_y = np_utils.to_categorical(data_test_y, num_classes)    # Replace output label number with one hot encoding
#data_validation_y = np_utils.to_categorical(data_validation_y, num_classes)    # Replace output label number with one hot encoding => Not required anymore because we replace the label around in preprocessing



def normalization_min_max(data_in_train, data_in_test):
    # Important: The MinMaxScaler already ensures that the divisor in scaling is not zero (otherwise columns will like "Bwd PSH Flag, "Fwd URG Flag", ... will become empty because of division by zero)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data_in_train)   # Fit just with train data (=> this calculates the min and max values for the .transform() call)

    # Apply the MinMax Scaler and convert back from a numpy array to a pandas dataframe object
    data_out_train = pd.DataFrame(scaler.transform(data_in_train.values), columns=data_in_train.columns, index=data_in_train.index)
    #data_out_validation = pd.DataFrame(scaler.transform(data_in_validation.values), columns=data_in_validation.columns, index=data_in_validation.index)
    data_out_test = pd.DataFrame(scaler.transform(data_in_test.values), columns=data_in_test.columns, index=data_in_test.index)

    return data_out_train, data_out_test


data_train_x, data_test_x = normalization_min_max(data_train_x, data_test_x)


# Start of the keras model
# The default values for the arguments represent the best parameters which we found
def create_model_neuronal_network(optimizer=Adam, lr=0.0001, init_mode='normal', activation="relu", number_neurons_per_layer=69):
    global num_features

    model = Sequential([
        Dense(number_neurons_per_layer, input_shape=(num_features,), kernel_initializer=init_mode),   
        BatchNormalization(),
        Activation(activation),

        Dense(number_neurons_per_layer, kernel_initializer=init_mode),                                
        BatchNormalization(),
        Activation(activation),

        Dense(number_neurons_per_layer, kernel_initializer=init_mode),                                
        BatchNormalization(),
        Activation(activation),

        Dense(number_neurons_per_layer, kernel_initializer=init_mode),                                
        BatchNormalization(),
        Activation(activation),

        Dense(num_classes),
        Activation('softmax'),
    ]) 


    optimizer = optimizer(lr=lr)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])

    model.summary()
    return model

model = KerasClassifier(
    build_fn=create_model_neuronal_network,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1)



# Parameter Grid for hyperparameter search with gradsearch

#lr_candidates = [0.0001, 0.0005, 0.00005]  #lr_candidates = [1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5]
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']  # especially [Adam, RMSprop]
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
optimizer_candidates = [Adam,]
lr_candidates = [0.0001,]
init_mode  = ['normal',]
activation = ["relu", ]
number_neurons_per_layer = [30, 69, 150, 200]


param_grid = {
    "optimizer": optimizer_candidates,
    "lr": lr_candidates,
    "init_mode" : init_mode,
    "activation" : activation,
    "number_neurons_per_layer" : number_neurons_per_layer
#    'class_weight':[{0: 500, 1: 1, 2: 1, 3:1 , 4:1 , 5:1 , 6:1 , 7:1 , 8:1}]
    }

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=1,
    verbose=1,
    cv=3)

grid_result = grid.fit(data_train_x, data_train_y)

# Summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

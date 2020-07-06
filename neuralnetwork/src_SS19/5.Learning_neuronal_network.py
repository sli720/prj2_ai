"""
This script builds on the results we gained from 
"3.Find_architecture_neuronal_network.py"
and
"4.Hyperparameter_search_neuronal_network.py".

Based on the identified architecture and the best hyperparameters this script learns the neuronal network.
We are using 850 epochs, a very high value which will not be reached because we are using early stopping.
This means that the script checks after every epoch if the new model is better than the previous ones.
If 50 epochs do not result in a better model the script stops the learning processes.
The best models are stored after each epoch (using the ModelCheckpoint callback).

Moreover, we are using weights to ensure that the IDS does not create false positives (it ensures that the
benign lavel is punished 500 times harder (with weight 500) if it was incorrectly predicted).

This scripts stores 3 files, one for the model architecture, one for the scaler to normalize input data and one
for the weights. This output must be used in the next script "6.Predict_neuronal_network.py"

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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


from sklearn.externals import joblib    # to save the scaler

from my_logging import get_logger
from dataset import *
from config import *
from visualization import *



logger = get_logger("Learning Neuronal_Network")


# =========== START CONFIG ================

min_max_normalization = True   # True... minMax ; False... Standard derivation normalization
batch_size = 256  # from hyperparamter search seemed to be the best value; 512 can also be used for testing because it's faster; Higher values will result in worse results, 512 is still ok.
epochs = 850
stop_after_so_many_epochs_without_improvement = 50
benign_label_weight = 500
# =========== END CONFIG ================


scaler = None

# Read the data
logger.info("Going to read train and test dataset...")
data_train = load_dataset(file_path_trainset, skip_rows = None, already_preprocessed = True)
data_test = load_dataset(file_path_testset, skip_rows = None, already_preprocessed = True)
logger.info("Finished reading train and test set")


# Split the train dataset into a validation and a train dataset
data_train, data_validation = train_test_split(data_train, test_size=train_validationn_set_ration_for_cross_validation, stratify = data_train["Label"])


# Split data into network input (data_x) and network output (data_y)
data_train_y = data_train["Label"]
data_train_x = data_train.drop(columns=["Label",])

data_test_y = data_test["Label"]
data_test_x = data_test.drop(columns=["Label",])

data_validation_y = data_validation["Label"]
data_validation_x = data_validation.drop(columns=["Label",])


# Extract required information 
num_classes_train = data_train_y.nunique()  # number of distinct output (data_train_y) values; output values can be e.g.: BENIGN, Infilteration, sqli, ...
num_features_train = len(data_train_x.columns)

num_classes_test = data_test_y.nunique() 
num_features_test = len(data_test_x.columns)

num_classes_validation = data_validation_y.nunique() 
num_features_validation = len(data_validation_x.columns)

trainset_size = len(data_train)
validationset_size = len(data_validation)
testset_size = len(data_test)


# Print data associated with the dataset
logger.info("Total number output classes  : {}".format(num_classes))
logger.info("Number output classes (train): {}".format(num_classes_train))
logger.info("Number output classes (vali): {}".format(num_classes_validation))
logger.info("Number output classes (test): {}".format(num_classes_test))

logger.info("Total number input features  : {}".format(num_features))
logger.info("Number input features (train): {}".format(num_features_train))
logger.info("Number input features (vali): {}".format(num_features_validation))
logger.info("Number input features (test): {}".format(num_features_test))

logger.info("Size of train dataset: {}".format(trainset_size))
logger.info("Size of validation dataset: {}".format(validationset_size))
logger.info("Size of test dataset: {}".format(testset_size))


# One hot encoding
data_train_y = np_utils.to_categorical(data_train_y, num_classes)    # Replace output label number with one hot encoding
data_test_y = np_utils.to_categorical(data_test_y, num_classes)    # Replace output label number with one hot encoding
data_validation_y = np_utils.to_categorical(data_validation_y, num_classes)    # Replace output label number with one hot encoding


def normalization_min_max(data_in_train, data_in_validation, data_in_test):
    global scaler
    # Important: The MinMaxScaler already ensures that the divisor in scaling is not zero (otherwise columns will like "Bwd PSH Flag, "Fwd URG Flag", ... will become empty because of division by zero)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data_in_train)   # Fit just with train data (=> this calculates the min and max values for the .transform() call)

    # Apply the MinMax Scaler and convert back from a numpy array to a pandas dataframe object
    data_out_train = pd.DataFrame(scaler.transform(data_in_train.values), columns=data_in_train.columns, index=data_in_train.index)
    data_out_validation = pd.DataFrame(scaler.transform(data_in_validation.values), columns=data_in_validation.columns, index=data_in_validation.index)
    data_out_test = pd.DataFrame(scaler.transform(data_in_test.values), columns=data_in_test.columns, index=data_in_test.index)

    return data_out_train, data_out_validation, data_out_test



data_train_x, data_validation_x, data_test_x = normalization_min_max(data_train_x, data_validation_x, data_test_x)


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



model = create_model_neuronal_network()



# save the scaler
joblib.dump(scaler, scaler_save_filename) 
logger.info("Saved scaler to disk: %s" % scaler_save_filename)

# Let's first save the model architecture as JSON
model_json = model.to_json()
with open(model_save_filename, "w") as json_file:
    json_file.write(model_json)
logger.info("Saved model architecture to disk: %s" % model_save_filename)
# The weights will later be written with ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=stop_after_so_many_epochs_without_improvement)
mc = ModelCheckpoint(weights_save_filename, monitor='val_loss', verbose=1, mode='min', save_best_only=True)


my_weights = {0: benign_label_weight, 1: 1, 2: 1, 3:1 , 4:1 , 5:1 , 6:1 , 7:1 , 8:1}

history = model.fit(
    x=data_train_x, 
    y=data_train_y, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=[data_validation_x, data_validation_y],
    class_weight=my_weights,
    callbacks=[es, mc]
    )



# Test the DNN
score = model.evaluate(data_test_x, data_test_y, batch_size=batch_size)
print("Test performance: ", score)


"""
if save_model_at_end:
    model_json = model.to_json()
    with open(model_save_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_save_filename)  # serialize weights to HDF5
    logger.info("Saved model to disk")

   


if visualize_at_end:
    # Plot accuracy
    plot_accuracy(history.history['acc'], history.history['val_acc'])

    # Plot a confusion matrix
    y_pred = model.predict(data_test_x)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(data_test_y, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    #plot_confusion_matrix(cm, classes=[c for c in range(num_classes)])
    plot_confusion_matrix(cm, classes=labels_to_use)
"""
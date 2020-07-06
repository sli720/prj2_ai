"""
This script assumes that a model was already created and saved by the script:
5.Learning_neuronal_network.py
or with:
3.Find_architecture_neuronal_network.py

It loads a dataset and calculates how good the network performs (score) and prints a confusion matrix

"""
from collections import Counter
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
from keras.models import model_from_json
#from sklearn.externals import joblib
import joblib
import sys
#from sklearn.externals import joblib
from my_logging import get_logger
from dataset import *
from config import *
from visualization import *




# ----------------------- Config start -----------------------
batch_size = 256
#UPDATE_SS20: Adapt paths
to_test_model_save_filename = "C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\NewProject\\model.json"
to_test_weights_save_filename = "C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\NewProject\\model.h5"
to_test_scaler_save_filename = "C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\NewProject\\scaler.save"

to_test_file_path_trainset = "C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\OldProject\\train_filtered.csv"
to_test_file_path_testset = "C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\OldProject\\test_filtered.csv"

# ----------------------- Config end -----------------------







logger = get_logger("Check performance Neuronal_Network")

# Load the model
json_file = open(to_test_model_save_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(to_test_weights_save_filename)

# Load the scaler
scaler = joblib.load(to_test_scaler_save_filename) 

# Load the new input data

logger.info("Going to read the test datasets...")
data_train = load_dataset(to_test_file_path_trainset, already_preprocessed = True)
data_test = load_dataset(to_test_file_path_testset, already_preprocessed = True)
logger.info("Finished reading train and test set")

# I also want to visualize a confusion matrix on the full input dataset (train + test set)
# Copying and merging the files like below is not very performant, however
# the runtime of this script is just 1-2 minutes and therefore performance is not that important
data_all = data_train.copy()
data_all = data_all.append(data_test)


# Split data into network input (data_x) and network output (data_y)
data_train_y = data_train["Label"]
data_train_x = data_train.drop(columns=["Label",])




data_test_y = data_test["Label"]
data_test_x = data_test.drop(columns=["Label",])

data_all_y = data_all["Label"]
data_all_x = data_all.drop(columns=["Label",])

data_train_y = np_utils.to_categorical(data_train_y, num_classes)    # Replace output label number with one hot encoding
data_test_y = np_utils.to_categorical(data_test_y, num_classes)    # Replace output label number with one hot encoding
data_all_y = np_utils.to_categorical(data_all_y, num_classes)    # Replace output label number with one hot encoding


# Normalize the data
data_train_x = pd.DataFrame(scaler.transform(data_train_x.values), columns=data_train_x.columns, index=data_train_x.index)
data_test_x = pd.DataFrame(scaler.transform(data_test_x.values), columns=data_test_x.columns, index=data_test_x.index)
data_all_x = pd.DataFrame(scaler.transform(data_all_x.values), columns=data_all_x.columns, index=data_all_x.index)



optimizer = Adam(lr=0.0001) 
loaded_model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])



# loaded_model.summary()

logger.info("Going to start evaluate on BOTH SETS MERGED (train+test) ...")
score = loaded_model.evaluate(data_all_x, data_all_y, batch_size=batch_size)
print("Fullset performance: ", score)

logger.info("Going to start plot a confusion matrix (this can take some time) ...")
y_pred = loaded_model.predict(data_all_x)
y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(data_all_y, axis=1)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=labels_to_use)

#UPDATE_SS20: Don't need confusion matrices for train and test set; commented it out
'''

for i in range(len(y_pred)):
        print(str(y_pred[i]) + " " + str(y_true[i]))




logger.info("Going to start evaluate on TRAIN SET ...")
score = loaded_model.evaluate(data_train_x, data_train_y, batch_size=batch_size)
print("Trainset performance: ", score)

logger.info("Going to start plot a confusion matrix (this can take some time) ...")
y_pred = loaded_model.predict(data_train_x)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(data_train_y, axis=1)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=labels_to_use)





logger.info("Going to start evaluate on TEST SET...")
score = loaded_model.evaluate(data_test_x, data_test_y, batch_size=batch_size)
print("Testset performance: ", score)


logger.info("Going to start plot a confusion matrix (this can take some time) ...")
y_pred = loaded_model.predict(data_test_x)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(data_test_y, axis=1)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=labels_to_use)
'''


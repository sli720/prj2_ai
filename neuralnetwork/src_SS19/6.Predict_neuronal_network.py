"""
This script assumes that a model was already created and saved by the script:
5.Learning_neuronal_network.py
or with:
3.Find_architecture_neuronal_network.py

It loads the saved model and predicts labels for new input data which must be passed via
argv[1]

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
from keras.models import model_from_json
from sklearn.externals import joblib
import sys



from my_logging import get_logger
from dataset import *
from config import *
from visualization import *



logger = get_logger("Predict with Neuronal_Network")

if len(sys.argv) == 2:
        file_path = sys.argv[1]
else:
        logger.warn("No argument passed")
        sys.exit(-1)


logger.info("Input filepath is: %s" % file_path)

# Load the model
json_file = open(model_save_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_save_filename)

# Load the scaler
scaler = joblib.load(scaler_save_filename) 

# Load the new input data
data_test_x = load_dataset(file_path, skip_rows = None, already_preprocessed = False, with_label = False) 
data_test_x = pd.DataFrame(scaler.transform(data_test_x.values), columns=data_test_x.columns, index=data_test_x.index)

optimizer = Adam(lr=0.0001) 
loaded_model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])


# loaded_model.summary()

logger.info("Going to start predict...")
y_pred = loaded_model.predict(data_test_x)
logger.info("Finished predict")
y_pred = np.argmax(y_pred, axis=1)      # convert one hot encoding back


indexes_of_attacks = np.where(y_pred != 0)
#attack_types = y_pred[indexes_of_attacks]

# This is not performant code because it iterates over every entry
# I assume that it can be implemented in a more performant way by
# using numpy code, however, currently it's fast enough for us
for x in indexes_of_attacks[0]:
        print(str(x) + "," + str(y_pred[x]))

"""
# Debugging code to obtain performance of the model
batch_size = 256
score = loaded_model.evaluate(data_test_x, data_test_y, batch_size=batch_size)
print("Test performance: ", score)
"""


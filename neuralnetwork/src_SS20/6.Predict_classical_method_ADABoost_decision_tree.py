"""
This script assumes that a model was already created and saved by the script:
4.hyperparameter_search_and_learning_classical_methods.py

It loads the saved model and predicts labels for new input data which must be passed via
argv[1]

"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import xgboost
import joblib
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from my_logging import get_logger
from dataset import *

# =========== START CONFIG ================
from config import file_path_trainset, file_path_testset, output_dir_path

# =========== END CONFIG ================
logger = get_logger("Predict with AdaBoost")


if len(sys.argv) == 2:
        file_path = sys.argv[1]
else:
        logger.warn("No argument passed")
        sys.exit(-1)


logger.info("Input filepath is: %s" % file_path)

data_test_x = load_dataset(file_path, skip_rows = None, already_preprocessed = False, with_label = False) 

logger.info("Loading best parameter set for Ada Boost from: " + output_dir_path + 'AdaBoostClassifier_best_parameter_set.pkl')
clf2 = load(os.path.join(output_dir_path, 'AdaBoostClassifier_best_parameter_set.pkl'))

logger.info("Finished loading, starting to predict")
y_pred = clf2.predict(data_test_x.values)

indexes_of_attacks = np.where(y_pred != 0)

# This is not performant code because it iterates over every entry
# I assume that it can be implemented in a more performant way by
# using numpy code, however, currently it's fast enough for us
for x in indexes_of_attacks[0]:
        print(str(x) + "," + str(y_pred[x]))


"""
# Debugging code to measure the performance of the model

accuracy = accuracy_score(data_test_y, y_pred)
logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
"""
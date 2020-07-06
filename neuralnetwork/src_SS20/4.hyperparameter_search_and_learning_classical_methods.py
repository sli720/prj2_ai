"""
This script implements the following classical learning methods:
    *) XGBoost
    *) AdaBoost with DecisionTreeClassifier
    *) RandomForest
    *) ExtraTrees
    *) Gradient Boosting
We mainly focused our tests on XGBoost and AdaBoost.
Our main tests were performed with the neuronal network, however, we used XGBoost and AdaBoost to get a comparision
with classic methods.

This script performs GridSearch to find good hyper parameters and saves the best models to a file.

Runtime: Several days on the full sets; Several hours (~24h) on the small sets
Especially the number of estimators have a strong impact on the runtime.
The parameters inside the create_model..() functions can be modified to test different hyperparameters.

Good results can easily be obtained by using AdaBoost with DecisionTreeClassifier and just 5 estimators (short runtime and good results)

"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost
from xgboost.sklearn import XGBClassifier
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

logger = get_logger("Ensembling_Methods")


# Read the data
logger.info("Going to read train and test dataset...")
data_train = load_dataset(file_path_trainset, skip_rows = None, already_preprocessed = True)
data_test = load_dataset(file_path_testset, skip_rows = None, already_preprocessed = True)
logger.info("Finished reading train and test set")

# Split data into network input (data_x) and network output (data_y)
data_train_y = data_train["Label"]
data_train_x = data_train.drop(columns=["Label",])

data_test_y = data_test["Label"]
data_test_x = data_test.drop(columns=["Label",])



# Extract required information 
num_classes = data_train_y.nunique()  # 15 in our case (full dataset)
num_features = len(data_train_x.columns)

# Check if the test set has a similar number of classes / features
num_classes_test = data_test_y.nunique() 
num_features_test = len(data_test_x.columns)
train_size = len(data_train)
test_size = len(data_test)

# Print data associated with the dataset
logger.info("Number output classes: {}".format(num_classes))
logger.info("Number input features: {}".format(num_features))
logger.info("Number output classes (test): {}".format(num_classes_test))
logger.info("Number input features (test): {}".format(num_features_test))
logger.info("Size of train dataset: {}".format(train_size))
logger.info("Size of test dataset: {}".format(test_size))

if(num_classes != num_classes_test or num_features != num_features_test):
    logger.error("Error, the number of features / classes is different in test and train set")
    sys.exit(-1)

    

def evaluate_model(model, model_name, parameters, data_train_x, data_train_y, data_test_x, data_test_y):
    clf = GridSearchCV(model, parameters, cv=5)
    logger.info("Going to start Gridsearch for: >{}<".format(model_name))
    clf.fit(data_train_x, data_train_y)
    logger.info("Gridsearch done")
    logger.info("Best score: %s" % (clf.best_score_))
    logger.info("Best parameter set: %s" % (clf.best_params_))
    save_path = os.path.join(output_dir_path, model_name + '_best_parameter_set.pkl')
    logger.info("Saving best parameter set to: " + save_path)
    joblib.dump(clf.best_estimator_, save_path)
    '''
    # Debugging code which can show the score of a model
    lr_score = model.score(data_test_x, data_test_y)
    lr_score_str = np.float64(lr_score).astype(str)

    logger.info('Model >{}< has score: {} '.format(model_name, lr_score_str))
    try:
        pass
        logger.info("Important features:")
        logger.info(model.feature_importances_)
    except:
        pass 
    '''

    #conf_mat = confusion_matrix(data_test_y, data_test_y_predictions)
    #print(conf_mat)




def create_model_ada_boost_classifier():
    from sklearn.ensemble import AdaBoostClassifier
    AB = Pipeline([('scaler', StandardScaler()), ('classifier', AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))]) # todo estimators 5
    ABparameters = {'classifier__learning_rate': [0.01, 0.03], 'classifier__n_estimators': [50, 200]}
    return (AB, "AdaBoostClassifier", ABparameters)

def create_model_random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    RF = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(criterion='entropy', bootstrap=True))])
    RFparameters = {
        'max_depth': [80, 90, 100, 110],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
        }
    return (RF, "Random Forest Classifier", RFparameters)

def create_model_extra_trees_classifier():
    from sklearn.ensemble import ExtraTreesClassifier
    ET = Pipeline([('scaler', StandardScaler()), ('classifier', ExtraTreesClassifier(n_estimators=25, criterion='gini', max_features='auto', bootstrap=False))])
    return (ET, "Extra Trees Classifier")

def create_model_gradient_boosting_classifier():
    from sklearn.ensemble import GradientBoostingClassifier
    GB = Pipeline([('scaler', StandardScaler()), ('classifier', GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=25, max_features='auto'))])
    return (GB, "Gradient Boosting Classifier")

def create_model_xgboost_classifier():
    XG = Pipeline([('scaler', StandardScaler()), ('classifier', XGBClassifier())])
    XGparameters = {'classifier__learning_rate': [0.01, 0.03], 'classifier__n_estimators': [50, 200]}
    return (XG, "XGBoostClassifier", XGparameters)



#models_to_test = [create_model_xgboost_classifier, create_model_ada_boost_classifier, create_model_random_forest_classifier, create_model_extra_trees_classifier, create_model_gradient_boosting_classifier]
models_to_test = [create_model_ada_boost_classifier, create_model_xgboost_classifier]

for model_function in models_to_test:
    model, model_name, parameters = model_function()
    evaluate_model(model, model_name, parameters, data_train_x, data_train_y, data_test_x, data_test_y)    

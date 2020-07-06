'''
SS20: This code should be used to filter a dataset and get only for defined labels.
'''

import sys
from my_logging import get_logger
from dataset import *
from config import *
from visualization import *

#Load train and test set
data_train = load_dataset("C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\OldProject\\train.csv", already_preprocessed = True)
data_test = load_dataset("C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\OldProject\\test.csv", already_preprocessed = True)

#Trainset: Select data by label and save the filtered train set
data_train_new = data_train[data_train['Label'].isin([0, 4, 5, 6])]
data_train_new.to_csv(file_path_trainset_full, index=False)

#Testset: Select data by label and save the filtered test set
data_test_new = data_test[data_test['Label'].isin([0, 4, 5, 6])]
data_test_new.to_csv(file_path_testset_full, index=False)
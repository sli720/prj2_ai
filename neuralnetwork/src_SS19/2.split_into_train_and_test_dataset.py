"""
This script can split the merged file from step 1 into two datasets, the train and test dataset.
Moreover, it stores smaller versions of these datasets because hyperparameter search (and testing) on the full
sets would take too long on our systems, so we create _small versions which contain all attacks
and just some benign traffic datapoints.
"""
import os
import os.path
import sys
from my_logging import get_logger
from sklearn.model_selection import train_test_split
from dataset import *
import gc
from config import *

logger = get_logger("create_train_and_test_set")

logger.info('Before loading')
full_dataset = load_dataset(file_path_merged_csv_file, None, True )
logger.info('After loading')
# print_dataset_statistics(full_dataset)

# Now start to split the full set into a train and test set
# The training and test set are split just one time to get constant results
# However, the real training and validation set are split at every execution 
# This other split is done in the neuronal network script
logger.info("Before split")
trainset_full, testset_full = train_test_split(full_dataset, test_size=train_test_set_ration, stratify = full_dataset["Label"])   #     # stratify => ensure that rations are applied to the label classes
logger.info("After split")

logger.info("Statistics for trainset_full dataset:")
print_dataset_statistics(trainset_full)

logger.info("Statistics for testset_full dataset:")
print_dataset_statistics(testset_full)

# Save the full datasets
logger.info("Going to write full dataset files...")
trainset_full.to_csv(file_path_trainset_full, index=False)
testset_full.to_csv(file_path_testset_full, index=False)
logger.info("Finished writing")

# Now save small versions of the sets

# Release memory from the two sets
trainset_full.drop(trainset_full.index, inplace=True)
del trainset_full

testset_full.drop(testset_full.index, inplace=True)
del testset_full

gc.collect()

# Now create a small testset where all attacks are present and some benign traffic is available
merged_datasets = None

for label_name in labels_attacks:
    label_number = label_number_assocation[label_name]
    subset_dataset = full_dataset.loc[full_dataset['Label'] == label_number]  # get all rows for the specific attack >label_name<
    if merged_datasets is None:
        merged_datasets = subset_dataset
    else:
        merged_datasets = merged_datasets.append(subset_dataset)

# Now merge some benign traffic into it
label_number = label_number_assocation["benign"]
subset_dataset_benign = full_dataset.loc[full_dataset['Label'] == label_number]
subset_dataset_benign_small = subset_dataset_benign.sample(n=number_benign_datapoints_for_small_set)    # get a subset of it

merged_datasets = merged_datasets.append(subset_dataset_benign_small)

trainset_small, testset_small = train_test_split(merged_datasets, test_size=train_test_set_ration,  stratify = merged_datasets["Label"])  #  stratify = ["Label"]

logger.info("Statistics for trainset_small dataset:")
print_dataset_statistics(trainset_small)

logger.info("Statistics for testset_small dataset:")
print_dataset_statistics(testset_small)

# Write the small datasets to files
logger.info("Going to write small dataset files...")
trainset_small.to_csv(file_path_trainset_small, index=False)
testset_small.to_csv(file_path_testset_small, index=False)
logger.info("Finished writing")
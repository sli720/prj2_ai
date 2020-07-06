"""
We are working with the dataset from: https://www.unb.ca/cic/datasets/ids-2018.html
The first step is to merge all CSV files into one file, remove duplicates and preprocess the data.
This is what this script is doing.
"""

from dataset import *
import gc
import os
import os.path
import sys

from config import *




# This are hardcoded values for the skip rows.
# This are CSV headers which are stored in the middle of a CSV file (because 2 CSV files were incorrectly merged)
# The number is the row number with the header and thus must be skipped
#UPDATE_SS20: The following lines are not needed anymore.
'''
hardcoded_skip_rows = {}
hardcoded_skip_rows["Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"] = [1000000]
hardcoded_skip_rows["Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"] = [414,19762,19907,39020,60810,76529,81060,85449,89954,91405,92658,95061,331113,331114,331115,331116,331117,331118,331119,331120,331121,331122,331123,331124,331125]
hardcoded_skip_rows["Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"] = []
hardcoded_skip_rows["Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"] = [21839,43118,63292,84014,107720,132410,154206,160207,202681,228584,247718,271677,296995,322939,344163,349510,355080,360661,366040,367414,368614,371160,377705,399544,420823,440997,461719,485425,510115,534074,559392,585336,606560]
'''


# It's more performant to use the above hardcoded skip_row values,
# but they can also be extracted dynamically using this function
def get_dynamic_skip_rows(filename, csv_dir_path):
    lowered_filename = filename.lower().replace("-", "_")[:-4]
    file_path = os.path.join(csv_dir_path, filename)

    skip_rows = []
    index = 0
    with open(file_path, "r") as fobj:
        while True:
            line = fobj.readline()
            if line: 
                if line.startswith("Dst Port,Protocol") and index != 0:
                    skip_rows.append(index)
            else:
                break
            index += 1

    return skip_rows
    


merged_datasets = None  # This variable will hold at the end all merged datapoints

for f in os.listdir(csv_dir_path):
    file_name, file_ext = os.path.splitext(f)
    if file_ext == '.csv':

        #UPDATE_SS20: Commented out the following lines
        '''
        if use_hardcoded_skip_rows:
            skip_rows = hardcoded_skip_rows[f]
        else:
            skip_rows = get_dynamic_skip_rows(f, csv_dir_path)
        '''
        file_path = os.path.join(csv_dir_path, f)
        current_dataset = load_dataset(file_path)

        if merged_datasets is None:
            merged_datasets = current_dataset
        else:
            logger.info('Going to start merge')
            merged_datasets = merged_datasets.append(current_dataset)
            logger.info('Finished merge')

            # Release memory from current_dataset
            current_dataset.drop(current_dataset.index, inplace=True)
            del current_dataset
            gc.collect()

            logger.info('Going to drop duplicates')
            merged_datasets.drop_duplicates(inplace=True)
            logger.info('Finished dropping duplicates')
        logger.info('merged_datasets has now a size of {}'.format(len(merged_datasets)))





logger.info('Dataset statistics before we drop inadequate label data:')
print_dataset_statistics(merged_datasets)


# Drop the rows with a label which we marked as inadequate label (e.g.: too less data for predictions or bad input data)
for lbl in labels_to_drop:
    lbl_number = label_number_assocation[lbl]       # converts the label name to the label number
    indexes = merged_datasets[ merged_datasets['Label'] == lbl_number].index
    merged_datasets.drop(indexes , inplace=True)


logger.info('Dataset statistics after we dropped inadequate label data:')
print_dataset_statistics(merged_datasets)

logger.info('Now going to drop duplicate entries with different labels...')
merged_datasets.drop_duplicates(subset=columns_without_label, inplace=True)

logger.info('Final dataset:')
print_dataset_statistics(merged_datasets)

logger.info('Going to write final dataset')
merged_datasets.to_csv(file_path_merged_csv_file, index=False)
logger.info('Finished')
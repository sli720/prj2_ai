"""
This code is used to load the dataset into memory.
This functionality is implemented in the function >load_dataset()<
"""

import pandas as pd
import numpy as np
import os
import os.path
import sys
from my_logging import get_logger
import gc

from config import *
 

logger = get_logger("Dataset.py")

# ==================================== DEFINITION OF ALL OUR .CSV FILES ===================================

datatypes_per_column = {
    "Dst Port" : np.uint16,             # values between 0 and 65535 
    "Protocol" : np.uint8,              # just 3 different values betwen 0 and 17
    "Flow Duration" : np.int64,         # values between -919011000000 and 120000000; TODO: How can a duration be negative ? 
    "Tot Fwd Pkts" : np.uint32,         # values between 1 and 309629
    "Tot Bwd Pkts" : np.uint32,         # values between 1 and 123118 
    "TotLen Fwd Pkts" : np.uint32,      # values between 1 and 144391846
    "TotLen Bwd Pkts" : np.uint32,      # values between 1 and 156360426
    "Fwd Pkt Len Max" : np.uint16,      # value between 0 and 64440
    "Fwd Pkt Len Min" : np.uint16,      # value between 0 and 1460
    "Fwd Pkt Len Mean" : np.float16,     # similar as above
    "Fwd Pkt Len Std" : np.float16,      # similar as above
    "Bwd Pkt Len Max" : np.uint16,      # similar as above
    "Bwd Pkt Len Min" : np.uint16,      # similar as above
    "Bwd Pkt Len Mean" : np.float16,     # similar as above
    "Bwd Pkt Len Std" : np.float16,      # similar as above

    "Flow Byts/s" : np.float32,          # value between 0 and 1806642857; => A converter is used, the datatype can't be specified therefore here
    "Flow Pkts/s" : np.float32,          # value between 0 and 6000000; => A converter is used, the datatype can't be specified therefore here
    "Flow IAT Mean" : np.float32,         # value between -828220000000 and 120000000; TODO : Why can this be negative ?!
    "Flow IAT Std" : np.float32,         # value between 0 and 474354474600
    "Flow IAT Max" : np.int64,          # value between -828220000000 and 968434000000; TODO : Why can this be negative ?!
    "Flow IAT Min" : np.int64,          # value between -947405000000 and 120000000; TODO : Why can this be negative ?!
    
    "Fwd IAT Tot" : np.int64,           # similar as above
    "Fwd IAT Mean" : np.float32,          # similar as above
    "Fwd IAT Std" : np.float32,           # similar as above
    "Fwd IAT Max" : np.int64,           # similar as above
    "Fwd IAT Min" : np.int64,           # similar as above

    "Bwd IAT Tot" : np.uint32,          # Strange the "Bwd" values are all positive (the Fwd not); so all in range between 0 and 120000000 => We can just uint32 instead of 64-bit datatype
    "Bwd IAT Mean" : np.float32,         # similar as above
    "Bwd IAT Std" : np.float32,          # similar as above
    "Bwd IAT Max" : np.uint32,          # similar as above
    "Bwd IAT Min" : np.uint32,          # similar as above
    
    "Fwd PSH Flags" : np.uint8,          # just 0 and 1
    # "Bwd PSH Flags"                   # Can be removed because they are always zero
    "Fwd URG Flags" : np.uint8,          # just 0 and 1
    # "Bwd URG Flags"                   # Can be removed because they are always zero

    "Fwd Header Len" : np.uint32,       # values between 0 and 2275036
    "Bwd Header Len" : np.uint32,       # values between 0 and 2462372
    "Fwd Pkts/s" : np.float32,           # values between 0 and 6000000
    "Bwd Pkts/s" : np.float32,           # values between 0 and 2000000

    "Pkt Len Min" : np.uint16,           # values between 0 and 1460
    "Pkt Len Max" : np.uint16,           # values between 0 and 65160
    "Pkt Len Mean" : np.float16,           # values between 0 and 17344
    "Pkt Len Std" : np.float16,           # values between 0 and 22788

    "Pkt Len Var" : np.float32,          # values between 0 and 519000000

    "FIN Flag Cnt" : np.uint8,          # just 0 and 1
    "SYN Flag Cnt" : np.uint8,          # just 0 and 1
    "RST Flag Cnt" : np.uint8,          # just 0 and 1
    "PSH Flag Cnt" : np.uint8,          # just 0 and 1
    "ACK Flag Cnt" : np.uint8,          # just 0 and 1
    "URG Flag Cnt" : np.uint8,          # just 0 and 1

    # "CWE Flag Count"                  # Can be removed because they are always zero

    "ECE Flag Cnt" : np.uint8,          # just 0 and 1

    "Down/Up Ratio" : np.uint16,        # values between 0 and 311
    "Pkt Size Avg" : np.float16,        # values between 0 and 17478
    "Fwd Seg Size Avg" : np.float16,        # values between 0 and 16529
    "Bwd Seg Size Avg" : np.float16,        # values between 0 and 33879

    # "Fwd Byts/b Avg"                  # Can be removed because they are always zero
    # "Fwd Pkts/b Avg"                  # Can be removed because they are always zero
    # "Fwd Blk Rate Avg"                  # Can be removed because they are always zero
    # "Bwd Byts/b Avg"                  # Can be removed because they are always zero
    # "Bwd Pkts/b Avg"                  # Can be removed because they are always zero
    # "Bwd Blk Rate Avg"                  # Can be removed because they are always zero

    "Subflow Fwd Pkts" : np.uint32,     # values between 1 and 309629
    "Subflow Fwd Byts" : np.uint32,     # values between 1 and 144391846
    "Subflow Bwd Pkts" : np.uint32,     # values between 0 and 123118
    "Subflow Bwd Byts" : np.uint32,     # values between 0 and 156360426
    
    "Init Fwd Win Byts" : np.int32,     # values between -1 and 65535 ; TODO: What does -1 mean in this context?
    "Init Bwd Win Byts" : np.int32,     # values between -1 and 65535 ; TODO: What does -1 mean in this context?

    "Fwd Act Data Pkts" : np.uint32,     # values between 0 and 309628
    "Fwd Act Data Pkts" : np.uint8,     # values between 0 and 56
    "Active Mean" : np.float32,          # values between 0 and 113269143
    "Active Std" : np.float32,           # similar as above
    "Active Max" : np.uint32,           # similar as above
    "Active Min" : np.uint32,           # similar as above
    "Idle Mean" : np.float32,            # values between 0 and 395571421052
    "Idle Std" : np.float32,             # values between 0 and 262247866338
    "Idle Max" : np.uint64,             # values between 0 and 968434000000
    "Idle Min" : np.uint64,             # values between 0 and 239934000000

    "Label" : np.uint8               # ==> I can't define the datatype of label here because we overwrite it with a converter
}

# Since the input files are pretty big we just want to read the absolut requiered fields into memory
# The timestamp is not helpful in our network, and since it's a string, it would consume lots of memory
# I therefore avoid reading it
#columns_to_read = ["Dst Port","Protocol","Flow Duration","Label"]
columns_to_read = ["Dst Port",
"Protocol",
# "Timestamp",      # not required for the NN; we don't want to learn that attacks start at specific times
"Flow Duration",
"Tot Fwd Pkts",
"Tot Bwd Pkts",
"TotLen Fwd Pkts",
"TotLen Bwd Pkts",
"Fwd Pkt Len Max",
"Fwd Pkt Len Min",
"Fwd Pkt Len Mean",
"Fwd Pkt Len Std",
"Bwd Pkt Len Max",
"Bwd Pkt Len Min",
"Bwd Pkt Len Mean",
"Bwd Pkt Len Std",
"Flow Byts/s",
"Flow Pkts/s",
"Flow IAT Mean",
"Flow IAT Std",
"Flow IAT Max",
"Flow IAT Min",
"Fwd IAT Tot",
"Fwd IAT Mean",
"Fwd IAT Std",
"Fwd IAT Max",
"Fwd IAT Min",
"Bwd IAT Tot",
"Bwd IAT Mean",
"Bwd IAT Std",
"Bwd IAT Max",
"Bwd IAT Min",
"Fwd PSH Flags",
# "Bwd PSH Flags",      # Can be removed because they are always zero
"Fwd URG Flags",
# "Bwd URG Flags",      # Can be removed because they are always zero
"Fwd Header Len",
"Bwd Header Len",
"Fwd Pkts/s",
"Bwd Pkts/s",
"Pkt Len Min",
"Pkt Len Max",
"Pkt Len Mean",
"Pkt Len Std",
"Pkt Len Var",
"FIN Flag Cnt",
"SYN Flag Cnt",
"RST Flag Cnt",
"PSH Flag Cnt",
"ACK Flag Cnt",
"URG Flag Cnt",
# "CWE Flag Count",       # Can be removed because they are always zero
"ECE Flag Cnt",
"Down/Up Ratio",
"Pkt Size Avg",
"Fwd Seg Size Avg",
"Bwd Seg Size Avg",
# "Fwd Byts/b Avg",         # Can be removed because they are always zero
# "Fwd Pkts/b Avg",         # Can be removed because they are always zero
# "Fwd Blk Rate Avg",       # Can be removed because they are always zero
# "Bwd Byts/b Avg",         # Can be removed because they are always zero
# "Bwd Pkts/b Avg",         # Can be removed because they are always zero
# "Bwd Blk Rate Avg",       # Can be removed because they are always zero
"Subflow Fwd Pkts",
"Subflow Fwd Byts",
"Subflow Bwd Pkts",
"Subflow Bwd Byts",
"Init Fwd Win Byts",
"Init Bwd Win Byts",
"Fwd Act Data Pkts",
"Fwd Seg Size Min",
"Active Mean",
"Active Std",
"Active Max",
"Active Min",
"Idle Mean",
"Idle Std",
"Idle Max",
"Idle Min",
"Label"]


columns_without_label = columns_to_read[:]
columns_without_label.remove("Label")

# ===================================== Start Converters =====================================


def fix_column_infinity_and_nan_values(val):
    # The .csv file columns are calculated with the following formular
    # "Flow Byts/s" :=  "TotLen Fwd Pkts" / ("Flow Duration" / 1000000.0)
    # => Some datapoints (rows) have a "Flow Duration" of zero. 
    # ==> The result will be that "Flow Byts/s" will be "infinity" because of divison by zero
    # If we keep "infinity" as string as a value we can't do math operations with the column
    # And it wasts a lot of space because the column datatype would be "object" instead of "floatXX"
    # Same applies for the "Flow Pkts/s" column
    lowered = val.lower() 
    if lowered == "infinity" or lowered == "nan":
        return float(0)
    return float(val)   


def convert_label_names_to_numbers(label_name):
    label_name = label_name.lower()
    if label_name not in label_number_assocation:
        print("TODO implement label number: %s" % label_name)
        sys.exit(-1)
    return np.int32(label_number_assocation[label_name])



def get_indexes_with_headers_which_must_be_skipped(dataset):
    indexes = dataset[ dataset['Label'] == "Label" ].index  # Get indexes for which Label column has the value "Label" ...
    tmp = ""
    tmp += "["
    for index in indexes:
        tmp += "%s+1," % str(index)
    tmp = tmp[:-1]  # remove last ","
    tmp += "]"
    print(tmp)
    sys.exit(-1)





# ===================================== Start Main Code to load the dataset =====================================

def load_dataset(file_path, skip_rows, already_preprocessed = False, with_label = True):
    global datatypes_per_column, columns_to_read
    logger.info('Going to start loading: {}'.format(file_path))
    
    if with_label:
        columns_we_read = columns_to_read
    else:
        columns_we_read = columns_without_label

    # Skip rows is the row from excel -1 (-1 because the header)
    # If we execute this:
    # indexes = dataset[ dataset['Label'] == "Label" ].index  # Get indexes for which Label column has the value "Label" ...
    # print(indexes)
    # it's the shown index +1
    # The input csv files contain rows with again the csv header! In such a case the Label value will be "Label"
    # ==> Remove those lines
    # Link which explains how to do this: https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
    # At first I dropped them with the above code and then:
    # dataset.drop(indexes, inplace=True)   # ... and delete these rows
    # However, this would still consume lots of memory because all columns will be loaded as "Object"
    # I therefore specify in skiprows which rows are csv-headers to skip them

    if already_preprocessed == False:
        dataset = pd.read_csv(  file_path, 
                                sep=',',                     # sep='\s*,\s*'  ==> We don't use a regex here because that would fallback to python implementation which is slow!
                                skipinitialspace=True, 
                                skiprows=skip_rows, 
                                usecols=columns_we_read,
                                converters={"Flow Byts/s" : fix_column_infinity_and_nan_values,
                                            "Flow Pkts/s" : fix_column_infinity_and_nan_values,
                                            "Label" : convert_label_names_to_numbers
                                            },
                                dtype = datatypes_per_column 
                                )
        # Change the data type to the correct one for the columns where a converter was used
        if with_label == False:
            dataset = dataset.astype({"Flow Byts/s" : np.float32, "Flow Pkts/s" : np.float32})
        else:
            dataset = dataset.astype({"Flow Byts/s" : np.float32, "Flow Pkts/s" : np.float32, "Label" : np.uint8})
    else:
        # Converters are not required and all datatypes will be set via "dtype" and must not be corrected afterwards
        dataset = pd.read_csv(  file_path, 
                                sep=',',                     # sep='\s*,\s*'  ==> We don't use a regex here because that would fallback to python implementation which is slow!
                                skipinitialspace=True, 
                                skiprows=skip_rows, 
                                usecols=columns_we_read,
                                dtype = datatypes_per_column 
                                )
    logger.info('Finished loading: {}'.format(file_path))

    if already_preprocessed == False:
        logger.info('Going to drop duplicates: {}'.format(file_path))
        logger.info('Number of datapoints before dropping duplicates: {}'.format(len(dataset)))
        dataset.drop_duplicates(inplace=True)
        logger.info('Finished dropping duplicates: {}'.format(file_path))

    logger.info('Loaded dataset has {} datapoints'.format(len(dataset)))

    return dataset



def print_dataset_statistics(dataset, display_value_ranges = False, display_head = False, memory_usage = False):
    print("\nNumber of rows: {}".format(len(dataset)))

    print("\nNumber of label occurences:")
    print(dataset.groupby('Label').size())

    if memory_usage:
        print("\nMemory usage:")
        dataset.info(memory_usage='deep')

    if display_value_ranges:
        print("\nColumn value ranges:")
        for column_name in dataset:
            max_idx = dataset[column_name].idxmax()
            max_val = dataset.loc[max_idx][column_name]
            min_idx = dataset[column_name].idxmin()
            min_val = dataset.loc[min_idx][column_name]
            number_different_unique_values_in_column = dataset[column_name].nunique()
            print("%s\t\t%d - %d (%d)" % (column_name.ljust(16), min_val, max_val, number_different_unique_values_in_column))
    if display_head:
        print("\nHead:")
        print(dataset.head())
    print("\n")



if __name__ == "__main__":
    pass

"""
Configuration file
Most values should not be changed, however, the paths can (and should) be adapted.
"""

import os



# When we start to merge / process the initial CSV files, should we use hardcoded (precomputed)
# skip rows? True... would be faster, however, that just works with exactly the same input data
# Use False when processing new input data
use_hardcoded_skip_rows = True





train_test_set_ration = 0.2     # 0.2 means 80 % train data, 20% test data
train_validationn_set_ration_for_cross_validation = 0.2 # 0.2 means 80% train data and 20% validation data
number_benign_datapoints_for_small_set = 574615     # the small datasets (which we use to find a good network / hyperparameters) contain all attacks and some benign traffic




# Instead of strings for the labels we keep numbers in-memory to make the dataset in-memory smaller
# For this we use a converter which replaces the labels during reading
# Important: keys must be stored in lower case
label_number_assocation = {}
label_number_assocation["benign"] = 0
label_number_assocation["ddos attacks-loic-http"] = 1
label_number_assocation["ddos attack-hoic"] = 2
label_number_assocation["dos attacks-hulk"] = 3
label_number_assocation["bot"] = 4
label_number_assocation["ssh-bruteforce"] = 5
label_number_assocation["dos attacks-goldeneye"] = 6        # 40668 occurences
label_number_assocation["dos attacks-slowloris"] = 7        # just 9905 occurences !
label_number_assocation["ddos attack-loic-udp"] = 8         # just 1378 occurences !


# The following are labels which we drop!

# we have enough datapoints for this attack, however, the features can't be used to detect it! 
# If we would like to keep it, we would need to extract other features. For now we just drop 
# this attack because all models currently just 50:50 randomly predict this attack
label_number_assocation["infilteration"] = 9     

# brute force -web          	       553
# brute force -xss          	       228
# sql injection           	           84
# dos attacks-slowhttptest             55
# ftp-bruteforce          	           54
label_number_assocation["brute force -web"] = 10
label_number_assocation["brute force -xss"] = 11
label_number_assocation["sql injection"] = 12
label_number_assocation["dos attacks-slowhttptest"] = 13
label_number_assocation["ftp-bruteforce"] = 14

labels_to_drop = ["infilteration", "brute force -web", "brute force -xss", "sql injection", "dos attacks-slowhttptest", "ftp-bruteforce"]
labels_to_use = ["benign", "ddos attacks-loic-http", "ddos attack-hoic", "dos attacks-hulk", "bot", "ssh-bruteforce", "dos attacks-goldeneye", "dos attacks-slowloris", "ddos attack-loic-udp"]  # this is just for confusion_matrix visualization; note that it should be the same order as >label_number_assocation<



labels_attacks = [] # Label names of all attacks which we cover
for label_name in label_number_assocation:
    if label_name not in labels_to_drop:
        if label_name != "benign":
            labels_attacks.append(label_name)


num_classes_to_drop = len(labels_to_drop)
num_classes = len(label_number_assocation) - num_classes_to_drop    # should be 15-6= 9 
num_features = 69


scaler_save_filename = "scaler.save"
model_save_filename = "model.json"
weights_save_filename = "model.h5"




file_path_merged_csv_file = "merged.csv"
file_path_trainset_full = "train.csv"
file_path_testset_full = "test.csv"
file_path_trainset_small = "train_small.csv"
file_path_testset_small = "test_small.csv"
    



currentSystem = "Server"    # "Server" ... our server, "PC1" ... rene laptop, "PC2" ... rene pc

if currentSystem == "Server":    # paths on the server (Linux)
    csv_dir_path = "/opt/ai-project/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
    output_dir_path = "/opt/ai-project/output/"
    log_dir = "/opt/ai-project/output/logs/"
elif currentSystem == "fredelaptop":    # paths on the server (Linux)
    csv_dir_path = "/opt/ai-project/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
    output_dir_path = "C:/Studium/Technikum/Semester 2/prj2/Daten/Out/"
    log_dir = "C:/Studium/Technikum/Semester 2/prj2/Daten/Out/logs/"
elif currentSystem == "PC1":   # Local paths from rene (Windows)
    csv_dir_path = "D:\\dataset2018\\Processed Traffic Data for ML Algorithms"
    output_dir_path = "D:\\dataset2018\\"
    log_dir = "D:\\technikum\\semester2\\Proj2\\Datasets\\logs\\"
elif currentSystem == "PC2": 
    csv_dir_path = "M:\\dataset2018\\Processed Traffic Data for ML Algorithms"
    output_dir_path = "M:\\dataset2018\\"
    log_dir = "C:\\NeuronalNetwork\\logs\\"
else:
    print("Wrong system, exiting...")
    sys.exit(-1)



scaler_save_filename = os.path.join(output_dir_path, scaler_save_filename)
model_save_filename = os.path.join(output_dir_path, model_save_filename)
weights_save_filename = os.path.join(output_dir_path, weights_save_filename)

file_path_merged_csv_file = os.path.join(output_dir_path, file_path_merged_csv_file)
file_path_trainset_full = os.path.join(output_dir_path, file_path_trainset_full)
file_path_testset_full = os.path.join(output_dir_path, file_path_testset_full)
file_path_trainset_small = os.path.join(output_dir_path, file_path_trainset_small)
file_path_testset_small = os.path.join(output_dir_path, file_path_testset_small)

# Configure the current set as either the _small or _full one
# This are the sets which are really used during training
#file_path_trainset = file_path_trainset_small
#file_path_testset = file_path_testset_small
file_path_trainset = file_path_trainset_full
file_path_testset = file_path_testset_full
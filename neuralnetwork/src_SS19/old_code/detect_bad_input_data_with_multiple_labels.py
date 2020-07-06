from dataset import *
from config import *


current_dataset = load_dataset(file_path_merged_csv_file, None, True )
# print_dataset_statistics(current_dataset)

#duplicateRowsDF = current_dataset[current_dataset.duplicated(columns_without_label, keep=False)]
#print("duplicateRowsDF length: %d" % len(duplicateRowsDF))

print_dataset_statistics(current_dataset)

current_dataset.drop_duplicates(subset=columns_without_label, inplace=True)

print("After:")
#duplicateRowsDF = current_dataset.duplicated(columns_without_label, keep=False).index
#print(duplicateRowsDF)

#indexes = merged_datasets[ merged_datasets['Label'] == lbl_number].index


# duplicateRowsDF.to_csv("D:\\dataset2018\\duplicates2.csv", index=False)


print_dataset_statistics(current_dataset)
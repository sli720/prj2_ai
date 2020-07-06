ReadMe

How to install Anaconda with a terminal:

1. Download the current version: 
	wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
2. Run: 
	bash Anaconda3-2019.03-Linux-x86_64.sh
3. Create a Pyton 3 Anaconda environment with:
	conda create --name my_env python=3
4. Activate it with: 
	conda activate my_env
5. Install required packages: 
	conda install -c anaconda package_name
	
Packages we used:
	- keras
	- pandas
	- tensorflow
	- scikit-learn
	- py-xgboost / py-xgboost-cpu
	- matplotlib
	- joblib
	
On a machine with a GUI you can use Anaconda Navigator to create an enviroment, install required packages and start Visual Studio Code from there to start developing.

How to use our project:
1. Use 1.merge_all_csv_files.py to merge all .csv files from the Dateset. The location is defined in config.py.

2. Use split_into_train_and_test_dataset.py to create a training and a test set according to the ratio set in config.py.

3. Find the optimal architecture for the neuronal network with 3.Find_architecture_neuronal_network.py

4. Use the hyperparameter search to find and save the best parameters from the parameters given in each model.

5. Train the neuronal network to recognize malicious traffic.

6. Use 6.Predict_XXX.py with a .csv file to get a prediction whether there was malicious traffic or not.

Models for each method can be found in the folders which are named accordingly.
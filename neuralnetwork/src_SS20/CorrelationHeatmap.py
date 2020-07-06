'''
SS20: This source code is from https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3 .
It is used to generate a heatmap and display the correlation between (many) variables.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
data = pd.read_csv('C:\\Users\mwech\\PycharmProjects\\AISecSS20\\Daten\\NewProject\\merged.csv', index_col=0)

#Specifiy by index which features should be involved
data = data.iloc[:,0:35]

#Calculate the correlations
corr = data.corr()

#Intialize the plot
fig = plt.figure()

#The following commands specify how the plot should look
ax = fig.add_subplot(111)
#Create a color bar to display correlation between -1 and 1
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

#Set axis ticks for each axis
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)

#To avoid cutting off some labels specify the following command
plt.tight_layout()

#Display the plot
plt.show()

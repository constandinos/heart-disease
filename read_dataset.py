# read_dataset.py
# Created by: Constandinos Demetriou, 2021

import pandas as pd
from sklearn.utils import shuffle

#Set the names of features
column_names = ['age', 'sex', 'cp',	'trestbps',	'chol',	'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', \
                'ca', 'thal', 'target']

#Read data
df = pd.read_csv('data/processed.cleveland.data', sep=',', header=None)
df.columns = column_names

#Remove rows with missing data
for i in column_names:
    df = df[df[i] != '?']

#Change data type
df = df.astype({'age':int, 'sex':int, 'cp':int, 'trestbps':int, 'chol':int, 'fbs':int, 'restecg':int, 'thalach':int, \
                'exang':int, 'oldpeak':float, 'slope':int, 'ca':float, 'thal':float, 'target':int})
df = df.astype({'ca':int, 'thal':int})

#Process data
df.loc[df['target'] >= 1, 'target'] = 1
df['cp'] = df['cp'] - 1
df['slope'] = df['slope'] - 1
df.loc[df['thal'] == 3, 'thal'] = 0
df.loc[df['thal'] == 6, 'thal'] = 1
df.loc[df['thal'] == 7, 'thal'] = 2

#Random shuffle rows
df = shuffle(df)

#Write data to a csv file
df.to_csv('data/dataset2.csv', index=False)

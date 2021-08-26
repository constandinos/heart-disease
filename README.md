Predict heart disease
=====================
In this project, we implement a system that predicts if a patient has a heart disease.

Preamble
--------
Author: Constandinos Demetriou

Copyright (c) 2021

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/

Introduction
------------
Over the past decade, heart disease, also known as cardiovascular disease, has remained the leading cause of death worldwide. An estimate from the World Health Organization is that more than 17.9 million deaths occur each year worldwide due to cardiovascular disease and of these deaths, 80% are due to coronary heart disease and stroke. Common risk factors are smoking, excessive alcohol and caffeine consumption, stress and lack of exercise along with other factors such as obesity, hypertension, high  cholesterol are predisposing factors for heart disease. Effective and timely medical diagnosis of heart disease plays a crucial role in preventing death.

Machine learning is one of the fastest growing areas of AI. Machine learning algorithms can analyze a huge amount of data from various fields, one of the most important of which is the medical field. Using machine learning we can understand complex and non-linear correlations between various factors, reducing the error in the predicted results. By exploring huge data sets it is possible to extract hidden critical information for decision making. With the machine learning algorithms we can build models which we train with a set of data and after the models "learn" we can perform predictions on new data. In this work, we tested machine learning classification algorithms for predicting heart disease in patients.

Data
----
This database from [UCI Machine learning repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

Attribute Information:
1. age
1. sex
1. chest pain type (4 values)
1. resting blood pressure
1. serum cholestoral in mg/dl
1. fasting blood sugar > 120 mg/dl
1. resting electrocardiographic results (values 0,1,2)
1. maximum heart rate achieved
1. exercise induced angina
1. oldpeak = ST depression induced by exercise relative to rest
1. the slope of the peak exercise ST segment
1. number of major vessels (0-3) colored by flourosopy
1. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

Machine Learning Classifiers
----------------------------
* Logistic Regression
* k-Nearest Neighbors (kNN)
* Multilayer Perceptron (MLP)
* Decision Tree
* Random Forest
* Gaussian Naive Bayes

Installation and Usage
----------------------
Make sure you have [python 3.8](https://www.python.org/downloads/release/python-380/)  installed in your system. 
The following command clone the repository.
```bash
git clone https://github.com/constandinos/heart-disease
cd heart-disease
```

### Interface

#### main.py
This class creates a GUI for input data.
```bash
cd src/
python main.py
```
![Image](figures/gui.png?raw=true "Example of main interface")

### Experiments

#### process_dataset.py
This class read the dataset and process data.
```bash
cd src/classes/
python process_dataset.py <dataset directory>
eg. python process_dataset.py ../../data/processed.cleveland.data
```

#### visualization.py
This class creates plots to visualize the data.
```bash
cd src/classes/
python visualization.py <dataset directory>
eg. python visualization.py ../../data/dataset.csv
```

#### build_model.py
This class find the best machine learning model to fit the data.
```bash
cd src/classes/
python build_model.py <dataset directory> <num of cpus>
eg. python build_model.py ../../data/dataset.csv 16
```

References
----------
* [Effective heart disease prediction system using data mining techniques](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5863635/)
* [Study of cardiovascular disease prediction model based on random forest in eastern China](https://www.researchgate.net/publication/340103704_Study_of_cardiovascular_disease_prediction_model_based_on_random_forest_in_eastern_China)
* [Survey of Machine Learning Algorithms for Disease Diagnostic](https://www.researchgate.net/publication/312629315_Survey_of_Machine_Learning_Algorithms_for_Disease_Diagnostic)

Predict heart disease
=====================
In this project, we implement a system that predicts if a patient has a heart disease.


Installation
------------
Make sure you have [python 3.8](https://www.python.org/downloads/release/python-380/)  installed in your system. 
The following command clone the repository.
```bash
git clone https://github.com/constandinos/heart-disease
cd heart-disease
```


Interface
---------
#### main.py
This class creates a GUI for input data.
```bash
cd src/
python main.py
```


Experiments
-----------
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

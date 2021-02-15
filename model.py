import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def read_data(filename):
    """
    Read data from a file into a dataframe.
    """

    return pd.read_csv(filename)

def cross_validation(estimator, x_train, y_train, k_folds=10, score_type='f1_weighted'):
    """
	This function will apply k-folds cross validation to calculate the average
	f1_weighted score in order to select the machine learning algorithm with
    highest score.

	Parameters
	----------
	clf_list: list of estimators
		Estimator (ml or nn) algorithm
	x_train: numpy array
		The train data
	y_train: numpy array
		The labels of train data
	k_folds: integer
		The number of folds
	score_type: string
		The name of score type

	Returns
	-------
	estimator_score: list of floats
		This list contains the best cross validation f1 scores of machine
        learning algorithms
	estimator_std: list of floats
		This list contains the cross validation standard deviations of machine
        learning algorithms
   """

    estimator_score, estimator_std = None, None  # return results
    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)

    # k-fold cross validation
    # print("Start "+str(k_folds)+"-folds cross validation...")
    f1_score = model_selection.cross_val_score(estimator, x_train, y_train, cv=kfold, scoring=score_type, n_jobs=-1)
    # append results to the return lists
    estimator_score = f1_score.mean()
    estimator_std = f1_score.std()
    # print("End cross validation")

    return estimator_score, estimator_std

def grid_search_cross_validation(clf_list, x_train, y_train, x_test, y_test, k_folds=10, score_type='f1_weighted'):
    """
	This function will apply grid search to search over specified parameter
    values for an estimator to find the optimal parameters for a machine
    learning algorithm.
	Also, this function will apply k-folds cross validation to calculate the
    average f1_weighted score in order to select the machine learning algorithm
    with highest score.

	Parameters
	----------
	clf_list: list of tuples with name of
		Each tuple contains the name of machine learning algorithm, the
        initialization estimator and a set with the parameters
	x_train: numpy array
		The train data
	y_train: numpy array
		The labels of train data
	k_folds: integer
		The number of folds
	score_type: string
		The name of score type

	Returns
	-------
	model_names: list of strings
		This list contains the names of machine learning algorithms
	model_scores: list of floats
		This list contains the best cross validation f1 scores of machine
        learning algorithms
	model_std: list of floats
		This list contains the cross validation standard deviations of machine
        learning algorithms
    test_scores: list of floats
        This list contains the evaluation score on the test data
   """

    model_names, model_scores, model_std, test_scores = [], [], [], []  # return list
    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)
    # kfold = model_selection.StratifiedKFold(n_splits=k_folds, shuffle=True)
    for name, model, parameters in clf_list:
        # grid search
        print("Grid search for " + name)
        search = GridSearchCV(model, parameters, scoring=score_type, cv=kfold, n_jobs=-1)
        search.fit(x_train, y_train)
        print("Best parameters: " + str(search.best_params_))
        best_est = search.best_estimator_  # estimator with the best parameters

        # k-fold cross validation
        f1_mean, f1_std = cross_validation(best_est, x_train, y_train, k_folds, score_type)
        # append results to the return lists
        model_names.append(name)
        model_scores.append(f1_mean)
        model_std.append(f1_std)

        y_pred = best_est.predict(x_test)
        test_score = f1_score(y_test, y_pred, average='weighted')
        test_scores.append(test_score)

    return model_names, model_scores, model_std, test_scores



df = read_data('data/dataset.csv')

X = df.drop('target', axis=1) # features
Y = df.target # target

# Split out dataset
x_train, x_test , y_train , y_test = train_test_split (X, Y, test_size =0.2) # test = 20%, train = 80%

clf_list = [("logistic_regression", LogisticRegression(), {'C': np.logspace(-4, 4, 20), \
                                                           'max_iter': [100, 200, 300, 400, 500]}),
            ("k-nn", KNeighborsClassifier(), {'n_neighbors': np.arange(1, 25), \
                                              'metric': ['euclidean', 'minkowski']}),
            ("mlp", MLPClassifier(), {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)], \
                                      'alpha': [0.0001, 0.05], \
                                      'learning_rate': ['constant', 'adaptive'], \
                                      'max_iter': [300, 500, 800, 1000, 2000]}),
            ("random_forest", RandomForestClassifier(), {'n_estimators': [200, 500, 1000], \
                                                         'max_features': ['sqrt', 'log2'], \
                                                         'max_depth': [50, 100, 200, 300]}),
            ("svc", SVC(), {'C': [0.1, 1, 10, 100], \
                            'gamma': [0.01, 0.1, 1], \
                            'kernel': ['rbf', 'linear', 'sigmoid']})]


model_names, model_scores, model_std, test_scores = grid_search_cross_validation(clf_list, x_train, y_train, x_test, y_test)
print(model_names)
print(model_scores)
print(model_std)
print(test_scores)

"""
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

pred = lr_clf.predict(X_test)
print(classification_report(y_test, pred))
"""
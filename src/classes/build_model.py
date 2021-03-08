# build_model.py
#
# This class find the best machine learning model to fit the data.
#
# Created by: Constandinos Demetriou, Mar 2021

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def read_data(filename):
    """
    Read data from a file into a dataframe.

    Parameters
	----------
	filename: str
	    The name of file with dataset
    """

    return pd.read_csv(filename)


def cross_validation(estimator, x_train, y_train, score_type, k_folds, num_cpus):
    """
	Applies k-folds cross validation to calculate the algorithm score in order to select
	the machine learning algorithm with highest score.

	Parameters
	----------
	estimator: estimators
		Estimator (ml or nn) algorithm
	x_train: numpy array
		The train data
	y_train: numpy array
		The labels of train data
	score_type: str
		The name of score type
	k_folds: int
		The number of folds
    num_cpus: int
        The number of cpus that will use this function

	Returns
	-------
	estimator_score: float
		The cross validation scores of machine learning algorithm
	estimator_std: float
		The cross validation standard deviations of machine learning algorithms
   """

    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)

    # k-fold cross validation
    score = model_selection.cross_val_score(estimator, x_train, y_train, cv=kfold, scoring=score_type, n_jobs=num_cpus)
    # append results to the return lists
    estimator_score = score.mean()
    estimator_std = score.std()

    return estimator_score, estimator_std


def grid_search_cross_validation(clf_list, x_train, y_train, x_test, y_test, num_cpus=-1, k_folds=10,
                                 score_type='accuracy'):
    """
	Aplies grid search to search over specified parameter values for an estimator
	to find the optimal parameters for a machine learning algorithm.
	Also, this function will apply k-folds cross validation to calculate the
    algorithm score in order to select the machine learning algorithm
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
	x_test: numpy array
		The test data
	y_test: numpy array
		The labels of test data
	num_cpus: int
        The number of cpus that will use this function
	k_folds: integer
		The number of folds
	score_type: string
		The name of score type

	Returns
	-------
	estimators: list of strings
		This list contains the names of machine learning algorithms
	acc: list of floats
		This list contains the best accuracy for each algorithm
	pred_prob: list of floats
		This list contains the predicted probabilities for each algorithm
   """

    estimators, acc, pred_prob = [], [], []

    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)

    for name, model, parameters in clf_list:
        print('Estimator: ' + name)
        estimators.append(name)

        # grid search
        search = GridSearchCV(model, parameters, scoring=score_type, cv=kfold, n_jobs=num_cpus)
        search.fit(x_train, y_train)
        print('Best parameters: ' + str(search.best_params_))
        best_est = search.best_estimator_  # estimator with the best parameters

        # k-fold cross validation
        accuracy_kfold, std_kfold = cross_validation(best_est, x_train, y_train, score_type, k_folds, num_cpus)
        print('kfold cross validation mean accuracy: ' + '{:.2f}'.format(accuracy_kfold))
        print('kfold cross validation standard deviation: ' + '{:.2f}'.format(std_kfold))

        # model evaluation
        y_pred = best_est.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc.append(accuracy)
        print('Evaluation accuracy: ' + '{:.2f}'.format(accuracy))

        # predict probabilities
        pred_prob.append(best_est.predict_proba(x_test))
        print()

    return estimators, acc, pred_prob


def plot_roc_curve(estimators, pred_prob, y_test):
    """
    Plots AUC-ROC curve.

    Parameters
	----------
    estimators: list of strings
		This list contains the names of machine learning algorithms
	pred_prob: list of floats
		This list contains the predicted probabilities for each algorithm
	y_test: numpy array
		The labels of test data
    """

    plt.figure()
    # define the color list fot ruc curve
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'gray']
    print('AUC-ROC curve')
    for i, esti in enumerate(estimators):
        # roc curve for models
        fpr, tpr, thresh = roc_curve(y_test, pred_prob[i][:, 1], pos_label=1)
        # auc scores
        auc_score = roc_auc_score(y_test, pred_prob[i][:, 1])
        print(esti + ': ' + '{:.2f}'.format(auc_score))
        # plot roc curves
        plt.plot(fpr, tpr, color=colors[i], label=esti + ' (AUC=' + '{:.2f}'.format(auc_score) + ')')

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend()
    plt.savefig('../../figures/roc.png')
    plt.clf()


def plot_evaluation_accuracy(estimators, acc):
    """
    Plots accuracy for each machine learning algorithm.

    Parameters
    ----------
    estimators: list of strings
        This list contains the names of machine learning algorithms
	acc: list of floats
		This list contains the best accuracy for each algorithm
    """

    plt.figure(figsize=(10, 5))
    plt.title('Evaluation')
    plt.xlabel('Machine Learning Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.bar(estimators, [i * 100 for i in acc])
    plt.savefig('../../figures/evaluation.png')
    plt.clf()


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# read dataset
dataset = read_data('../../data/dataset.csv')

# drop outliers
dataset = dataset.drop(dataset[~(np.abs(stats.zscore(dataset)) < 3).all(axis=1)].index)

# get data (X) and labels (Y)
X = dataset.drop('target', axis=1)
Y = dataset.target

# normalize data
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values

# split to train (80%) and test (20%) set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32, stratify=Y)

# create list with all possible parameters for each estimator
clf_list = [('LogisticRegression', LogisticRegression(), {'C': [1, 2, 3, 4],
                                                          'penalty': ['none', 'l1', 'l2', 'elasticnet'],
                                                          'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                                                          'max_iter': [100, 300, 500, 600, 700, 800, 900, 1000]}),
            ('kNN', KNeighborsClassifier(), {'n_neighbors': np.arange(1, 30),
                                             'weights': ['uniform', 'distance'],
                                             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                             'metric': ['euclidean', 'minkowski', 'manhattan']}),
            ('MLP', MLPClassifier(), {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                      'solver': ['lbfgs', 'sgd', 'adam'],
                                      'alpha': [0.0001, 0.05],
                                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                      'max_iter': [200, 300, 500, 800, 1000, 1500, 2000, 2500, 3000]}),
            ('DecisionTree', DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'],
                                                        'splitter': ['best', 'random'],
                                                        'max_features': ['auto', 'sqrt', 'log2'],
                                                        'max_depth': range(1, 15),
                                                        'min_samples_split': range(1, 10),
                                                        'min_samples_leaf': range(1, 10)}),
            ('RandomForest', RandomForestClassifier(), {'n_estimators': [100, 500, 1000, 1500],
                                                        'criterion': ['gini', 'entropy'],
                                                        'max_depth': [None, 4, 50, 100, 200, 300],
                                                        'max_features': ['auto', 'sqrt', 'log2']}),
            ('SVC', SVC(), {'probability': [True],
                            'C': [0.1, 1, 10, 100, 1000],
                            'gamma': ['scale', 'auto'],
                            'kernel': ['linear', 'rbf', 'sigmoid']})]
"""
clf_list = [('LogisticRegression', LogisticRegression(), {'C': [1], 'penalty': ['l2'], 'solver': ['lbfgs'],
                                                          'max_iter': [100]}),
            ('kNN', KNeighborsClassifier(), {'algorithm': ['auto'], 'metric': ['manhattan'], 'n_neighbors': [26],
                                             'weights': ['distance']}),
            ('MLP', MLPClassifier(), {'activation': ['identity'], 'alpha': [0.0001], 'hidden_layer_sizes': [(100,)],
                                      'learning_rate': ['constant'], 'max_iter': [300], 'solver': ['lbfgs']}),
            ('DecisionTree', DecisionTreeClassifier(), {'criterion': ['entropy'], 'max_depth': [5],
                                                        'max_features': ['auto'], 'min_samples_leaf': [9],
                                                        'min_samples_split': [8], 'splitter': ['random']}),
            ('RandomForest', RandomForestClassifier(), {'n_estimators': [100], 'criterion': ['gini'],
                                                        'max_depth': [None], 'max_features': ['auto']}),
            ('SVC', SVC(), {'probability': [True], 'C': [10], 'gamma': ['scale'], 'kernel': ['linear']})]
"""
# run grid search and cross validation
estimators, acc, pred_prob = grid_search_cross_validation(clf_list, x_train, y_train, x_test, y_test)

# plot evaluation accuracy
plot_evaluation_accuracy(estimators, acc)
# plot roc curve
plot_roc_curve(estimators, pred_prob, y_test)

# fit the model on training set on best algorithm
model = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=100)
model.fit(x_train, y_train)
# save the model to disk
filename = '../../model/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

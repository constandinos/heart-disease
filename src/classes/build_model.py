#######################################################################################################################
# build_model.py
# This class find the best machine learning model to fit the data.
#
# Execution commands:
# python build_model.py <dataset directory> <num of cpus>
# eg. python build_model.py ../../data/dataset.csv 16
#
# Created by: Constandinos Demetriou, 2021
#######################################################################################################################

import sys
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate


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
        The  data
    y_train: numpy array
        The labels of data
    score_type: str
        The name of score type
    k_folds: int
        The number of folds
    num_cpus: int
        The number of cpus that will use this function

    Returns
    -------
    avg_accuracy: float
        The cross validation accuracy of machine learning algorithm
    std_accuracy: float
        The cross validation standard deviations of machine learning algorithms
    """

    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # k-fold cross validation
    score = model_selection.cross_val_score(estimator, x_train, y_train.values.ravel(), cv=kfold, scoring=score_type,
                                            n_jobs=num_cpus)
    # append results to the return lists
    avg_accuracy = score.mean()
    std_accuracy = score.std()

    return avg_accuracy, std_accuracy


def grid_search_cross_validation(clf_list, x_train, y_train, num_cpus, score_type='accuracy', k_folds=5):
    """
    Applies grid search to search over specified parameter values for an estimator
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
        The  data
    y_train: numpy array
        The labels of data
    num_cpus: int
        The number of cpus that will use this function
    score_type: string
        The name of score type
    k_folds: integer
        The number of folds

    Returns
    -------
    model_names: list of strings
        This list contains the names of machine learning algorithms
    best_estimators: list of estimators
        This list contains the best estimators
    best_parameters: list of dict
        This list contains the best parameters for each machine learning algorithm.
    kfold_accuracy: list of floats
        This list contains the accuracy from k-fold cross validation
    kfold_std: list of floats
        This list contains the standard deviation from accuracy from k-fold cross validation
   """

    model_names, best_estimators, best_parameters, kfold_accuracy, kfold_std = [], [], [], [], []

    print('Start... Grid search and Cross validation\n')
    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)

    for name, model, parameters in clf_list:
        print(name)
        model_names.append(name)

        # grid search
        search = GridSearchCV(model, parameters, scoring=score_type, cv=kfold, n_jobs=num_cpus)
        search.fit(x_train, y_train)

        # results of grid search
        params = search.cv_results_['params']
        mean_test_score = search.cv_results_['mean_test_score']
        mean_test_score = [x * 100 for x in mean_test_score]
        df_grid_results = pd.DataFrame({'params': params, 'mean_test_score(%)': mean_test_score})
        print(tabulate(df_grid_results, headers='keys', showindex=False))
        print()

        # get best estimator
        best_parameters.append(search.best_params_)

        best_est = search.best_estimator_  # estimator with the best parameters
        best_estimators.append(best_est)

        # k-fold cross validation
        avg_accuracy, std_accuracy = cross_validation(best_est, x_train, y_train, score_type, k_folds, num_cpus)
        kfold_accuracy.append('{:.2f}'.format(avg_accuracy * 100))
        kfold_std.append('{:.2f}'.format(std_accuracy * 100))

    return model_names, best_estimators, best_parameters, kfold_accuracy, kfold_std


def predictions(x_train, y_train, x_test, y_test, model_names, best_estimators):
    """
    Splits dataset to train and test, predict probabilities, plot decision tree and confusion matrix for each machine
    learning algorithm.

    Parameters
    ----------
    x_train: numpy array
        The train  data
    y_train: numpy array
        The labels of train data
    x_test: numpy array
        The  test data
    y_test: numpy array
        The labels of test data
    model_names: list of strings
        This list contains the names of machine learning algorithms
    best_estimators: list of estimators
        This list contains the best estimators

    Returns
    -------
    pred_accuracy: list of floats
        This list contains the predicted accuracy for each algorithm on test set
    pred_prob: list of floats
        This list contains the predicted probabilities for each algorithm
    """

    pred_accuracy, pred_prob = [], []

    for i in range(0, len(model_names)):
        # fitting
        # clf = best_estimators[i].fit(x_train, y_train)

        # model evaluation
        y_pred = best_estimators[i].predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        pred_accuracy.append('{:.2f}'.format(accuracy * 100))

        # predict probabilities
        pred_prob.append(best_estimators[i].predict_proba(x_test))

        # plot confusion matrix
        plot_confusion_matrix(best_estimators[i], x_test, y_test)
        plt.title(model_names[i])
        plt.savefig('../../figures/confusion_matrix_' + model_names[i] + '.png')
        plt.clf()

    return pred_accuracy, pred_prob


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
    y_labels = [float(x) for x in acc]
    plt.bar(estimators, y_labels)
    plt.savefig('../../figures/evaluation.png')
    plt.clf()


def plot_roc_curve(model_names, pred_prob, y_test):
    """
    Plots AUC-ROC curve.

    Parameters
    ----------
    model_names: list of strings
        This list contains the names of machine learning algorithms
    pred_prob: list of floats
        This list contains the predicted probabilities for each algorithm
    y_test: numpy array
        The labels of test data

    Returns
    -------
    auc_list: list of floats
        This list contains AUC scores
    """

    auc_list = []
    plt.figure()
    # define the color list fot ruc curve
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'gray']
    for i, esti in enumerate(model_names):
        # roc curve for models
        fpr, tpr, thresh = roc_curve(y_test, pred_prob[i][:, 1], pos_label=1)
        # auc scores
        auc_score = roc_auc_score(y_test, pred_prob[i][:, 1])
        auc_list.append('{:.2f}'.format(auc_score * 100))
        # plot roc curves
        plt.plot(fpr, tpr, color=colors[i], label=esti + ' (AUC=' + '{:.2f}'.format(auc_score * 100) + '%)')

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

    return auc_list


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define input files and num of cpus
filein = sys.argv[1]
num_cpus = int(sys.argv[2])

# read dataset
dataset = read_data(filein)

# drop outliers
# dataset = dataset.drop(dataset[~(np.abs(stats.zscore(dataset)) < 3).all(axis=1)].index)

# get data (X) and labels (Y)
X = dataset.drop('target', axis=1)
Y = dataset.target

# normalize data
# X = (X - np.min(X)) / (np.max(X) - np.min(X)).values

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
            ('GaussianNB', GaussianNB(), {})]
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
            ('GaussianNB', GaussianNB(), {})]
"""
# grid search and cross validation
model_names, best_estimators, best_parameters, kfold_accuracy, kfold_std = grid_search_cross_validation(clf_list,
                                                                                                        x_train,
                                                                                                        y_train,
                                                                                                        num_cpus)

# print cross validation results
print('5-cross validation results')
df = pd.DataFrame({'Model': model_names, 'Accuracy(%)': kfold_accuracy, 'Std(%)': kfold_std,
                   'BestParameters': best_parameters})
print(tabulate(df, headers='keys', showindex=False))

# evaluation on test set
pred_accuracy, pred_prob = predictions(x_train, y_train, x_test, y_test, model_names, best_estimators)
plot_evaluation_accuracy(model_names, pred_accuracy)

# plot ROC AUC curve
auc_list = plot_roc_curve(model_names, pred_prob, y_test)

print('\nEvaluation on test set')
df = pd.DataFrame({'Model': model_names, 'Accuracy(%)': pred_accuracy, 'AUC(%)': auc_list})
print(tabulate(df, headers='keys', showindex=False))

"""
# fit the model on training set on best algorithm
model = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=100)
model.fit(x_train, y_train)
# save the model to disk
filename = '../../model/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
"""

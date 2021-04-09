#######################################################################################################################
# visualization.py
# This class creates plots to visualize the data.
#
# Execution commands:
# python visualization.py <dataset directory>
# eg. python visualization.py ../../data/dataset.csv
#
# Created by: Constandinos Demetriou, 2021
#######################################################################################################################

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sex(df):
    """
    Plot data about sex (Females, Males).

    Parameters
	----------
	df: numpy array
		The dataset
    """

    # Data to plot
    values = df.groupby(['sex']).size().values
    labels = 'Females', 'Males'
    # Plot
    plt.pie(values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.savefig('../../figures/sex.png')
    plt.clf()


def plot_target(df):
    """
    Plot data about target (Normal, Heart Disease).

    Parameters
	----------
	df: numpy array
		The dataset
    """

    # Data to plot
    values = df.groupby(['target']).size().values
    labels = 'Normal', 'Heart Disease'
    # Plot
    plt.pie(values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.savefig('../../figures/target.png')
    plt.clf()


def plot_sex_target(df):
    """
    Plot data about sex and target.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    # Data to plot
    pd.crosstab(df.sex, df.target).plot(kind='bar')
    # Plot
    labels = ['Female', 'Male']
    plt.xticks([0, 1], labels, rotation=0)
    plt.legend(['Normal', 'Heart Disease'])
    plt.ylabel('Frequency')
    plt.xlabel('')
    plt.savefig('../../figures/sex_target.png')
    plt.clf()


def plot_age(df):
    """
    Plot data about age.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    # Data to plot
    pd.crosstab(df.age, df.target).plot(kind='bar', figsize=(20, 6))
    # Plot
    plt.get_current_fig_manager().canvas.set_window_title('age')
    plt.legend(['Normal', 'Heart Disease'])
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('../../figures/age.png')
    plt.clf()


def plot_age_target_density(df):
    """
    Plot distribution of age with target.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    g = sns.displot(data=df, x='age', hue='target', kind='kde')
    g._legend.remove()
    plt.legend(['Normal', 'Heart Disease'])
    g.savefig('../../figures/age_target_density.png')
    plt.clf()


def plot_age_target_sex_density(df):
    """
    Plot distribution of age, target, sex.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    g = sns.displot(data=df, x='age', col='sex', hue='target', kind='kde')
    g._legend.remove()
    plt.legend(['Normal', 'Heart Disease'])
    g.savefig('../../figures/age_target_sex_density.png')
    plt.clf()


def correlation_analysis(df):
    """
    Correlation analysis between all features.

    Parameters
	----------
	df: numpy array
		The dataset

	Returns
	-------
	corr_target_labels: list of stings
	    The labels
	corr_target_values: list of floats
	    The correlation values
    """

    # Correlation analysis to data
    corr_matrix = df.corr()
    # Plot results
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt='.2f',
                     cmap='YlGnBu');
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('../../figures/correlation_analysis.png')
    plt.clf()

    # Select values for correlation with target
    corr_target = corr_matrix['target'].drop('target')
    corr_target_labels = corr_target.keys().values
    corr_target_values = corr_target.values

    return corr_target_labels, corr_target_values


def correlation_with_target(labels, values):
    """
    Correlation analysis for target.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(labels, abs(values))
    plt.ylabel('Correlation')
    plt.savefig('../../figures/correlation_for_target.png')
    plt.clf()


def plot_chest_pain(df):
    """
    Plot each type of chest pain with target.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    cp = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']
    normal = []
    disease = []
    total = len(df)
    for i in range(0, 4):
        normal.append(len(df[(df['cp'] == i) & (df['target'] == 0)]) / total)
        disease.append(len(df[(df['cp'] == i) & (df['target'] == 1)]) / total)

    data = pd.DataFrame({'cp': cp, 'Normal': normal, 'Heart Disease': disease})
    data.plot(x='cp', y=['Normal', 'Heart Disease'], kind='bar')
    plt.xticks(rotation=0)
    plt.xlabel('')
    plt.ylabel('Percentage of medical cases (%)')
    plt.savefig('../../figures/chest_pain.png')
    plt.clf()


def plot_pairplot(df):
    """
    Plot pairwise relationships in a dataset.

    Parameters
	----------
	df: numpy array
		The dataset
    """

    sns.pairplot(df)
    plt.savefig('../../figures/pairplot.png')
    plt.clf()


# =============================================================================#
#                                      MAIN                                    #
# =============================================================================#

# define the dataset name
filename = sys.argv[1]

# Read dataset
df = pd.read_csv(filename)

# Visualize the data
plot_sex(df)
plot_target(df)
plot_sex_target(df)
plot_age(df)
plot_age_target_density(df)
plot_age_target_sex_density(df)
labels, values = correlation_analysis(df)
correlation_with_target(labels, abs(values))
plot_chest_pain(df)
plot_pairplot(df)

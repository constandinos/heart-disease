import pandas as pd
import pickle


def build_data(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Creates a dataframe with test data.

    Parameters
	----------
    age: int
        Age of patient
    sex: int
        Sex of patient
    cp: int
        Chest pain type
    trestbps: int
        Resting blood pressure
    chol: int
        Serum cholestoral
    fbs: int
        Fasting blood sugar
    restecg: int
        Resting electrocardiographic results
    thalach: int
        Maximum heart rate achieved
    exang: int
        Exercise induced angina
    oldpeak: int
        ST depression induced by exercise relative to rest
    slope: int
        The slope of the peak exercise ST segment
    ca: int
        Number of major vessels (0-3) colored by flourosopy
    thal: int
        Thalassemia

	Returns
	-------
	df: numpy array
	    The test data
    """

    df = pd.DataFrame({'age': [age],
                       'sex': [sex],
                       'cp': [cp],
                       'trestbps': [trestbps],
                       'chol': [chol],
                       'fbs': [fbs],
                       'restecg': [restecg],
                       'thalach': [thalach],
                       'exang': [exang],
                       'oldpeak': [oldpeak],
                       'slope': [slope],
                       'ca': [ca],
                       'thal': [thal]})

    return df


def make_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
                    filename='../model/finalized_model.sav'):
    """
    Predicts if a patient has a heart problem.

    Parameters
    ----------
    age: int
        Age of patient
    sex: int
        Sex of patient
    cp: int
        Chest pain type
    trestbps: int
        Resting blood pressure
    chol: int
        Serum cholestoral
    fbs: int
        Fasting blood sugar
    restecg: int
        Resting electrocardiographic results
    thalach: int
        Maximum heart rate achieved
    exang: int
        Exercise induced angina
    oldpeak: int
        ST depression induced by exercise relative to rest
    slope: int
        The slope of the peak exercise ST segment
    ca: int
        Number of major vessels (0-3) colored by flourosopy
    thal: int
        Thalassemia
    filename: str
        The directory with stored model

    Returns
	-------
	y_predict: int
	    The prediction
    """

    # build test data
    x_test = build_data(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    # load model
    loaded_model = pickle.load(open(filename, 'rb'))

    # make prediction
    y_predict = loaded_model.predict(pd.DataFrame(x_test))

    return y_predict[0]

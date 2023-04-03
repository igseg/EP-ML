import pandas  as pd
import numpy   as np
import matplotlib.pyplot as plt
import pickle
import shap
from xgboost                 import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split # Maybe RandomizedSearchCV is more interesting

def variable_impact(data, target_variable, variables = 'all' ):
    data = data[~pd.isna(data[target_variable])]

    if variables == 'all':
        variables = list(data.keys())
        variables = [x for x in variables if x not in (target_variable, 'HB010','HB020','HB030','DB020')]

    y = data[target_variable]
    X = data[variables]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.367, random_state=42)

    best_cv = pickle.load(open('Models/{}.sav'.format('cv_best'), 'rb'))
    best_params = best_cv.get_params()

    xgb_regressor = XGBRegressor(**best_params)
    xgb_regressor.fit(X_train,y_train)

    explainer   = shap.TreeExplainer(xgb_regressor)
    shap_values = explainer.shap_values(X_train, check_additivity=False)

    shap.summary_plot(shap_values, X_train[variables])
    shap_values = shap_values / np.sum(np.mean(np.abs(shap_values),axis=0)) ## normalizing the average to sum 1
    shap.summary_plot(shap_values, X_train[variables], plot_type='bar')

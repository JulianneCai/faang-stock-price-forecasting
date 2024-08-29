import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, KFold 
from sklearn.svm import SVR

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical 

import xgboost as xgb 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


class HyperparameterTuner:
    def __init__(self, df, target):
        self._df = df 
        self._target = target

        self._x = df.drop(target, axis=1)
        self._y = df[target]
        
        self._x_train, self._x_test,\
        self._y_train, self._y_test = train_test_split(
            self._x, self._y, 
            test_size=0.2,
            train_size=0.8,
            shuffle=False,
            stratify=None
        )
        
        self._cross_val = KFold(n_splits=5, shuffle=False, random_state=None)
    
    def tune_xgb_regressor(self):
        """
        WARNING: this takes a while to run!

        Tunes the hyperparameters for an XGBRegressor using Bayesian optimisation.
        Model performance is measured using k-fold cross validation with 5 folds.

        Returns:
            XGBoost.XGBRegressor: an XGB regressor with the optimal hyperparameters
        """

        param_space = {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 13),
            'learning_rate': Real(0.01, 1.0),
            'gamma': Real(0, 5.0),
            'subsample': Real(0.5, 1),
        }

        optimiser = BayesSearchCV(XGBRegressor(),
                                  param_space,
                                  n_iter = 150,
                                  verbose=10
                                  )
        
        optimiser.fit(self._x_train, self._y_train)

        print(f'Best hyperparameters for XGBRegressor are: {optimiser.best_params_}')

        return optimiser.best_estimator_ 

    def tune_lgbm_regressor(self):
        """
        WARNING: this takes a while to run!

        Tunes the hyperparameters for an XGBRegressor using Bayesian optimisation.
        Model performance is measured using k-fold cross validation with 5 folds.

        Returns:
            XGBoost.XGBRegressor: an XGB regressor with the optimal hyperparameters
        """

        param_space = {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 13),
            'learning_rate': Real(0.01, 1.0)
        }

        optimiser = BayesSearchCV(LGBMRegressor(),
                                  param_space,
                                  n_iter = 150,
                                  verbose=10
                                  )
        
        optimiser.fit(self._x_train, self._y_train)

        print(f'Best hyperparameters for LGBMRegressor are: {optimiser.best_params_}')

        return optimiser.best_estimator_ 
    
    def tune_svc(self):
        param_space = {
            'C': Real(1, 5),
            'kernel': ['poly', 'rbf'],
            'degree': Integer(1, 5),
        }

        optimiser = BayesSearchCV(SVR(),
                                  param_space,
                                  n_iter=100,
                                  verbose=10)
        
        optimiser.fit(self._x_train, self._y_train)

        print(f'Best hyperparameters for SVC is: {optimiser.best_params_}')

        return optimiser.best_estimator_
    
    def get_y_pred(self, estimator):
        """
        Get predictions for dependent variable. For best results, use one of the 
        functions above to tune hyperparameters, and then pass it through this function.

        Params:
            estimator (XGBRegressor, LGBMRegressor, SVR, etc.): a choice of estimator 
        
        Returns:
            pandas.DataFrame: predictions based off the estimator.
        """
        if isinstance(estimator, (XGBRegressor, LGBMRegressor)):
            model = estimator.fit(self._x_train, self._y_train,
                                  eval_set=[(self._x_train, self._y_train), 
                                            (self._x_test, self._y_test)])
            y_pred = model.predict(self._x_test)
            return y_pred
        elif isinstance(estimator, SVR):
            model = estimator.fit(self._x_train, self._y_train)
            y_pred = model.predict(self._x_test)
            return y_pred 

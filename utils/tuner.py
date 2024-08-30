import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, KFold 
from sklearn.svm import SVR

from statsmodels.tsa.arima.model import ARIMA

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import xgboost as xgb 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


class HyperparameterTuner:
    def __init__(self, df, target, test_size):
        self._df = df 
        self._target = target
        self._test_size = test_size

        self._x = df.drop(target, axis=1)
        self._y = df[target]
        
        self._x_train, self._x_test,\
        self._y_train, self._y_test = train_test_split(
            self._x, self._y, 
            test_size=self._test_size,
            train_size=1-self._test_size,
            shuffle=False,
            stratify=None
        )
        
        self._cross_val = KFold(n_splits=5, shuffle=False, random_state=None)

    def get_x_train(self):
        return self._x_train 
    
    def get_x_test(self):
        return self._x_test 
    
    def get_y_train(self):
        return self._y_train 
    
    def get_y_test(self):
        return self._y_test

    def get_df(self):
        return self._df
    
    def set_df(self, df):
        self._df = df

    def get_target(self): 
        return self._target
    
    def set_target(self, target):
        self._target = target


class XGBTuner(HyperparameterTuner):
    def __init__(self, df, target):
        super().__init__(df, target)
    
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
    
class LGBMTuner(HyperparameterTuner):
    def __init__(self, df, target):
        super().__init__(df, target)

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
    

class ARIMATuner(HyperparameterTuner):
    def __init__(self, df, target):
        super().__init__(df, target)

    def _aic_arima_model(self, p, q, d):
        cfg = (p, q, d)
        model = ARIMA(self.get_y_train, order=cfg)
        model_fit = model.fit()

        return model_fit.aic

    def _bic_arima_model(self, p, q, d):
        cfg = (p, q, d)
        model = ARIMA(self._y_train, order=cfg)
        model_fit = model.fit()

        return model_fit.bic

    def _arima_grid_search(self, method):
        """
        Tunes ARIMA hyperparameters by performing a grid search, using either AIC or BIC.

        Params:
            x_train (pandas.DataFrame): the training dataset of the independent variables 
            y_true (pandas.Series): the testing values of the dependent variable 
            method (str): the scoring function -- can be either aic or bic
        
        Returns:
            tuple: the hyperparmeters p, q, and d
        """
        best_score = float('inf') 
        score = None
        best_cfg = None

        #  the p and q hyperparameters are given by statistically significant 
        #  lags on the autocorrelation plot
        p_values = self.get_lags(self.get_y_train)
        q_values = self.get_lags(self.get_y_train)
        d_values = []

        #  if the time series is stationary, then it does not have a unit root, hence d=0
        if self.is_stationary(self.get_y_train):
            d_values.append(0)
        else:
            #  TODO: find out how to calculate these d values
            #  this is just a placeholder in the meantime
            d_values = [1, 2, 3, 4]

        for p in p_values:
            for q in q_values:
                for d in d_values:
                    cfg = (p, q, d)
                    if method == 'aic':
                        score = self._aic_arima_model(self.get_y_train, p, q, d)
                    elif method == 'bic':
                        score = self._bic_arima_model(self.get_y_train, p, q, d)
                    print(f'score = {score}, cfg = {cfg}')
                    if score < best_score:
                        best_score = score
                        best_cfg = cfg 
        return best_cfg


class GARCHTuner(HyperparameterTuner):
    def __init__(self, df, target):
        super().__init__(df, target)

    def _garch_grid_search_optimal_hyperparameters(self, p_values, q_values):
        raise NotImplementedError
    
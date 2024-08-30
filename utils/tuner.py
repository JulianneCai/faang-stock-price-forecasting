import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, KFold 
from sklearn.svm import SVR

from utils.estimator import ARIMAEstimator, GARCHEstimator

from statsmodels.tsa.arima.model import ARIMA

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import xgboost as xgb 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


class HyperparameterTuner:
    """
    Abstract class representing an object that tunes the hyperparameters of a 
    machine learning estimator.
    """
    def __init__(self, df, target, test_size):
        """
        Params:
            df (pandas.DataFrame): dataset that we wish to perform analysis on
            target (str): name of target feature 
            test_size (float): size of testing dataset (must be between 0 and 1.0)
        """
        self._df = df 
        self._target = target
        self._test_size = test_size

        #  independent variable 
        self._x = df.drop(target, axis=1)
        #  dependent variable
        self._y = df[target]
        
        #  splitting independent and dependent variables into training and testing datasets
        self._x_train, self._x_test,\
        self._y_train, self._y_test = train_test_split(
            self._x, self._y, 
            test_size=self._test_size,
            train_size=1-self._test_size,
            shuffle=False,
            stratify=None
        )
        
        self._cross_val = KFold(n_splits=5, shuffle=False, random_state=None)

    def bayesian_optimisation(self):
        """
        Abstract method implemented by subclasses
        """
        pass

    def grid_search(self):
        """
        Abstract method implemented by subclasses
        """
        pass

    def random_forest(self):
        """
        Abstract method implemented by subclasses
        """
        pass

    def get_x_train(self):
        """
        Returns:
            pandas.DataFrame: training dataset of independent variables
        """
        return self._x_train 
    
    def get_x_test(self):
        """
        Returns:
            pandas.DataFrame: testing dataset of indepedent variables
        """
        return self._x_test 
    
    def get_y_train(self):
        """
        Returns:
            pandas.Series: training dataset of dependent variable
        """
        return self._y_train 
    
    def get_y_test(self):
        """
        Returns:
            pandas.Series: testing dataset of dependent variable
        """
        return self._y_test

    def get_df(self):
        """
        Returns:
            pandas.DataFrame: entire dataset
        """
        return self._df
    
    def set_df(self, df):
        """
        Params:
            df (pandas.DataFrame): new dataset
        """
        self._df = df

    def get_target(self): 
        """
        Returns:
            str: name of the target feature
        """
        return self._target
    
    def set_target(self, target):
        """
        Params:
            target (str): name of new target feature
        """
        self._target = target


class XGBTuner(HyperparameterTuner):
    """
    Concrete class representing an object that tunes an XGBRegressor object.
    """
    def __init__(self, df, target):
        super().__init__(df, target)
    
    def bayesian_optimisation(self):
        """
        WARNING: this takes a while to run!

        Tunes the hyperparameters for an XGBRegressor using Bayesian optimisation.
        Model performance is measured using k-fold cross validation with k=5.

        Returns:
            skopt.BayesSearchCV: returns a fitted optimiser. Hyperparameters and estimators 
            can be extracted using the best_params_() and best_estimator_() method
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

        return optimiser
    
    def random_forest(self):
        """
        Tunes hyperparameters using a random forest regressor.
        """
        raise NotImplementedError

    
class LGBMTuner(HyperparameterTuner):
    """
    Concrete class representing an object that tunes an LGBMRegressor object.
    """
    def __init__(self, df, target):
        super().__init__(df, target)

    def bayesian_optimisation(self):
        """
        WARNING: this takes a while to run!

        Tunes the hyperparameters for an XGBRegressor using Bayesian optimisation.
        Model performance is measured using k-fold cross validation with 5 folds.

        Returns:
            skopt.BayesSearchCV: returns a fitted optimiser. Hyperparameters and estimators 
            can be extracted using the best_params_() and best_estimator_() method
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

        return optimiser 


class SVMTuner(HyperparameterTuner):
    """
    Concrete class that tunes an SVR (support vector regressor) object.
    """
    def __init__(self, df, target, test_size):
        super().__init__(df, target, test_size)
    
    def bayesian_optimisation(self):
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

        return optimiser
    

class ARIMATuner(HyperparameterTuner):
    """
    Concrete class representing an object that tunes ARIMA hyperparameters:
    parameters p and q, and the unit root term d
    """
    def __init__(self, df, target):
        super().__init__(df, target)

    def _aic_arima_model(self, p, q, d):
        """
        Calculates the Akaike Information Criterion (AIC) of an ARIMA(p,q,d) model

        Returns:
            float: the AIC of an ARIMA(p,q,d) model
        """
        cfg = (p, q, d)
        model = ARIMA(self.get_y_train, order=cfg)
        model_fit = model.fit()

        return model_fit.aic

    def _bic_arima_model(self, p, q, d):
        """
        Calculates the Bayesian Information Criterion (BIC) of an ARIMA(p,q,d) model.

        Returns:
            float: the BIC of an ARIMA(p,q,d) model
        """
        cfg = (p, q, d)
        model = ARIMA(self._y_train, order=cfg)
        model_fit = model.fit()

        return model_fit.bic

    def grid_search(self, method):
        """
        Tunes ARIMA hyperparameters by performing a grid search, using 
        Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC)
        as the scoring function.

        Params:
            x_train (pandas.DataFrame): the training dataset of the independent variables 
            y_true (pandas.Series): the testing values of the dependent variable 
            method (str): the scoring function -- one of either {'aic', 'bic'}
        
        Returns:
            tuple: the hyperparmeters p, q, and d
        """
        best_score = float('inf') 
        score = None
        best_cfg = None

        #  the p and q hyperparameters are given by statistically significant 
        #  lags on the autocorrelation plot
        p_values = self.get_lags(self.get_y_train())
        q_values = self.get_lags(self.get_y_train())
        d_values = []

        #  if the time series is stationary, then it does not have a unit root, hence d=0
        if self.is_stationary(self.get_y_train()):
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
                        score = self._aic_arima_model(self.get_y_train(), p, q, d)
                    elif method == 'bic':
                        score = self._bic_arima_model(self.get_y_train(), p, q, d)
                    print(f'score = {score}, cfg = {cfg}')
                    if score < best_score:
                        best_score = score
                        best_cfg = cfg 
        model = ARIMAEstimator(self.get_y_train(), order=best_cfg)
        return model


class GARCHTuner(HyperparameterTuner):
    """
    Concrete class representing an object that tunes the hyperparameters of a 
    GARCH model.
    """
    def __init__(self, df, target):
        super().__init__(df, target)

    def grid_search(self, p_values, q_values):
        raise NotImplementedError
        best_p, best_q = None, None
        model = GARCHEstimator(self.get_y_train(), p=best_p, q=best_q)
        return model

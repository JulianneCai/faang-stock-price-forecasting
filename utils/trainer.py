import pandas as pd
import yfinance as yf
import numpy as np
from numpy import sqrt, log

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import arch

from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import adfuller


class Trainer:
    """
    Class that contains methods for extracting time series features, and hyperparameter values. 
    It has ARIMATrainer and GARCHTrainer, which tunes hyperparameters for an ARIMA(p,q,d) model, 
    and a GARCH(p,q) model. It also has a subclass VolatilityForecaster, which uses the tuned 
    ARIMA and GARCH models to forecast volatility of a given stock.
    """
    def __init__(self, symbol, period, test_size, train_size):
        """
        Params:
            symbol (str): the stock symbol (e.g. "GOOG" for Google, "AAPL" for apple)

            period (str): time period of stocks 

            test_size (float): size of test dataset. Sum of test_size and train_size must equal 1

            train_size (float): size of training dataset. Sum of test_size and train_size must equal 1
        """
        self._test_size = test_size
        self._train_size = train_size
        self._symbol = symbol
        self._period = period


    def generate_features(self):
        """
        Generates all the features that we need

        Returns:
            pandas.DataFrame: dataframe with all the new features
        """
        df = yf.Ticker(self._symbol).history(period=self._period)

        df['daily_returns_close'] = df['Close'].pct_change()
        df['daily_returns_open'] = df['Open'].pct_change()

        df['daily_returns_close_squared'] = df['daily_returns_close'] ** 2
        df['daily_returns_open_squared'] = df['daily_returns_open'] ** 2

        window = 2

        df['hist_vol_close'] = df['daily_returns_close'].rolling(window=window).std() * sqrt(252 / window)
        df['hist_vol_open'] = df['daily_returns_open'].rolling(window=window).std() * sqrt(252 / window)

        df['log_returns_close'] = log(df['Close']).diff()
        df['log_returns_open'] = log(df['Open']).diff()

        df['real_vol_close'] = df['log_returns_close'].rolling(window=window).std() * sqrt(252 / window)
        df['real_vol_open'] = df['log_returns_open'].rolling(window=window).std() * sqrt(252 / window)

        df['volume_change'] = df['Volume'].diff()
        
        return df

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


    def partition_data(self, df):
        """
        Splits dataset into testing and training datasets.

        Params:
            df (pandas.DataFrame): the time series data

        Returns: 
            pandas.DataFrame: tuple containing training and testing dataset, in that order
        """
        train, test = train_test_split(df,
                                                   train_size=self._train_size,
                                                   test_size=self._test_size,
                                                   shuffle=False,
                                                   stratify=None)
        return train, test
    
    def is_stationary(self, df):
        """
        Checks if the time series is stationary using the Augmented Dickey-Fuller test

        :params df: time series data 
        :type df: pandas.DataFrame 

        :returns: (bool) true if stationary, false if not
        """
        p_value = adfuller(df)[1]
        print(p_value)
        if p_value <= 0.05:
            return True 
        else:
            return False

 
    def get_lags(self, df):
        selection_results = ar_select_order(df, maxlag=8)
        return selection_results.ar_lags

    def get_symbol(self):
        return self._symbol
    
    def set_symbol(self, symbol):
        self._symbol = symbol 

    def get_period(self):
        return self._period 
    
    def set_period(self, period):
        self._period = period

    def get_train_size(self):
        return self._train_size

    def set_train_size(self, train_size):
        self._train_size = train_size

    def get_test_size(self):
        return self._test_size
    
    def set_test_size(self, test_size):
        self._test_size = test_size


class ARIMATrainer(Trainer):
    def __init__(self, symbol, period, test_size, train_size):
        super().__init__(symbol, period, test_size, train_size)
    
    def walk_forward_eval(self, y_train, y_test):
        raise NotImplementedError
    
    def forecast_out_of_sample(self, y_train, y_test):
        raise NotImplementedError

    def get_p_values(self):
        return self._p_values
    
    def set_p_values(self, p_values):
        self._p_values = p_values

    def get_q_values(self):
        return self._q_values
    
    def set_q_values(self, q_values):
        self._q_values = q_values

    def get_d_values(self):
        return self._d_values
    
    def set_d_values(self, d_values):
        self._d_values = d_values


class GARCHTrainer(Trainer):
    def __init__(self, symbol, period, test_size, train_size):
        super().__init__(symbol, period, test_size, train_size)

    def walk_forward_eval(self, y_train, y_test):
        raise NotImplementedError
    
    def forecast_out_of_sample(self, y_train, y_test):
        raise NotImplementedError

    def get_p_values(self):
        return self._p_values 
    
    def set_p_values(self, p_values):
        self._p_values = p_values 

    def get_q_values(self):
        return self.get_q_values
    
    def set_q_values(self, q_values):
        self._q_values = q_values

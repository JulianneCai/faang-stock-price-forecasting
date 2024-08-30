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
    Abstract class representing an object that trains a machine learning estimator 
    on a training dataset, and then uses that to predict features, either in-sample 
    or out-of-sample.
    """
    def __init__(self, symbol, period):
        """
        Params:
            symbol (str): the stock symbol (e.g. "GOOG" for Google, "AAPL" for apple)

            period (str): time period of stocks -- must be one of 
            {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}

            test_size (float): size of test dataset. Sum of test_size and train_size must equal 1

            train_size (float): size of training dataset. Sum of test_size and train_size must equal 1
        """
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
        TODO: move this to subclass

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
        
    def walk_forward_eval(self):
        """
        Abstract method implemented by subclasses
        """
        pass

    def forecast_out_of_sample(self):
        """
        Abstract method implemented by subclasses
        """
        pass
    
    def forecast_in_sample(self):
        """
        Abstract method implemented by subclasses
        """
        pass

 
    def get_lags(self, df):
        """
        Obtains the statistically significant autocorrelation lags

        Returns:
            list of int: statistically siginificant lag values
        """
        selection_results = ar_select_order(df, maxlag=8)
        return selection_results.ar_lags

    def get_symbol(self):
        """
        Returns:
            str: stock symbol
        """
        return self._symbol
    
    def set_symbol(self, symbol):
        """
        Params:
            symbol (str): new stock symbol
        """
        self._symbol = symbol 

    def get_period(self):
        """
        Returns:
            str: time period
        """
        return self._period 
    
    def set_period(self, period):
        """
        Params:
            period (str): new time period
        """
        self._period = period


class XGBoostTrainer(Trainer):
    def __init__(self, symbol, period):
        super().__init__(symbol, period)

    def forecast_in_sample(self, estimator, x_train, y_train, x_test, y_test):
        model = estimator.fit(x_train, y_train, 
                              eval_set = [(x_train, y_train), (x_test, y_test)]
                              )
        y_pred = model.predict(x_test)
        return y_pred


class LGBMTrainer(Trainer):
    def __init__(self, symbol, period):
        super().__init__(symbol, period)

    def forecast_in_sample(self, estimator, x_train, y_train, x_test, y_test):
        model = estimator.fit(x_train, y_train, 
                              eval_set = [(x_train, y_train), (x_test, y_test)]
                              )
        y_pred = model.predict(x_test)
        return y_pred


class SVMTrainer(Trainer):
    def __init__(self, symbol, period):
        super().__init__(symbol, period)

    def forecast_in_sample(self, estimator, x_train, y_train, x_test, y_test):
        model = estimator.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred


class ARIMATrainer(Trainer):
    """
    Concrete class representing an object that trains an ARIMA model, and then 
    predicts future values of target feature.
    """
    def __init__(self, symbol, period, test_size, train_size):
        super().__init__(symbol, period, test_size, train_size)
    
    def walk_forward_eval(self, estimator, y_train, y_test):
        """
        Performs walk-forward analysis on the testing dataset. Model trains itself on the 
        training dataset, and makes a prediction for the next time step. The true value of 
        the next time step is appended to the training dataset, and the model is re-fitted using 
        the same parameters.

        Params:
            y_train (pandas.Series): the time series training dataset 
            y_test (pandas.Series): the time series testing dataset
        Returns:
            pandas.Series: predictions made by the ARIMA model
        """
        raise NotImplementedError
    
    def forecast_out_of_sample(self, estimator, y_train, y_test, horizon):
        """
        Forecasts the target feature for the next few timesteps.

        Params:
            y_train (pandas.Series): the time series training dataset
            y_test (pandas.Series): the time series testing dataset 
            horizon (int): the number of time steps to forecast 
        
        Returns:
            pandas.Series: values forecasted by the ARIMA model
        """
        raise NotImplementedError
    
    def forecast_in_sample(self, estimator, y_train, y_test):
        """
        Forecasts y_test values using y_train values.

        Params:
            y_train (pandas.Series): the time series training dataset
            y_test (pandas.Series): the time series testing dataset 

        Returns:
            pandas.Series: values forecasted by the ARIMA model
        """
        raise NotImplementedError


class GARCHTrainer(Trainer):
    """
    Concrete class representing an object that trains a GARCH model, and then 
    predicts future values of target feature. GARCH models are only used for 
    volatility forecasting.
    """
    def __init__(self, symbol, period, test_size, train_size):
        super().__init__(symbol, period, test_size, train_size)

    def walk_forward_eval(self):
        """
        Performs walk-forward analysis on the testing dataset. Model trains itself on the 
        training dataset, and makes a prediction for the next time step. The true value of 
        the next time step is appended to the training dataset, and the model is re-fitted using 
        the same parameters.

        Params:
            y_train (pandas.Series): the time series training dataset 
            y_test (pandas.Series): the time series testing dataset
        Returns:
            pandas.Series: predictions made by the GARCH model
        """
        raise NotImplementedError
    
    def forecast_out_of_sample(self):
        """
        Forecasts the target feature for the next few timesteps.

        Params:
            y_train (pandas.Series): the time series training dataset
            y_test (pandas.Series): the time series testing dataset 
            horizon (int): the number of time steps to forecast 
        
        Returns:
            pandas.Series: values forecasted by the GARCH model
        """
        raise NotImplementedError
    
    def forecast_in_sample(self, y_train, y_test):
        """
        Forecasts y_test values using y_train values.

        Params:
            y_train (pandas.Series): the time series training dataset
            y_test (pandas.Series): the time series testing dataset 

        Returns:
            pandas.Series: values forecasted by the GARCH model
        """
        raise NotImplementedError

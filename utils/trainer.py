import pandas as pd
import yfinance as yf
import numpy as np
from numpy import sqrt, log

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import arch

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

        window = 2

        df['hist_vol_close'] = df['daily_returns_close'].rolling(window=window).std() * sqrt(252 / window)
        df['hist_vol_open'] = df['daily_returns_open'].rolling(window=window).std() * sqrt(252 / window)

        df['log_returns_close'] = log(df['Close']).diff()
        df['log_returns_open'] = log(df['Open']).diff()

        df['real_vol_close'] = df['log_returns_close'].rolling(window=window).std() * sqrt(252 / window)
        df['real_vol_open'] = df['log_returns_open'].rolling(window=window).std() * sqrt(252 / window)

        df['volume_change'] = df['Volume'].diff()
        
        return df

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


    def _mse_arima_model(self, x_train, y_true, p, q, d):
        cfg = (p, q, d)
        model = ARIMA(x_train, order=cfg)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(y_true))
        y_pred = pd.Series(forecast)

        return mean_squared_error(y_true, y_pred)
    
    def _aic_arima_model(self, x_train, y_true, p, q, d):
        cfg = (p, q, d)
        model = ARIMA(x_train, order=cfg)
        model_fit = model.fit()

        return model_fit.aic

    def _bic_arima_model(self, x_train, y_true, p, q, d):
        cfg = (p, q, d)
        model = ARIMA(x_train, order=cfg)
        model_fit = model.fit()

        return model_fit.bic

    def grid_search_optimal_hyperparameters(self, x_train, y_true):
        best_mse = float('inf') 
        best_cfg = None

        #  the p and q hyperparameters are given by statistically significant 
        #  lags on the autocorrelation plot
        p_values = self.get_lags(x_train)
        q_values = self.get_lags(x_train)
        d_values = []

        print(p_values)

        #  if the time series is stationary, then it does not have a unit root, hence d=0
        if self.is_stationary(x_train):
            d_values.append(0)
        else:
            #  TODO: find out how to calculate these d values
            #  this is just a placeholder in the meantime
            d_values = [1, 2, 3, 4]

        for p in p_values:
            for q in q_values:
                for d in d_values:
                    cfg = (p, q, d)
                    mse = self._mse_arima_model(x_train, y_true, p, q, d)
                    print(cfg, mse)
                    if mse < best_mse:
                        best_mse = mse
                        best_cfg = cfg 
        return best_cfg

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
    def __init__(self, symbol, period, test_size, train_size, p_values, q_values):
        super().__init__(symbol, period, test_size, train_size)

        self._train = super()._train
        self._test = super()._test


    def grid_search_optimal_hyperparameters(self, p_values, q_values):
        raise NotImplementedError
    
    def get_p_values(self):
        return self._p_values 
    
    def set_p_values(self, p_values):
        self._p_values = p_values 

    def get_q_values(self):
        return self.get_q_values
    
    def set_q_values(self, q_values):
        self._q_values = q_values

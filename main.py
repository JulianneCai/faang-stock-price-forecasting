import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from utils.trainer import Trainer, ARIMATrainer

from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from skopt import BayesSearchCV
from skopt.space import Integer

import arch

import warnings


if __name__ == "__main__":
    stock_id = ["GOOG", "MSFT", "AMZN", "META", "AAPL"]

    trainer = ARIMATrainer(symbol='AAPL', period='5y', test_size=0.2, train_size=0.8)

    df = trainer.generate_features()
    print(df.head())
    df.dropna(inplace=True)

    x = df.drop('hist_vol_close', axis=1)
    y = df['hist_vol_close']


    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=0.2,
                                                        train_size=0.8,
                                                        shuffle=False,
                                                        stratify=None)

    selector = ar_select_order(y_train, maxlag=12, glob=True) 

    params = selector.ar_lags
    print(params)

    plt.show()

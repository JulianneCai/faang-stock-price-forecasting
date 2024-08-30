# Stock Price and Volatility Forecasting using ML Methods
This project uses XGBoost, LightGBM, ARIMA models, and the Keras API from tensorflow to forecast stock prices of FAANG companies. GARCH models are also used to forecast the realised volatilties of these stocks.

XGBoost and LightGBM hyperparameters are tuned using Bayesian optimisers (from the scikit-optimize package). ARIMA hyperparameters were determined by differencing the time series to determine possible d values, and the lag values p and q were determined using the statistically significant lags of the autocorrelation plot of the target feature. A grid search method was then used to determine the optimal hyperparameters for the ARIMA model, using Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC) for scoring.

# Model Architecture
![Untitled Diagram drawio](https://github.com/user-attachments/assets/3fce05cf-d2b7-469e-a9d7-b6b80fb5cb57)

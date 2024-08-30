from statsmodels.tsa.arima.model import ARIMA

from arch import arch_model


class ARIMAEstimator:
    def __init__(self, series, order):
        self._series = series
        self._order = order 
    
    def get_params(self):
        return self._order 
    
    def set_params(self, order):
        self._order = order
    
    def get_estimator(self):
        return ARIMA(self._y_train, order=self._order)


class GARCHEstimator:
    def __init__(self, series, p, q):
        self._series = series
        self._p = p
        self._q =q 

    def get_p(self):
        return self._p 
    
    def set_p(self, p):
        self._p = p

    def get_q(self):
        return self._q 
    
    def set_q(self, q):
        self._q = q 

    def get_estimator(self):
        return arch_model(self._series, 
                          vol='GARCH',
                          p=self._p,
                          q=self._q
                          )

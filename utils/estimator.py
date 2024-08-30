from statsmodels.tsa.arima.model import ARIMA

from arch import arch_model

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.svm import SVR


class ARIMAEstimator:
    def __init__(self, series, order):
         self._series = series
         self._order = order 
    
    def get_params(self):
        return self._order 
    
    def set_params(self, order):
        self._order = order
    
    def get_estimator(self):
        return ARIMA(self._series, order=self._order)


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


class XGBEstimator:
    def __init__(self, n_estimators, max_depth, learning_rate, gamma, subsample):
        self._n_estimators = n_estimators
        self._max_depth = max_depth 
        self._learning_rate = learning_rate 
        self._gamma = gamma 
        self._subsample = subsample

    def get_n_estimators(self):
        return self._n_estimators

    def set_n_estimators(self, n_estimators):
        self._n_estimators = n_estimators

    def get_max_depth(self):
        return self._max_depth 
    
    def set_max_depth(self, max_depth):
        self._max_depth = max_depth

    def get_learning_rate(self):
        return self._learning_rate
    
    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def get_gamma(self):
        return self._gamma 
    
    def set_gamma(self, gamma):
        self._gamma = gamma 

    def get_subsample(self):
        return self._subsample 
    
    def set_subsample(self, subsample):
        self._subsample = subsample
    
    def get_estimator(self):
        return  XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            gamma=self._gamma,
            subsample=self._subsample
        )


class LGBMEstimator:
    def __init__(self, n_estimators, max_depth, learning_rate):
        self._n_estimators = n_estimators
        self._max_depth = max_depth 
        self._learning_rate = learning_rate 

    def get_n_estimators(self):
        return self._n_estimators

    def set_n_estimators(self, n_estimators):
        self._n_estimators = n_estimators

    def get_max_depth(self):
        return self._max_depth 
    
    def set_max_depth(self, max_depth):
        self._max_depth = max_depth

    def get_learning_rate(self):
        return self._learning_rate
    
    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def get_estimator(self):
        return LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate
        )


class SVMEstimator:
    def __init__(self, C, kernel, degree):
        self._C = C 
        self._kernel = kernel 
        self._degree = degree 
    
    def get_C(self):
        return self._C 
    
    def set_C(self, C):
        self._C = C 

    def get_kernel(self):
        return self._kernel 
    
    def set_kernel(self, kernel):
        self._kernel = kernel 

    def get_degree(self):
        return self._degree 
    
    def set_degree(self, degree):
        self._degree = degree

    def get_estimator(self):
        return SVR(
            C=self._C,
            kernel=self._kernel,
            degree=self._degree
        )

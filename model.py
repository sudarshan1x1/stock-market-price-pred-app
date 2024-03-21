# *********** Importing Libraries ***********
import pandas as pd 
import numpy as np
from datetime import datetime 
import yfinance as yf
import matplotlib.pyplot as plt1
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
import warnings
# *********** Ignore Warnings ***********

warnings.filterwarnings('ignore')


class Model:
    forecast = int(10)

    def __init__(self):
        self.symbol = None
        self.df = None
        self.lr_pred = None
        self.forecast_set = None
        self.mean = None
        self.order_type = ''
        self.y_test = None
        self.y_test_pred = None
        self.today_stock = None 

    def get_data(self):
        end = datetime.now()
        start = datetime(end.year-2, end.month, end.day)
        data = yf.download(self.symbol, start=start, end=end)
        self.df = pd.DataFrame(data=data)
        self.today_stock = self.df.iloc[-1]

    def linear_reg_algo(self):
        
        self.df['Close after n days'] = self.df['Close'].shift(-self.forecast)
        df_1 = self.df[['Close', 'Close after n days']]
        y = np.array(df_1.iloc[:-self.forecast,-1]) 
        y = np.reshape(y, (-1,1)) 
        X = np.array(df_1.iloc[:-self.forecast,0:-1]) 

        X_forecasted = np.array(df_1.iloc[-self.forecast:,0:-1]) 

        X_train = X[0:int(0.8*len(self.df)),:]
        X_test = X[int(0.8*len(self.df)):,:]
        y_train = y[0:int(0.8*len(self.df)),:]
        
        self.y_test = y[int(0.8*len(self.df)):,:]

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_forecasted = sc.transform(X_forecasted)

        regressor = LinearRegression(n_jobs=-1)
        regressor.fit(X_train, y_train)

        self.y_test_pred = regressor.predict(X_test)
        self.y_test_pred = self.y_test_pred*(1.02)

        self.forecast_set = regressor.predict(X_forecasted)
        self.forecast_set = self.forecast_set*(1.02)
        self.mean = self.forecast_set.mean()
        self.lr_pred = self.forecast_set[0,0]
        self.order_type = self._signal()
        self._display_graph()


        return self.today_stock, self.lr_pred, self.forecast_set, self.order_type

    def _signal(self):

        if self.today_stock['Close'] < self.mean:
            self.order_type = "BUY"

        else:
            self.order_type = "SELL"

        return self.order_type
    
    def _display_graph(self):
        plt1.figure(figsize=(7,7))
        plt1.plot(self.y_test, color = 'red', label = 'Real Stock Price')
        plt1.plot(self.y_test_pred, color = 'blue', label = 'Predicted Stock Price')
        plt1.title('Stock Price Prediction')
        plt1.xlabel('Time')
        plt1.ylabel('Stock Price')
        plt1.legend(loc='upper left')
        plt1.savefig('linear_reg_algo.png')
        


    


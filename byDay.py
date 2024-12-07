import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
import pmdarima as pm
from pmdarima import model_selection
import statsmodels.api as sm
import warnings

data = pd.read_csv('bike_sales.csv')

data['OrderDate'] = pd.to_datetime(data['OrderDate'])

# Aggregate sales data by day and product
daily_sales = data.groupby(['OrderDate', 'ProductID'])['OrderQty'].sum().reset_index()

predicted_sales = []
last_order_date = data['OrderDate'].max()

for product_id in daily_sales['ProductID'].unique():
    product_sales = daily_sales[daily_sales['ProductID'] == product_id].copy()
    
    if product_sales['OrderDate'].max() < last_order_date - timedelta(days=10):
        continue
        
    product_sales = product_sales.set_index('OrderDate').sort_index()
    product_sales.reset_index()
    product_sales.index = pd.DatetimeIndex(product_sales.index).to_period('D')
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = sm.tsa.SARIMAX(endog=product_sales['OrderQty'], order=(3,0,2), seasonal_order=(3,0,2,5), disp=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=1)
        
        # Create scatter plot showing all historical data and prediction
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(product_sales)), product_sales['OrderQty'], 
                   color='blue', label='Historical Sales', marker='o')
        plt.scatter(len(product_sales), forecast.values[0], 
                   color='red', label='Prediction', marker='x', s=100)
        plt.title(f'Sales History and Prediction for Product {product_id}')
        plt.xlabel('Time Period (Days)')
        plt.ylabel('Order Quantity')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Compare to previous day's actual sales
        last_day_sales = product_sales['OrderQty'].iloc[-1]
    
        predicted_sales.append({
            'ProductID': product_id,
            'PredictedOrderQty': forecast.values[0],
            'LastDaySales': product_sales['OrderQty'].iloc[-1],
            'PercentageIncrease': forecast.values[0] / last_day_sales
        })
    
predicted_sales_df = pd.DataFrame(predicted_sales)

print(predicted_sales_df)

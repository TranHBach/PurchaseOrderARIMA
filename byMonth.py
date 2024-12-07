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
data['Month'] = data['OrderDate'].dt.to_period('M')

# Aggregate sales data by month and product
monthly_sales = data.groupby(['Month', 'ProductID'])['OrderQty'].sum().reset_index()

predicted_sales = []
last_order_date = data['OrderDate'].max()

for product_id in monthly_sales['ProductID'].unique():
    product_sales = monthly_sales[monthly_sales['ProductID'] == product_id].copy()
    
    # Convert Month period to datetime for comparison
    product_sales['OrderDate'] = product_sales['Month'].dt.to_timestamp()
    if product_sales['OrderDate'].max() < last_order_date - timedelta(days=30):
        continue
        
    product_sales = product_sales.set_index('Month').sort_index()
    product_sales.reset_index()
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = sm.tsa.SARIMAX(endog=product_sales['OrderQty'], order=(3,1,1), disp=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=1)
        
        # Create scatter plot showing all historical data and prediction
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(product_sales)), product_sales['OrderQty'], 
                   color='blue', label='Historical Sales', marker='o')
        plt.scatter(len(product_sales), forecast.values[0], 
                   color='red', label='Prediction', marker='x', s=100)
        plt.title(f'Sales History and Prediction for Product {product_id}')
        plt.xlabel('Time Period (Months)')
        plt.ylabel('Order Quantity')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Compare to previous month's actual sales
        last_month_sales = product_sales['OrderQty'].iloc[-1]
    
        predicted_sales.append({
            'ProductID': product_id,
            'PredictedOrderQty': forecast.values[0],
            'LastMonthSales': product_sales['OrderQty'].iloc[-1],
            'PercentageIncrease': forecast.values[0] / last_month_sales
        })
    
predicted_sales_df = pd.DataFrame(predicted_sales)

print(predicted_sales_df)

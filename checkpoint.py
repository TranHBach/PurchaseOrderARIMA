import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
import pmdarima as pm
from pmdarima import model_selection

# Load the data
data = pd.read_csv('bike_sales.csv')

# Preprocess the data
data['OrderDate'] = pd.to_datetime(data['OrderDate'])
data['Week'] = data['OrderDate'].dt.isocalendar().week
data['Year'] = data['OrderDate'].dt.year

# Aggregate sales data by week and product
weekly_sales = data.groupby(['Year', 'Week', 'ProductID'])['OrderQty'].sum().reset_index()

# Prepare the data for modeling
predicted_sales = []
last_order_date = data['OrderDate'].max()
# Iterate over each product
for product_id in weekly_sales['ProductID'].unique():
    product_sales = weekly_sales[weekly_sales['ProductID'] == product_id].copy()
    # Create a datetime index from Year and Week
    product_sales['Date'] = product_sales.apply(lambda row: f"{int(row['Year'])}-W{int(row['Week']):02d}-1", axis=1)
    product_sales['Date'] = pd.to_datetime(product_sales['Date'], format="%Y-W%U-%w")
    if product_sales['Date'].max() < last_order_date - timedelta(days=10):
        continue
    product_sales = product_sales.set_index('Date').sort_index()
    product_sales.reset_index()
    product_sales.index = pd.DatetimeIndex(product_sales.index).to_period('W')
    # result = adfuller(product_sales['OrderQty'])
    # Split into train and test sets
    split_point = int(len(product_sales) * 0.8)
    train_data = product_sales.iloc[:split_point]
    test_data = product_sales.iloc[split_point:]
    model = ARIMA(train_data['OrderQty'], order=(1, 1, 0))
    model_fit = model.fit()
    
    # # Forecast the next 2 weeks
    forecast = model_fit.forecast(steps=1)
    
    # # Store the predictions
    predicted_sales.append({
        'ProductID': product_id,
        'PredictedOrderQty': forecast.sum(),
        'ActualOrderQty': test_data['OrderQty'].iloc[0]
    })

# Convert predictions to DataFrame
predicted_sales_df = pd.DataFrame(predicted_sales)

# Output the predictions
print(predicted_sales_df)

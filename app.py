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

# Load the data
data = pd.read_csv('bike_sales.csv')

# Preprocess the data
data['OrderDate'] = pd.to_datetime(data['OrderDate'])
data['Week'] = data['OrderDate'].dt.isocalendar().week
data['Year'] = data['OrderDate'].dt.year

# Aggregate sales data by week and product
weekly_sales = data.groupby(['Year', 'Week', 'ProductID'])['OrderQty'].sum().reset_index()
dictParams = {}
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
    # 1,1,1     1,1,1,12 : can predict the sudden spike
    # 3,0,2     3,0,1,5 : can predict the sudden spike better
    # 3,0,2     3,0,2,5 : can predict the sudden spike could be ok
    # 2,0,2     3,0,2,5 : can predict the sudden spike could be ok
    # 212
    # 202
    # 301 + (q + 1): Currently the best
    print("--------------------------------------------------------------");
    for p in range(0,4):
        for d in range(0,2):
            for q in range(0,3):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # Check for NaN values in train_data
                    if train_data['OrderQty'].isnull().any():
                        print(f"NaN values found in training data for product {product_id}. Skipping...")
                        continue
                    try:
                        model = sm.tsa.SARIMAX(endog=train_data['OrderQty'], order=(p,d + 1,q), seasonal_order=(p,d,q,5), disp=False, initialization='approximate_diffuse')
                        model_fit = model.fit(disp=False)
                        x = np.arange(test_data.shape[0])
                        # plt.scatter(x,test_data['OrderQty'],marker='x')
                        forecast_values = model_fit.forecast(steps=len(x))
                    # plt.scatter(x,forecast_values)
                        mse = np.square(np.subtract(test_data['OrderQty'].values, forecast_values.values)).mean()
                        print(mse, p, d, q)
                        key = f"{p}{d}{q}"
                        if key in dictParams:
                            dictParams[key] = dictParams[key] + mse
                        else:
                            dictParams[key] = mse
                    # plt.show()
                    except np.linalg.LinAlgError as e:
                        print(f"LinAlgError for product {product_id} with order ({p},{d},{q}): {e}")
                        continue
    # model_fit = model.fit()
    
    # # # Forecast the next 2 weeks
    # forecast = model_fit.forecast(steps=1)
    
    # # Store the predictions
    # predicted_sales.append({
    #     'ProductID': product_id,
    #     'PredictedOrderQty': forecast.sum(),
    #     'ActualOrderQty': test_data['OrderQty'].iloc[0]
    # })

# Convert predictions to DataFrame
print(dictParams)
# predicted_sales_df = pd.DataFrame(predicted_sales)

# # Output the predictions
# print(predicted_sales_df)
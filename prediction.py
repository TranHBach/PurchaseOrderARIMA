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
purchase_order_detail = pd.read_csv('purchaseOrderDetail.csv')
purchase_order_header = pd.read_csv('purchaseOrderHeader.csv')
purchase_order_detail = pd.merge(
    purchase_order_header,
    purchase_order_detail,
    on='PurchaseOrderID',
    how='inner'
)
purchase_order_detail['OrderDate'] = pd.to_datetime(purchase_order_detail['OrderDate'])

# print(purchase_order_detail)
bill_of_material = pd.read_csv('billOfMaterial.csv')
product_inventory = pd.read_csv('productInventory.csv')
product = pd.read_csv('product.csv')
def get_all_components(product_id, bom_data, multiplier=1, memo=None):
    if memo is None:
        memo = {}
    # If endDate is not null then it's no longer used.
    components = bom_data[
        (bom_data['ProductAssemblyID'] == product_id) & 
        (pd.isna(bom_data['EndDate']))
    ]
    
    if len(components) == 0:
        components = bom_data[bom_data['ProductAssemblyID'] == product_id]
        if len(components) == 0:
            return memo
    
    for _, row in components.iterrows():
        component_id = row['ComponentID']
        qty = row['PerAssemblyQty'] * multiplier
        memo[component_id] = memo.get(component_id, 0) + qty
        get_all_components(component_id, bom_data, qty, memo)
    return memo

data['OrderDate'] = pd.to_datetime(data['OrderDate'])
data['Week'] = data['OrderDate'].dt.isocalendar().week
data['Year'] = data['OrderDate'].dt.year
weekly_sales = data.groupby(['Year', 'Week', 'ProductID'])['OrderQty'].sum().reset_index()
last_order_date = data['OrderDate'].max()
predicted_sales = []
purchaseOrderIncrement = {}

for product_id in weekly_sales['ProductID'].unique():
    product_sales = weekly_sales[weekly_sales['ProductID'] == product_id].copy()

    product_sales['Date'] = product_sales.apply(lambda row: f"{int(row['Year'])}-W{int(row['Week']):02d}-1", axis=1)
    product_sales['Date'] = pd.to_datetime(product_sales['Date'], format="%Y-W%U-%w")
    if product_sales['Date'].max() < last_order_date - timedelta(days=7):
        continue
    product_sales = product_sales.set_index('Date').sort_index()
    product_sales.reset_index()
    product_sales.index = pd.DatetimeIndex(product_sales.index).to_period('W')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        p,d,q = 3,0,2
        model = sm.tsa.SARIMAX(endog=product_sales['OrderQty'], order=(5,d,q), seasonal_order=(1,d,q,6), disp=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=1)
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(product_sales)), product_sales['OrderQty'], 
                color='blue', label='Historical Sales', marker='o')
        plt.scatter(len(product_sales), forecast.values[0], 
                color='red', label='Prediction', marker='x', s=100)
        plt.title(f'Sales History and Prediction for Product {product_id}')
        plt.xlabel('Time Period (Weeks)')
        plt.ylabel('Order Quantity')
        plt.legend()
        plt.grid(True)
        plt.show()
    last_week_sales = product_sales['OrderQty'].iloc[-1]
    precentageIncrease = forecast.values[0] / last_week_sales
    componentList = get_all_components(product_id, bill_of_material)
    for component_id, qty in componentList.items():
        if component_id not in purchaseOrderIncrement:
            purchaseOrderIncrement[component_id] = {'count': 1, 'total_increase': precentageIncrease}
        else:
            purchaseOrderIncrement[component_id]['count'] += 1
            purchaseOrderIncrement[component_id]['total_increase'] += precentageIncrease
    
    predicted_sales.append({
        'ProductID': product_id,
        'PredictedOrderQty': forecast.values[0],
        'LastWeekSales': product_sales['OrderQty'].iloc[-1],
        'PercentageIncrease': precentageIncrease
    })
for component_id in componentList.keys():
    if component_id in purchaseOrderIncrement:
        avg_increase = purchaseOrderIncrement[component_id]['total_increase'] / purchaseOrderIncrement[component_id]['count']
        productPurchase = purchase_order_detail[purchase_order_detail['ProductID'] == component_id]
        productPurchase = productPurchase[productPurchase['OrderDate'] >= last_order_date - timedelta(days=7)]
        purchaseOrderIncrement[component_id]['purchase_order'] = productPurchase['OrderQty'].sum() * avg_increase
    else:
        purchaseOrderIncrement[component_id] = {'purchase_order': 0}
for component_id in purchaseOrderIncrement.keys():
    if 'purchase_order' in purchaseOrderIncrement[component_id]:
        print(f'Component ID: {component_id}, Purchase Order: {purchaseOrderIncrement[component_id]["purchase_order"]}')

predicted_sales_df = pd.DataFrame(predicted_sales)
predicted_sales_df.to_csv('predicted_sales.csv', index=False)
print(predicted_sales_df)

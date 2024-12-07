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

data = pd.read_csv('bike_sales.csv')
purchase_order_detail = pd.read_csv('purchaseOrderDetail.csv')
purchase_order_header = pd.read_csv('purchaseOrderHeader.csv')
bill_of_material = pd.read_csv('billOfMaterial.csv')
product_inventory = pd.read_csv('productInventory.csv')
product = pd.read_csv('product.csv')

purchase_order_detail = pd.merge(
    purchase_order_header,
    purchase_order_detail,
    on='PurchaseOrderID',
    how='inner'
)
purchase_order_detail['OrderDate'] = pd.to_datetime(purchase_order_detail['OrderDate'])

data['OrderDate'] = pd.to_datetime(data['OrderDate'])
data['Month'] = data['OrderDate'].dt.to_period('M')
monthly_sales = data.groupby(['Month', 'ProductID'])['OrderQty'].sum().reset_index()
last_order_date = data['OrderDate'].max()
predicted_sales = []
purchaseOrderIncrement = {}

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
        try:
            model = sm.tsa.SARIMAX(endog=product_sales['OrderQty'], order=(2,1,1), disp=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=1)
            # Comment these lines to disable plotting
            # plt.figure(figsize=(12, 6))
            # plt.scatter(range(len(product_sales)), product_sales['OrderQty'], 
            #     color='blue', label='Historical Sales', marker='o')
            # plt.scatter(len(product_sales), forecast.values[0], 
            #         color='red', label='Prediction', marker='x', s=100)
            # plt.title(f'Sales History and Prediction for Product {product_id}')
            # plt.xlabel('Time Period (Months)')
            # plt.ylabel('Order Quantity')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError for parameters p={2}, d={1}, q={1}: {e}")
        except Exception as e:
            print(f"Exception for parameters p={2}, d={1}, q={1}: {e}")
            
    last_month_sales = product_sales['OrderQty'].iloc[-1]
    precentageIncrease = forecast.values[0] / last_month_sales
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
        'LastMonthSales': product_sales['OrderQty'].iloc[-1],
        'PercentageIncrease': precentageIncrease * 100
    })

predicted_sales_df = pd.DataFrame(predicted_sales)
print(predicted_sales_df)
predicted_sales_df.to_csv('predicted_sales.csv', index=False)

for component_id in componentList.keys():
    if component_id in purchaseOrderIncrement:
        avg_increase = purchaseOrderIncrement[component_id]['total_increase'] / purchaseOrderIncrement[component_id]['count']
        productPurchase = purchase_order_detail[purchase_order_detail['ProductID'] == component_id]
        productPurchase = productPurchase[productPurchase['OrderDate'] >= last_order_date - timedelta(days=30)]
        purchaseOrderIncrement[component_id]['purchase_order'] = productPurchase['OrderQty'].sum() * avg_increase
    else:
        purchaseOrderIncrement[component_id] = {'purchase_order': 0}
for component_id in purchaseOrderIncrement.keys():
    if 'purchase_order' in purchaseOrderIncrement[component_id]:
        if purchaseOrderIncrement[component_id]['purchase_order'] > 0:
            print(f'Component ID: {component_id}, Purchase Order: {purchaseOrderIncrement[component_id]["purchase_order"]}')
            with open('purchase_orders_prediction.csv', 'a') as f:
                f.write(f"{component_id},{purchaseOrderIncrement[component_id]['purchase_order']}\n")


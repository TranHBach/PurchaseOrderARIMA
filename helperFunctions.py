import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Load the data
purchase_order_header = pd.read_csv('purchaseOrderHeader.csv')
purchase_order_detail = pd.read_csv('purchaseOrderDetail.csv')
bill_of_material = pd.read_csv('billOfMaterial.csv')
sales_detail = pd.read_csv('salesDetail.csv')
product_inventory = pd.read_csv('productInventory.csv')
product = pd.read_csv('product.csv')
subcategory = pd.read_csv('subCategory.csv')
category = pd.read_csv('category.csv')
ship_method = pd.read_csv('shipMethod.csv')

# First, get all products from category 1 (Bikes)
bikes_df = (product
    .merge(subcategory[['ProductSubcategoryID', 'ProductCategoryID']], 
           on='ProductSubcategoryID', how='left')
    .merge(category[['ProductCategoryID', 'Name']], 
           on='ProductCategoryID', how='left'))

# Filter only bikes (Category 1)
bike_products = bikes_df[bikes_df['ProductCategoryID'] == 1]['ProductID'].tolist()

# Filter sales data for bikes only
bike_sales = sales_detail[sales_detail['ProductID'].isin(bike_products)].copy()
bike_sales['ModifiedDate'] = pd.to_datetime(bike_sales['ModifiedDate'])

# Get the last date in our dataset
last_date = bike_sales['ModifiedDate'].max()
forecast_start = last_date + timedelta(days=1)
forecast_end = forecast_start + timedelta(days=13)

print(f"\nForecasting period: {forecast_start.strftime('%d/%m/%Y')} to {forecast_end.strftime('%d/%m/%Y')}")
def get_historical_purchases(purchase_data, product_id, days_window=90):
    """Analyze historical purchase patterns"""
    purchases = purchase_data[purchase_data['ProductID'] == product_id].copy()
    purchases['DueDate'] = pd.to_datetime(purchases['DueDate'])
    
    last_date = purchases['DueDate'].max()
    start_date = last_date - timedelta(days=days_window)
    recent_purchases = purchases[purchases['DueDate'] >= start_date]
    
    if len(recent_purchases) == 0:
        return 0, 0
    
    avg_order_size = recent_purchases['OrderQty'].mean()
    max_order_size = recent_purchases['OrderQty'].max()
    
    return avg_order_size, max_order_size

def save_cache_to_file():
    """Save both prediction caches to files"""
    # Save sales predictions
    sales_cache_df = pd.DataFrame()
    for product_id, prediction_df in sales_prediction_cache.items():
        prediction_df['product_id'] = product_id
        sales_cache_df = pd.concat([sales_cache_df, prediction_df])
    sales_cache_df.to_csv('sales_predictions_cache.csv', index=False)
    
    # Save purchase orders
    purchase_cache_df = pd.DataFrame([
        {'product_id': k, 'order_qty': v}
        for k, v in purchase_order_cache.items()
    ])
    purchase_cache_df.to_csv('purchase_orders_cache.csv', index=False)

def analyze_seasonality(sales_data, product_id):
    """Analyze sales patterns for a specific product"""
    product_sales = sales_data[sales_data['ProductID'] == product_id].copy()
    daily_sales = (product_sales.groupby('ModifiedDate')
                  .agg({'OrderQty': 'sum'})
                  .reset_index())
    
    # Calculate rolling averages
    if len(daily_sales) >= 7:
        daily_sales['7d_avg'] = daily_sales['OrderQty'].rolling(window=7, min_periods=1).mean()
    else:
        daily_sales['7d_avg'] = daily_sales['OrderQty'].mean()
    
    # Get recent average (last month if available)
    recent_sales = daily_sales.tail(30)['OrderQty'].mean()
    overall_avg = daily_sales['OrderQty'].mean()
    
    return recent_sales, overall_avg

def get_historical_average(sales_data, product_id, days_window=14):
    """Get average sales for the last N days"""
    product_sales = sales_data[sales_data['ProductID'] == product_id].copy()
    last_date = product_sales['ModifiedDate'].max()
    start_date = last_date - timedelta(days=days_window)
    
    recent_sales = product_sales[product_sales['ModifiedDate'] >= start_date]
    if len(recent_sales) == 0:
        return 0
    
    return recent_sales['OrderQty'].sum() / days_window

def predict_sales(sales_data, product_id, periods=14):
    """Predict sales for next 14 days for a specific product"""
    product_sales = sales_data[sales_data['ProductID'] == product_id].copy()
    daily_sales = (product_sales.groupby('ModifiedDate')
                  .agg({'OrderQty': 'sum'})
                  .reset_index()
                  .rename(columns={'ModifiedDate': 'ds', 'OrderQty': 'y'}))
    
    if len(daily_sales) < 2:
        return pd.DataFrame({'ds': pd.date_range(start=forecast_start, 
                                               periods=periods, freq='D'),
                           'yhat': [0] * periods})
    
    # Get historical average
    hist_avg = get_historical_average(sales_data, product_id)
    
    # Train Prophet model with very conservative settings
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.0001,  # Make it very conservative
        seasonality_prior_scale=0.01
    )
    model.fit(daily_sales)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    
    # Get only future predictions
    future_predictions = forecast[['ds', 'yhat']].tail(periods)
    
    # Use the minimum of prophet prediction and historical average
    future_predictions['yhat'] = future_predictions['yhat'].clip(lower=0, upper=hist_avg*1.2)
    
    return future_predictions

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

# Create cache dictionaries
sales_prediction_cache = {}
purchase_order_cache = {}
def load_cache_from_file():
    """Load both prediction caches from files if they exist"""
    try:
        # Load sales predictions
        sales_cache_df = pd.read_csv('sales_predictions_cache.csv')
        sales_cache_df['ds'] = pd.to_datetime(sales_cache_df['ds'])
        
        # Reconstruct sales prediction cache
        for product_id in sales_cache_df['product_id'].unique():
            product_predictions = sales_cache_df[sales_cache_df['product_id'] == product_id]
            sales_prediction_cache[product_id] = product_predictions[['ds', 'yhat']]
        
        # Load purchase orders
        purchase_cache_df = pd.read_csv('purchase_orders_cache.csv')
        for _, row in purchase_cache_df.iterrows():
            purchase_order_cache[row['product_id']] = row['order_qty']
            
        print("Cache loaded from files successfully")
    except FileNotFoundError:
        print("No cache files found - will create new predictions")

# Add at the start of your script, after creating the cache dictionaries
load_cache_from_file()

def get_cached_sales_prediction(sales_data, product_id, cache=sales_prediction_cache):
    """Get or calculate sales prediction with caching"""
    if product_id in cache:
        return cache[product_id]
    
    prediction = predict_sales(sales_data, product_id)
    cache[product_id] = prediction
    return prediction

def calculate_purchase_order(component_id, qty_needed, purchase_data, cache=purchase_order_cache):
    """Calculate reasonable purchase order quantity based on historical patterns"""
    if component_id in cache:
        return cache[component_id]
    
    avg_order, max_order = get_historical_purchases(purchase_data, component_id)
    
    if avg_order == 0:  # No historical purchase data
        # Use the qty_needed but cap it at a reasonable maximum
        order_qty = min(qty_needed, 1100)  # Default cap of 1000 units
    else:
        # Calculate number of orders needed
        num_orders = np.ceil(qty_needed / avg_order)
        # Use historical average order size
        order_qty = min(qty_needed, avg_order * 1.2)  # Allow 20% more than historical average
        
        # Never exceed historical maximum order size
        if max_order > 0:
            order_qty = min(order_qty, max_order * 1.5)  # Allow 50% more than historical maximum
    
    cache[component_id] = order_qty
    return order_qty

def calculate_component_cost(component_id, purchase_data):
    """Calculate average unit cost for a component from purchase history"""
    component_purchases = purchase_data[purchase_data['ProductID'] == component_id]
    if len(component_purchases) == 0:
        return 0
    return component_purchases['UnitPrice'].mean()

# Modify the prediction loop to use caching
all_predictions = {}
print("\nPredicted Sales for Next 2 Weeks:")
print("===================================")
total_bikes = 0
for bike_id in bike_products:
    # Use cached predictions instead of direct calculation
    predictions = get_cached_sales_prediction(bike_sales, bike_id)
    total_predicted = predictions['yhat'].sum()
    if total_predicted > 0:
        predicted_qty = round(total_predicted)
        all_predictions[bike_id] = predicted_qty
        total_bikes += predicted_qty
        print(f"Product {bike_id}: {predicted_qty} units")

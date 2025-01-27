import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the transactions data
transactions_df = pd.read_csv('data/Transactions.csv')

# Read all datasets
customers_df = pd.read_csv('data/Customers.csv')
products_df = pd.read_csv('data/Products.csv')

# Merge datasets for richer insights
transactions_with_products = transactions_df.merge(
    products_df, on='ProductID', how='left'
)
full_data = transactions_with_products.merge(
    customers_df, on='CustomerID', how='left'
)

# Basic data exploration
print("\n=== Basic Dataset Information ===")
print(transactions_df.info())
print("\n=== Summary Statistics ===")
print(transactions_df.describe())

# Time series analysis
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
transactions_df['Month'] = transactions_df['TransactionDate'].dt.month
transactions_df['DayOfWeek'] = transactions_df['TransactionDate'].dt.dayofweek

# Monthly sales analysis
monthly_sales = transactions_df.groupby('Month')['TotalValue'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.bar(monthly_sales['Month'], monthly_sales['TotalValue'])
plt.title('Monthly Sales Distribution')
plt.xlabel('Month')
plt.ylabel('Total Sales Value')
plt.savefig('task1_eda/monthly_sales.png')
plt.close()

# Product analysis
top_products = transactions_df.groupby('ProductID').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum'
}).sort_values('TotalValue', ascending=False).head(10)
print("\n=== Top 10 Products by Sales Value ===")
print(top_products)

# Customer analysis
customer_segments = transactions_df.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'TotalValue': 'sum',
    'Quantity': 'sum'
}).reset_index()

customer_segments = customer_segments.rename(columns={
    'TransactionID': 'NumberOfTransactions',
    'TotalValue': 'TotalSpent',
    'Quantity': 'TotalQuantity'
})

print("\n=== Customer Segments Summary ===")
print(customer_segments.describe())

# Quantity distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=transactions_df, x='Quantity', bins=20)
plt.title('Distribution of Order Quantities')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.savefig('task1_eda/quantity_distribution.png')
plt.close()

# Save insights to a file
with open('task1_eda/business_insights.txt', 'w') as f:
    f.write("Business Insights from EDA:\n\n")
    
    # Insight 1: Transaction Patterns
    avg_transaction = transactions_df['TotalValue'].mean()
    f.write(f"1. Average Transaction Value: ${avg_transaction:.2f}\n")
    
    # Insight 2: Popular Products
    f.write("\n2. Top Selling Products:\n")
    for idx, row in top_products.head().iterrows():
        f.write(f"   - Product {idx}: ${row['TotalValue']:.2f} in sales\n")
    
    # Insight 3: Customer Behavior
    avg_customer_transactions = customer_segments['NumberOfTransactions'].mean()
    f.write(f"\n3. Average Transactions per Customer: {avg_customer_transactions:.2f}\n")
    
    # Insight 4: Quantity Patterns
    most_common_quantity = transactions_df['Quantity'].mode()[0]
    f.write(f"\n4. Most Common Order Quantity: {most_common_quantity} units\n")
    
    # Insight 5: Monthly Trends
    peak_month = monthly_sales.loc[monthly_sales['TotalValue'].idxmax(), 'Month']
    f.write(f"\n5. Peak Sales Month: Month {peak_month}\n")

# Category analysis
category_performance = full_data.groupby('Category').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'TransactionID': 'count'
}).reset_index()

# Regional analysis
regional_sales = full_data.groupby('Region').agg({
    'TotalValue': 'sum',
    'CustomerID': 'nunique'
}).reset_index()

# Visualize category performance
plt.figure(figsize=(12, 6))
sns.barplot(data=category_performance, x='Category', y='TotalValue')
plt.title('Sales by Product Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('TASK1/THIPPARTHI_VIGNESH_category_sales.png')
plt.close()

# Additional insights in the text file
with open('TASK1/THIPPARTHI_VIGNESH_EDA.pdf', 'a') as f:
    f.write("\n\n=== Additional Insights ===\n")
    
    # Category insights
    top_category = category_performance.loc[
        category_performance['TotalValue'].idxmax(), 'Category'
    ]
    f.write(f"\n6. Best Performing Category: {top_category}\n")
    
    # Regional insights
    top_region = regional_sales.loc[
        regional_sales['TotalValue'].idxmax(), 'Region'
    ]
    f.write(f"\n7. Highest Revenue Region: {top_region}\n") 
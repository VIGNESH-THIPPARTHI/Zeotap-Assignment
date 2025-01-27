import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Read the datasets
transactions_df = pd.read_csv('data/Transactions.csv')
customers_df = pd.read_csv('data/Customers.csv')
products_df = pd.read_csv('data/Products.csv')

# Create customer feature vectors
def create_customer_features():
    # Transaction-based features
    customer_features = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',  # Number of transactions
        'TotalValue': ['sum', 'mean'],  # Total spend and average spend
        'Quantity': ['sum', 'mean'],  # Total quantity and average quantity
        'ProductID': 'nunique'  # Number of unique products bought
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['CustomerID', 'transaction_count', 'total_spend', 
                               'avg_spend', 'total_quantity', 'avg_quantity', 
                               'unique_products']
    
    # Add customer region (one-hot encoded)
    region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
    customer_features = customer_features.merge(
        pd.concat([customers_df['CustomerID'], region_dummies], axis=1),
        on='CustomerID',
        how='left'
    )
    
    # Calculate days since signup
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    customers_df['days_since_signup'] = (
        pd.Timestamp.now() - customers_df['SignupDate']
    ).dt.days
    
    customer_features = customer_features.merge(
        customers_df[['CustomerID', 'days_since_signup']],
        on='CustomerID',
        how='left'
    )
    
    return customer_features

# Create product category preferences
def add_category_preferences(customer_features):
    # Merge transactions with products to get categories
    trans_with_cats = transactions_df.merge(
        products_df[['ProductID', 'Category']], 
        on='ProductID',
        how='left'
    )
    
    # Calculate category preferences
    category_preferences = pd.crosstab(
        trans_with_cats['CustomerID'],
        trans_with_cats['Category'],
        values=trans_with_cats['TotalValue'],
        aggfunc='sum',
        normalize='index'
    ).fillna(0)
    
    # Merge with customer features
    return customer_features.merge(
        category_preferences,
        left_on='CustomerID',
        right_index=True,
        how='left'
    )

# Find lookalikes
def find_lookalikes(customer_id, feature_matrix, n_recommendations=3):
    # Get index of customer
    customer_index = feature_matrix.index[
        feature_matrix['CustomerID'] == customer_id
    ][0]
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(
        feature_matrix.drop('CustomerID', axis=1)
    )[customer_index]
    
    # Get indices of top similar customers (excluding self)
    similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
    
    # Get customer IDs and scores
    similar_customers = [
        (feature_matrix.iloc[idx]['CustomerID'], similarity_scores[idx])
        for idx in similar_indices
    ]
    
    return similar_customers

# Main execution
def main():
    # Create feature matrix
    customer_features = create_customer_features()
    customer_features = add_category_preferences(customer_features)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        customer_features.drop('CustomerID', axis=1)
    )
    
    # Create final feature matrix
    feature_matrix = pd.DataFrame(
        scaled_features,
        columns=customer_features.columns[1:],
    )
    feature_matrix['CustomerID'] = customer_features['CustomerID']
    
    # Generate lookalikes for first 20 customers
    results = []
    for cust_id in customers_df['CustomerID'].head(20):
        lookalikes = find_lookalikes(cust_id, feature_matrix)
        results.append({
            'customer_id': cust_id,
            'lookalikes': [
                {'similar_customer': cust, 'similarity_score': score}
                for cust, score in lookalikes
            ]
        })
    
    # Save results to CSV
    output_data = []
    for result in results:
        cust_id = result['customer_id']
        lookalikes_str = ';'.join([
            f"{l['similar_customer']}:{l['similarity_score']:.3f}"
            for l in result['lookalikes']
        ])
        output_data.append([cust_id, lookalikes_str])
    
    # Update the output path
    pd.DataFrame(
        output_data,
        columns=['customer_id', 'lookalikes']
    ).to_csv('TASK2/THIPPARTHI_VIGNESH_Lookalike.csv', index=False)

if __name__ == "__main__":
    main() 
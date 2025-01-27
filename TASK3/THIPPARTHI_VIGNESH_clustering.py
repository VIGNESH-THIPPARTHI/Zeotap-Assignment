import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer

# Read and prepare data
def prepare_data():
    # Read datasets
    transactions_df = pd.read_csv('data/Transactions.csv')
    customers_df = pd.read_csv('data/Customers.csv')
    products_df = pd.read_csv('data/Products.csv')
    
    # Create customer features
    customer_features = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': ['sum', 'mean', 'std'],
        'Quantity': ['sum', 'mean'],
        'ProductID': 'nunique'
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['CustomerID', 'transaction_count', 
                               'total_spend', 'avg_spend', 'spend_std',
                               'total_quantity', 'avg_quantity', 
                               'unique_products']
    
    # Add recency and customer tenure features
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    customers_df['days_since_signup'] = (
        pd.Timestamp.now() - customers_df['SignupDate']
    ).dt.days
    
    # One-hot encode region
    region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
    
    # Merge all features
    final_features = customer_features.merge(
        customers_df[['CustomerID', 'days_since_signup']],
        on='CustomerID',
        how='left'
    )
    
    final_features = pd.concat([
        final_features,
        region_dummies
    ], axis=1)
    
    # Fill NaN values with 0 for region columns
    region_columns = [col for col in final_features.columns if col.startswith('region_')]
    final_features[region_columns] = final_features[region_columns].fillna(0)
    
    return final_features

# Evaluate different numbers of clusters
def evaluate_clusters(X, max_clusters=10):
    n_clusters_range = range(2, max_clusters + 1)
    db_scores = []
    silhouette_scores = []
    inertia_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        db_scores.append(davies_bouldin_score(X, labels))
        silhouette_scores.append(silhouette_score(X, labels))
        inertia_scores.append(kmeans.inertia_)
        
    return db_scores, silhouette_scores, inertia_scores

# Visualize clustering results
def plot_clustering_metrics(db_scores, silhouette_scores, inertia_scores):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Davies-Bouldin Index
    ax1.plot(range(2, len(db_scores) + 2), db_scores, marker='o')
    ax1.set_title('Davies-Bouldin Index')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Score')
    
    # Silhouette Score
    ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Score')
    
    # Elbow Plot
    ax3.plot(range(2, len(inertia_scores) + 2), inertia_scores, marker='o')
    ax3.set_title('Elbow Plot')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Inertia')
    
    plt.tight_layout()
    plt.savefig('TASK3/THIPPARTHI_VIGNESH_clustering_metrics.png')
    plt.close()

# Visualize final clusters
def plot_clusters(X, labels, n_clusters):
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'Customer Segments ({n_clusters} clusters)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig('TASK3/THIPPARTHI_VIGNESH_cluster_visualization.png')
    plt.close()

def main():
    # Prepare data
    print("Preparing data...")
    features_df = prepare_data()
    
    # Handle missing values
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    
    # Select numeric columns
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    X = features_df[numeric_columns].values
    
    # Impute missing values
    X = imputer.fit_transform(X)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Evaluate different numbers of clusters
    print("Evaluating clustering metrics...")
    db_scores, silhouette_scores, inertia_scores = evaluate_clusters(X)
    
    # Plot evaluation metrics
    plot_clustering_metrics(db_scores, silhouette_scores, inertia_scores)
    
    # Find optimal number of clusters (minimum DB score)
    optimal_clusters = db_scores.index(min(db_scores)) + 2
    print(f"\nOptimal number of clusters: {optimal_clusters}")
    print(f"Best Davies-Bouldin Index: {min(db_scores):.3f}")
    
    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = final_kmeans.fit_predict(X)
    
    # Visualize final clusters
    plot_clusters(X, labels, optimal_clusters)
    
    # Save clustering results
    results_df = pd.DataFrame({
        'CustomerID': features_df['CustomerID'],
        'Cluster': labels
    })
    results_df.to_csv('TASK3/THIPPARTHI_VIGNESH_cluster_assignments.csv', index=False)
    
    # Generate clustering report
    with open('TASK3/THIPPARTHI_VIGNESH_clustering_report.txt', 'w') as f:
        f.write("Customer Segmentation Analysis Report\n")
        f.write("===================================\n\n")
        
        f.write(f"Number of clusters: {optimal_clusters}\n")
        f.write(f"Davies-Bouldin Index: {min(db_scores):.3f}\n")
        f.write(f"Silhouette Score: {silhouette_scores[optimal_clusters-2]:.3f}\n\n")
        
        f.write("Cluster Sizes:\n")
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        for cluster, size in cluster_sizes.items():
            f.write(f"Cluster {cluster}: {size} customers\n")
        
        # Add cluster characteristics
        f.write("\nCluster Characteristics:\n")
        features_df['Cluster'] = labels
        
        # Select only numeric columns for mean calculation
        numeric_cols = ['transaction_count', 'total_spend', 'avg_spend', 
                       'spend_std', 'total_quantity', 'avg_quantity', 
                       'unique_products', 'days_since_signup']
        
        cluster_means = features_df[numeric_cols + ['Cluster']].groupby('Cluster').mean()
        
        for cluster in range(optimal_clusters):
            f.write(f"\nCluster {cluster}:\n")
            f.write("- Average spend: ${:.2f}\n".format(
                cluster_means.loc[cluster, 'avg_spend']
            ))
            f.write("- Transaction count: {:.1f}\n".format(
                cluster_means.loc[cluster, 'transaction_count']
            ))
            f.write("- Customer tenure: {:.1f} days\n".format(
                cluster_means.loc[cluster, 'days_since_signup']
            ))
            f.write("- Total quantity: {:.1f}\n".format(
                cluster_means.loc[cluster, 'total_quantity']
            ))
            f.write("- Unique products: {:.1f}\n".format(
                cluster_means.loc[cluster, 'unique_products']
            ))

if __name__ == "__main__":
    main() 
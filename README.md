# Customer Analytics Project

## Project Overview
This project analyzes customer data to provide business insights through three main tasks:
1. Exploratory Data Analysis (EDA)
2. Customer Lookalike Model
3. Customer Segmentation

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### 2. Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. Create a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
project/
│
├── TASK1/
│   ├── THIPPARTHI_VIGNESH_eda.py           # EDA analysis script
│   └── THIPPARTHI_VIGNESH_EDA.pdf          # EDA report
│
├── TASK2/
│   ├── THIPPARTHI_VIGNESH_.py              # Lookalike model script
│   └── THIPPARTHI_VIGNESH_Lookalike.csv    # Model results
│
├── TASK3/
│   ├── THIPPARTHI_VIGNESH_clustering.py     # Clustering script
│   ├── THIPPARTHI_VIGNESH_clustering_report.txt  # Detailed clustering analysis
│   ├── THIPPARTHI_VIGNESH_cluster_assignments.csv # Cluster assignments
│   ├── THIPPARTHI_VIGNESH_clustering_metrics.png  # Evaluation metrics plot
│   └── THIPPARTHI_VIGNESH_cluster_visualization.png # Cluster visualization
│
└── data/                                    # Input data files
    ├── Transactions.csv
    ├── Customers.csv
    └── Products.csv
```

## Running the Analysis

### 1. EDA Analysis
```bash
python TASK1/THIPPARTHI_VIGNESH_eda.py
```
Outputs:
- THIPPARTHI_VIGNESH_EDA.pdf: Detailed analysis report
- Various visualization plots

### 2. Lookalike Model
```bash
python TASK2/THIPPARTHI_VIGNESH_.py
```
Output:
- THIPPARTHI_VIGNESH_Lookalike.csv: Similar customer recommendations

### 3. Customer Segmentation
```bash
python TASK3/THIPPARTHI_VIGNESH_clustering.py
```
Outputs:
- clustering_report.txt: Detailed segmentation analysis
- cluster_assignments.csv: Cluster labels for each customer
- clustering_metrics.png: Evaluation metrics visualization
- cluster_visualization.png: 2D visualization of clusters

## Troubleshooting
1. If you get a "module not found" error:
   - Make sure you've installed all requirements: `pip install -r requirements.txt`
   - Check that you're in the project root directory

2. If you get a file not found error:
   - Verify that the data files are in the data/ directory
   - Make sure you're running scripts from the project root directory

## Data Files Description
- Transactions.csv: Customer purchase history
- Customers.csv: Customer demographic information
- Products.csv: Product catalog and details #   Z e o t a p - A s s i g n m e n t  
 
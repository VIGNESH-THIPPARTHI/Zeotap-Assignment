# Customer Analytics Project

## Project Overview
This project analyzes customer data to provide business insights through three main tasks:

1. **Exploratory Data Analysis (EDA)**
2. **Customer Lookalike Model**
3. **Customer Segmentation**

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```plaintext
project/
│
├── TASK1/
│   ├── THIPPARTHI_VIGNESH_eda.py             # EDA analysis script
│   └── THIPPARTHI_VIGNESH_EDA.pdf            # EDA report
│
├── TASK2/
│   ├── THIPPARTHI_VIGNESH_.py                # Lookalike model script
│   └── THIPPARTHI_VIGNESH_Lookalike.csv      # Model results
│
├── TASK3/
│   ├── THIPPARTHI_VIGNESH_clustering.py      # Clustering script
│   ├── THIPPARTHI_VIGNESH_clustering_report.txt  # Detailed clustering analysis
│   ├── THIPPARTHI_VIGNESH_cluster_assignments.csv # Cluster assignments
│   ├── THIPPARTHI_VIGNESH_clustering_metrics.png  # Evaluation metrics plot
│   └── THIPPARTHI_VIGNESH_cluster_visualization.png # Cluster visualization
│
└── data/                                     # Input data files
    ├── Transactions.csv
    ├── Customers.csv
    └── Products.csv
```

---

## Running the Analysis

### 1. Exploratory Data Analysis (EDA)

Run the following command to perform EDA:

```bash
python TASK1/THIPPARTHI_VIGNESH_eda.py
```

**Outputs:**
- `THIPPARTHI_VIGNESH_EDA.pdf`: Detailed analysis report
- Various visualization plots

---

### 2. Customer Lookalike Model

Run the following command to generate lookalike customer recommendations:

```bash
python TASK2/THIPPARTHI_VIGNESH_.py
```

**Output:**
- `THIPPARTHI_VIGNESH_Lookalike.csv`: Similar customer recommendations

---

### 3. Customer Segmentation

Run the following command to perform customer segmentation:

```bash
python TASK3/THIPPARTHI_VIGNESH_clustering.py
```

**Outputs:**
- `THIPPARTHI_VIGNESH_clustering_report.txt`: Detailed segmentation analysis
- `THIPPARTHI_VIGNESH_cluster_assignments.csv`: Cluster labels for each customer
- `THIPPARTHI_VIGNESH_clustering_metrics.png`: Evaluation metrics visualization
- `THIPPARTHI_VIGNESH_cluster_visualization.png`: 2D visualization of clusters

---

## Data Files Description

- `Transactions.csv`: Customer purchase history
- `Customers.csv`: Customer demographic information
- `Products.csv`: Product catalog and details

---

## Troubleshooting

1. **Module not found error**:
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Verify you are in the project root directory

2. **File not found error**:
   - Confirm that the data files are located in the `data/` directory
   - Ensure scripts are executed from the project root directory

---

## Contributing

Feel free to submit issues or pull requests to enhance the project. For major changes, please open an issue first to discuss your ideas.

# Customer Segmentation using K-Means & Hierarchical Clustering

## Overview

This project was completed as **Week 4** of my AI/ML Internship and focuses on **Unsupervised Machine Learning** for customer segmentation. Unlike supervised learning, there is no target variable to predict. Instead, clustering techniques are used to discover hidden patterns and group customers based on their credit card usage behavior.

Two clustering algorithms were implemented and compared:

- **K-Means Clustering**
- **Agglomerative Hierarchical Clustering**

The project demonstrates how customer segmentation can help financial institutions improve marketing strategies, personalize customer experiences, manage risk, and make data-driven business decisions.

---

## Dataset

**Dataset:** [Credit Card Dataset for Clustering (CC GENERAL.csv)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

The dataset contains behavioral information for approximately **9,000 active credit card customers** over a six-month period. It consists of **18 numerical features** describing customer spending habits, payment behavior, credit utilization, and account activity.

Some important features include:

- **BALANCE** – Current account balance
- **PURCHASES** – Total purchase amount
- **ONEOFF_PURCHASES** – One-time purchase amount
- **INSTALLMENTS_PURCHASES** – Purchases made in installments
- **CASH_ADVANCE** – Cash advance amount
- **PURCHASES_FREQUENCY** – Frequency of purchases
- **CREDIT_LIMIT** – Assigned credit limit
- **PAYMENTS** – Total payments made
- **MINIMUM_PAYMENTS** – Minimum payment amount
- **TENURE** – Length of customer relationship

---

## Project Workflow

### 1. Data Preprocessing

- Loaded and explored the dataset.
- Removed the `CUST_ID` column because it is only a customer identifier.
- Identified missing values in `CREDIT_LIMIT` and `MINIMUM_PAYMENTS`.
- Imputed missing values using the **median** to preserve data distribution.
- Standardized all numerical features using **StandardScaler** before clustering.

---

### 2. K-Means Clustering

- Applied K-Means clustering for values of **k = 2 to 10**.
- Calculated **Inertia (Within-Cluster Sum of Squares)**.
- Generated the **Elbow Curve**.
- Calculated the **Silhouette Score** for each value of k.
- Selected the optimal number of clusters (**k = 3**).
- Assigned cluster labels to each customer.

---

### 3. Cluster Profiling

Each cluster was analyzed by calculating the mean value of every feature and visualized using a heatmap.

Three meaningful customer segments were identified:

- **High Spenders & Reward Seekers**
  - High purchase frequency
  - Large payment amounts
  - Higher balances

- **Low Activity Customers**
  - Low purchases
  - Low balances
  - Minimal credit card usage

- **Cash Advance Users**
  - Heavy cash advance usage
  - Moderate credit limits
  - Different spending behavior compared to other groups

---

### 4. PCA Visualization

Principal Component Analysis (PCA) was used to reduce the dataset into two principal components, allowing the clusters to be visualized in two-dimensional space.

---

### 5. Hierarchical Clustering

- Randomly sampled **300 observations** for efficient computation.
- Generated a **Dendrogram** using Ward linkage.
- Applied **Agglomerative Clustering**.
- Compared Hierarchical Clustering with K-Means using a cross-tabulation.
- Evaluated similarities and differences between both clustering methods.

---

## Results

- Both the **Elbow Method** and **Silhouette Score** suggested that **3 clusters** provided the best segmentation.
- Hierarchical Clustering produced results that closely matched the K-Means clusters.
- Customer groups were clearly distinguishable based on purchasing behavior and credit usage.
- K-Means proved to be the more practical algorithm for large datasets because of its speed, scalability, and ease of interpretation.

---

## Business Value

Customer segmentation enables businesses to:

- Identify high-value customers
- Personalize marketing campaigns
- Improve customer retention
- Optimize credit limit strategies
- Detect customers with higher financial risk
- Develop targeted financial products and promotional offers

Example business actions:

- **High Spenders:** Offer premium rewards and higher credit limits.
- **Low Activity Customers:** Encourage engagement through cashback offers and promotional campaigns.
- **Cash Advance Users:** Recommend personal loan products or financial planning services to reduce dependence on cash advances.

---

## Visualizations

The notebook includes the following visualizations:

- Missing Value Analysis
- Elbow Curve
- Silhouette Score Plot
- Cluster Heatmap
- PCA Cluster Visualization
- Hierarchical Clustering Dendrogram
- K-Means vs Hierarchical Cluster Comparison

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Hasan-zaman999/ITSimpleraInternship.git

cd "ITSimpleraInternship/Customer Segmentation using K-Means & Hierarchical Clustering"
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/week4_clustering.ipynb
```

---

## Repository Structure

```text
Customer Segmentation using K-Means & Hierarchical Clustering/
│
├── Data/
│   └── CC GENERAL.csv
│
├── notebooks/
│   └── week4_clustering.ipynb
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Apply DBSCAN for density-based clustering.
- Experiment with Gaussian Mixture Models.
- Build an interactive dashboard for cluster exploration.
- Automate cluster profiling and reporting.
- Compare additional clustering evaluation metrics.

---

## Author

**Hasan Zaman**

QC Analyst | AI/ML Intern | Chemistry Graduate | Machine Learning Enthusiast

- **GitHub:** https://github.com/Hasan-zaman999/
- **LinkedIn:** https://www.linkedin.com/in/hasan-z-aa1b85293/

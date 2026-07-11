# Steel Industry Energy Consumption Analysis & Predictive Modeling

This repository contains a comprehensive, two-part machine learning workflow focused on analyzing and predicting energy consumption within a real steel manufacturing plant. The project covers deep exploratory data analysis (EDA), advanced feature engineering, and the establishment of robust baseline regression models to forecast energy usage (Usage_kWh).

## 📁 Repository Structure

```text
├── data/
│   └── steel_industry_energy_consumption.csv   # Raw dataset folder
├── Energy_Consumption_EDA.ipynb                # Part 1: Exploratory Data Analysis & Feature Engineering
├── Energy_Consumption_Models.ipynb             # Part 2: Baseline Regression Modeling
├── requirement.txt                             # Project dependencies
└── readme.md                                   # Project documentation (this file)

📊 Part 1: Deep Exploratory Analysis & Feature Engineering
In the first phase (Energy_Consumption_EDA.ipynb), we thoroughly investigated the dataset structure, addressed data quality issues, and engineered key temporal and electrical domain features:

Temporal Engineering: Extracted hour_of_day, day_of_week, month, and an is_weekend binary indicator from the raw timestamp.

Domain-Specific Features: Engineered Power_Factor_Ratio by mapping leading vs. lagging current power factor metrics.

Target Labeling: Created a High_Load indicator to mark observations above the 75th percentile of energy usage.

Outlier Mitigation & Cleaning: Identified operational spikes using the Interquartile Range (IQR) method and safely handled missing value points via median imputation to safeguard downstream models.

Key EDA Insights
The Power Factor Connection: A strong correlation heatmap profile revealed that power factor metrics (Lagging_Current_Power_Factor and Leading_Current_Power_Factor) serve as primary linear indicators of load intensity.

Temporal Cyclicality: Energy usage exhibits definitive cyclical curves, peaking strongly during standard daytime manufacturing operational shifts and plunging overnight.

Load Type Discrepancies: Average kilowatt-hour consumption exhibits clean, step-like increases when transitioning sequentially across Light_Load, Medium_Load, and Maximum_Load boundaries.

🤖 Part 2: Baseline Regression Modeling
In the second phase (Energy_Consumption_Models.ipynb), we built an end-to-end predictive modeling pipeline. Categorical columns were transformed using One-Hot Encoding to maintain accurate structural relationships without imposing false ordinal hierarchies. The dataset was partitioned into an 80/20 train-test split using a fixed random_state=42 for exact reproducibility.

We trained and contrasted four diverse regression frameworks:

Linear Regression

Ridge Regression

Decision Tree Regressor

Random Forest Regressor


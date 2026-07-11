# Steel Industry Energy Consumption Analysis & Predictive Modeling

This repository contains a comprehensive, two-part machine learning workflow focused on analyzing and predicting energy consumption within a real steel manufacturing plant. The project covers deep exploratory data analysis (EDA), advanced feature engineering, and the establishment of robust baseline regression models to forecast energy usage (Usage_kWh).

## 📁 Repository Structure

## 📁 Repository Structure

```text
├── data/
│   ├── Steel_industry_data.csv                       # First raw data file
│   └── dataset_part2.csv                       # Second raw data file
├── Energy_Consumption_EDA.ipynb                # Part 1: Exploratory Data Analysis & Feature Engineering
├── Energy_Consumption_Models.ipynb             # Part 2: Baseline Regression Modeling
├── requirement.txt                             # Project dependencies
└── readme.md                                   # Project documentation (this file)
```
## 📊 Part 1: Deep Exploratory Analysis & Feature Engineering
In the first phase (Energy_Consumption_EDA.ipynb), we thoroughly investigated the dataset structure, addressed data quality issues, and engineered key temporal and electrical domain features:

1. Temporal Engineering: Extracted hour_of_day, day_of_week, month, and an is_weekend binary indicator from the raw timestamp.

2. Domain-Specific Features: Engineered Power_Factor_Ratio by mapping leading vs. lagging current power factor metrics.

3. Target Labeling: Created a High_Load indicator to mark observations above the 75th percentile of energy usage.

4. Outlier Mitigation & Cleaning: Identified operational spikes using the Interquartile Range (IQR) method and safely handled missing value points via median imputation to safeguard downstream models.

## Key EDA Insights
**The Power Factor Connection:** A strong correlation heatmap profile revealed that power factor metrics (Lagging_Current_Power_Factor and Leading_Current_Power_Factor) serve as primary linear indicators of load intensity.

**Temporal Cyclicality:** Energy usage exhibits definitive cyclical curves, peaking strongly during standard daytime manufacturing operational shifts and plunging overnight.

**Load Type Discrepancies:** Average kilowatt-hour consumption exhibits clean, step-like increases when transitioning sequentially across Light_Load, Medium_Load, and Maximum_Load boundaries.

## 🤖 Part 2: Baseline Regression Modeling
In the second phase (Energy_Consumption_Models.ipynb), we built an end-to-end predictive modeling pipeline. Categorical columns were transformed using One-Hot Encoding to maintain accurate structural relationships without imposing false ordinal hierarchies. The dataset was partitioned into an 80/20 train-test split using a fixed random_state=42 for exact reproducibility.

We trained and contrasted four diverse regression frameworks:

1. Linear Regression

2. Ridge Regression

3. Decision Tree Regressor

4. Random Forest Regressor


## 📈 Model Evaluation Metrics

| Regression Model | Test MAE | Test RMSE | Test R-squared | 5-Fold CV Mean RMSE | Status / Evaluation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | 2.6339 | 4.1460 | 0.9849 | 4.6048 | Underfitting; struggles with non-linear variations. |
| **Ridge Regression** | 4.3604 | 6.2666 | 0.9655 | 6.6866 | Underfitting; linear regularization bounds are too rigid. |
| **Decision Tree** | 0.5500 | 1.5182 | 0.9980 | 2.6124 | Severe Overfitting; memorizes specific training slices. |
| **Random Forest** | **0.3577** | **1.0318** | **0.9991** | **2.2127** | **Best Model; exceptional generalization & stability.** |


## Key Modeling Takeaways
**Non-Linear Dominance:** Linear and Ridge configurations fail to successfully model severe industrial spikes. Tree-based architectures capture these complex interaction surfaces seamlessly.

**Overfitting Discovery:** Single Decision Trees exhibit high variance—their performance degrades noticeably when subjected to 5-Fold Cross-Validation (RMSE increases from 1.5182 to 2.6124).

**The Selected Path Forward:** The Random Forest Regressor yields the lowest generalization error profile. Combining predictions from an ensemble of estimators successfully penalizes high-variance tree fluctuations, making it the superior architecture for factory load deployment.

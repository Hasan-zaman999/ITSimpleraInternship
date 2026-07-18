# Steel Industry Energy Consumption Analytics & Inference Infrastructure

An end-to-end Machine Learning and web infrastructure project built as part of an ML Engineering curriculum. This repository transitions an experimental data science workflow out of Jupyter Notebooks into a production-ready, self-contained web application powered by **FastAPI**, **Scikit-Learn (PCA + Random Forest Pipeline)**, and a **Bootstrap 5** frontend dashboard.

---

## 📂 Project Directory Structure

The repository is structured to mirror professional software engineering layouts, strictly decoupling training data, analytical notebooks, frontend assets, and production execution endpoints:

```text
├── Data/
│   └── steel_industry_engineered.csv  # The preprocessed/engineered dataset matrix
├── Notebook/
│   └── Week3_PCA.ipynb                # Comprehensive EDA, feature engineering, and pipeline export
├── Static/
│   ├── correlation_heatmap.png        # Exported analytical visualizations
│   ├── load_type_chart.png
│   └── weekly_trend_chart.png
├── templates/
│   ├── layout.html                    # Core Bootstrap 5 master template UI
│   ├── home.html                      # Application landing page
│   ├── dashboard.html                 # Embedded analytical visualization board
│   └── predict.html                   # Live inference calculation engine
├── main.py                            # Production FastAPI server orchestration logic
├── requirements.txt                   # Curated deployment dependencies
└── README.md                          # Project documentation / write-up (This File)
```

## ⚡ Key Architecture Components
## 📈 Part 1: Dimensionality Reduction Pipeline (PCA)

**Robust Feature Engineering:** Implements time-series extracted variables (NSM, hour_of_day, is_weekend, Power_Factor_Ratio) alongside categorical one-hot encoded operations.

***Strict Data Leaking Controls:** Enforces distinct train/test pipelines, ensuring scale parameters are computed exclusively on the training matrix.

***Principal Component Analysis:** Dynamically calculates component variance thresholds to compress high-dimensional feature spaces down to the optimal vectors representing 95% data variance.

***Model Serialization:** Exports the final sequential execution chain (StandardScaler -> PCA -> RandomForestRegressor) into a highly optimized binary file via joblib.

## 🌐 Part 2: FastAPI Production Server

***Asynchronous Execution Endpoints:** Utilizes FastAPI to parse, process, and return live ML calculations instantly.

***Dynamic Sparse Matrix Mapping:** Intercepts raw browser form submissions and maps inputs directly back to the exact dimensional orientation (20-column layout signature) the model demands.

***Template Render Processing:** Implements Jinja2 rendering environments to transition analytics into beautiful, web-accessible dashboard sections.

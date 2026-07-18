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

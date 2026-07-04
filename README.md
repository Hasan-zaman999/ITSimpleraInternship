# Online Retail II - Exploratory Data Analysis (EDA)

## Project Overview
This repository contains a full step-by-step Exploratory Data Analysis (EDA) on the **Online Retail II Dataset** hosted by the UCI Machine Learning Repository[span_3](start_span)[span_3](end_span). The dataset covers real transaction data from a UK-based online retailer between 2009 and 2011[span_4](start_span)[span_4](end_span). The objective of this task is to understand customer purchasing behaviors, spot data quality issues, identify sales trends over time, and provide actionable business insights before investing in a machine learning recommendation system[span_5](start_span)[span_5](end_span).

## Key Deliverables
* **Google Colab Notebook:** Contains fully structured, cleanly documented code written in sequential cells[span_6](start_span)[span_6](end_span).
* **Business Insights:** Comprehensive strategic takeaways evaluating data anomalies, domestic market concentration, and product performance metrics.

---

## Technical Architecture & Implementation

### 1. Data Loading & Profiling
* **Shape:** 525,461 rows and 8 initial features.
* **Attributes Studied:** Invoice number, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, and Country.
* **Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`[span_7](start_span)[span_7](end_span).

### 2. Data Quality & Profiling Check
* **Missing Values:** Identified major gaps in `Customer ID` (20.54% missing rows) and minor gaps in `Description` (0.56% missing rows).
* **Duplicates:** Detected and reported 6,865 duplicate entries.
* **Data Behavior (Outliers):** Utilized Box Plots to isolate heavy transaction spreads, highlighting critical operational returns and cancellations represented by extreme negative numbers in both price and quantity metrics.

### 3. Core Visualizations & Analytics Included
* **Top 10 Product Analysis:** Evaluated by total quantity sold vs. total gross revenue generated.
* **Geographical Distribution:** Visualized country performance showing extreme concentration in the United Kingdom.
* **Temporal Patterns:** Plotted line trends illustrating tight monthly revenue behaviors and Q4 holiday spikes.
* **Statistical Correlations:** Plotted a Seaborn heatmap establishing relational dependencies between numeric attributes.

---

## Key Business Insights Derived

1. **Market Concentration:** The business is heavily dependent on the UK economic landscape, making up over £8.19M of total revenue. Core retention strategies must stay domestic, while low-friction localized acquisition campaigns should target key European pipelines like EIRE and the Netherlands.
2. **Untracked Customers:** A 20.54% missing user ID metric flags a lack of persistent data for machine learning model training. Post-purchase loyalty hooks should be used to reduce guest checkouts.
3. **Cross-Merchandising Channels:** High-volume convenience drivers (e.g., "World War 2 Gliders") should be strategically paired or bundled with premium high-margin items (e.g., "Regency Cakestand 3 Tier") to naturally grow Average Order Value (AOV).
4. **Data Cleansing Requirements:** Outlier spikes highlight severe negative metrics in quantities/prices. A target data pipeline must separate cancellation codes from positive purchasing records to keep machine learning models from getting skewed.
5. **Supply Chain Scaling:** Operations must run tightly by August to successfully catch the massive annual peak in October and November where sales safely double standard monthly operations.

---

## How to Run the Notebook
1. Open Google Colab.
2. Upload the `.ipynb` file from this repository.
3. Run the cells sequentially from top to bottom[span_8](start_span)[span_8](end_span). The dataset will pull automatically from the source URL[span_9](start_span)[span_9](end_span).
# ITSimpleraInternship

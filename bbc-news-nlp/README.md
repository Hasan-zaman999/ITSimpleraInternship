
# Advanced NLP on BBC News Articles Dataset

![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blueviolet?style=for-the-badge)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFB01F.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Project Status](https://img.shields.io/badge/Status-Completed-success.svg?style=for-the-badge)

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Workflow Diagram](#workflow-diagram)
- [Technologies Used](#technologies-used)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Folder Structure](#folder-structure)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [Author Information](#author-information)

---

## Project Overview
✨ This project delves into advanced Natural Language Processing (NLP) techniques applied to the BBC News Articles Dataset. It covers a comprehensive pipeline from robust data preprocessing and exploratory data analysis to sophisticated text classification, topic modeling, and named entity recognition. The aim is to build and evaluate models that accurately categorize news articles and uncover their underlying themes, showcasing both traditional machine learning and transfer learning approaches.

## Motivation
In today's data-rich environment, efficiently organizing and understanding vast amounts of text data is crucial. This project is motivated by the need to demonstrate practical NLP skills in:
*   **Automated Content Categorization**: Streamlining news aggregation and content management.
*   **Information Extraction**: Identifying key entities and themes for deeper insights.
*   **Model Comparison**: Benchmarking classical ML against cutting-edge deep learning methods.

## Features
🚀
*   **Extensive Data Preprocessing**: Tokenization, lemmatization, stopword removal, and more.
*   **Named Entity Recognition (NER)**: Extracting Persons, Organizations, Locations, etc., using spaCy.
*   **Topic Modeling**: Discovering latent themes with Latent Dirichlet Allocation (LDA).
*   **Text Classification**: Comparing TF-IDF + Logistic Regression (baseline) with fine-tuned DistilBERT (transfer learning).
*   **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrices.
*   **Professional Documentation**: Generating `README.md`, `requirements.txt`, and academic-style `PROJECT_DOCUMENTATION.md`/`.pdf`.

## Dataset Description

| Attribute        | Description                                                               |
| :--------------- | :------------------------------------------------------------------------ |
| **Name**         | BBC News Articles Dataset                                                 |
| **Total Articles** | 2225                                                                      |
| **Categories**   | Business, Entertainment, Politics, Sport, Technology (5 categories)       |
| **Features**     | `category`, `filename`, `title`, `content`                                |
| **Cleanliness**  | No missing values or duplicate entries, ensuring high data quality.         |

The dataset provides a rich ground for exploring news content, enabling classification and thematic analysis across diverse journalistic domains.

## Workflow Diagram

```mermaid
graph TD
    A[Raw BBC News Data] --> B{Data Preprocessing}
    B --> C[Cleaned & Combined Text]

    C --> D[Named Entity Recognition (spaCy)]
    D --> D1[Extracted Entities (PERSON, ORG, GPE, etc.)]

    C --> E[Topic Modeling (LDA)]
    E --> E1[Discovered Topics (5-Topic Model: Business, Entertainment, Politics, Sport, Technology)]

    C --> F{Text Classification}
    F --> F1[Traditional ML (TF-IDF + Logistic Regression)]
    F --> F2[Transfer Learning (DistilBERT Fine-tuning)]

    F1 --> G[Baseline Model Evaluation (Accuracy: 98.88%)]
    F2 --> H[DistilBERT Model Evaluation (Accuracy: 98.43%)]

    G & H --> I[Model Comparison & Selection]
    I --> J[Saved Best Model (TF-IDF + Logistic Regression)]
```

## Technologies Used

| Library/Tool        | Purpose                                                                   |
| :------------------ | :------------------------------------------------------------------------ |
| **Python 3.9+**     | Core programming language                                                 |
| **Pandas**          | Data manipulation and analysis                                            |
| **NumPy**           | Numerical operations                                                      |
| **spaCy**           | Advanced NLP, especially for Tokenization, Lemmatization, and NER         |
| **Scikit-learn**    | Traditional ML models (TF-IDF, Logistic Regression), preprocessing, metrics |
| **Hugging Face Transformers** | State-of-the-art transfer learning (DistilBERT)                         |
| **Datasets**        | Efficient handling of NLP datasets for Transformers                       |
| **Matplotlib/Seaborn**| Data visualization and plotting                                           |
| **Jupyter/Colab**   | Interactive development environment                                       |
| **Joblib**          | Model persistence (saving/loading models)                                 |
| **ReportLab**       | PDF generation from Markdown                                              |

## Installation Guide
To set up the project locally, follow these steps:

1.  **Clone the repository** (if hosted on GitHub):
    ```bash
    git clone https://github.com/your-username/bbc-nlp-project.git
    cd bbc-nlp-project
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download spaCy models**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage Guide

1.  **Data Acquisition**: Ensure the `bbc-news-data.csv` dataset is accessible (e.g., in `/content/drive/MyDrive/` if using Google Colab).
2.  **Run the Notebook**: Open the main Jupyter Notebook (`bbc_nlp_project.ipynb` or your Colab file) and execute the cells sequentially.
    *   **Data Exploration & Preprocessing**: Initial data cleaning and text preparation.
    *   **Named Entity Recognition**: Extracting and visualizing entities.
    *   **Topic Modeling**: Discovering and interpreting article themes.
    *   **Text Classification**: Training and evaluating baseline and transfer learning models.
3.  **Review Outputs**: Inspect generated visualizations, classification reports, and the saved model artifacts.
4.  **Documentation Generation**: The provided code cells will generate `README.md`, `requirements.txt`, `PROJECT_DOCUMENTATION.md`, and `PROJECT_DOCUMENTATION.pdf` for comprehensive project sharing.

## Folder Structure

```
.  
├── bbc-nlp-project.ipynb  # Main Jupyter Notebook / Colab File
├── bbc-news-data.csv      # Raw dataset
├── requirements.txt       # Project dependencies
├── README.md              # Project overview and guide
├── PROJECT_DOCUMENTATION.md # Detailed academic documentation (Markdown)
├── PROJECT_DOCUMENTATION.pdf # Detailed academic documentation (PDF)
├── bbc_logistic_regression.pkl # Saved Logistic Regression model
├── tfidf_vectorizer.pkl   # Saved TF-IDF vectorizer
└── label_encoder.pkl      # Saved LabelEncoder
```

## Methodology

### Data Exploration and Preprocessing
- Initial checks for dimensions, data types, missing values, and duplicates.
- Aggressive text cleaning: lowercasing, punctuation/number removal, stopword removal, and lemmatization using spaCy.
- Combined `title` and `content` into a single `text` column for unified processing.

### Named Entity Recognition (NER)
- Employed spaCy's `en_core_web_sm` model to identify and categorize entities (PERSON, ORG, GPE, etc.).
- Analyzed entity frequency and distribution across news categories through visualizations.

### Topic Modeling
- Utilized Latent Dirichlet Allocation (LDA) for unsupervised topic discovery.
- `CountVectorizer` was used to transform text into a document-term matrix.
- Evaluated models with 4, 5, 6, and 7 topics based on perplexity and interpretability.
- Selected a 5-topic model due to its clear alignment with actual categories: Business, Entertainment, Politics, Sport, Technology.

### Text Classification

#### Traditional Machine Learning Baseline
- **Vectorization**: TF-IDF to convert text into numerical features.
- **Model**: Logistic Regression, a robust and interpretable classifier.
- **Performance**: Achieved **98.88% accuracy**.

#### Transfer Learning with DistilBERT
- **Model**: DistilBERT (`distilbert-base-uncased`) from Hugging Face Transformers.
- **Approach**: Fine-tuned the pre-trained model on the BBC dataset.
- **Performance**: Achieved **98.43% accuracy**.

## Results Summary

### Performance Comparison Table

| Model                        | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
| :--------------------------- | :------- | :-------------------- | :----------------- | :------------------- |
| **TF-IDF + Logistic Regression** | **98.88%** | **98.98%**            | **98.98%**         | **98.98%**           |
| Fine-Tuned DistilBERT        | 98.43%   | 98.44%                | 98.60%             | 98.43%               |

💡 **Conclusion**: The TF-IDF + Logistic Regression model performed marginally better on this dataset, achieving 98.88% accuracy. Its simplicity and strong performance make it the preferred choice for deployment in this context.

### Topic Modeling Results
- **Selected Model**: 5-Topic LDA Model
- **Key Observation**: Strong alignment between discovered topics and actual news categories, indicating successful thematic extraction.

### Named Entity Recognition Results
- **Most Frequent Entity Types**: PERSON, ORG, DATE, GPE.
- **Insights**: Different news categories exhibit distinct patterns in entity usage (e.g., 'Politics' and 'Sport' have higher PERSON entity counts).

## Future Improvements
🚀
- **Advanced Preprocessing**: Experiment with context-aware stopword removal or domain-specific lemmatization.
- **Ensemble Models**: Combine predictions from Logistic Regression and DistilBERT for potentially higher accuracy.
- **Hyperparameter Tuning**: More exhaustive tuning for both LDA and DistilBERT models.
- **Interpretability**: Implement SHAP or LIME for better understanding of model decisions, especially for DistilBERT.
- **Real-time Deployment**: Integrate the best model into a web application using frameworks like Flask or FastAPI.

## Acknowledgements
This project utilized the BBC News Articles Dataset. Special thanks to the developers and maintainers of `spaCy`, `scikit-learn`, `Hugging Face Transformers`, `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn` for their invaluable contributions to the open-source community.

## References
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.
- Hugging Face Transformers library: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
- spaCy library: [https://spacy.io/](https://spacy.io/)
- Scikit-learn library: [https://scikit-learn.org/](https://scikit-learn.org/)

## Author Information

**Name**: Google Colab Agent (AI Assistant)
**Role**: Machine Learning Engineer / NLP Specialist
**Contact**: [No direct contact provided]


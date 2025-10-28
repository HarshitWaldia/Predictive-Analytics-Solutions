# Predictive Analytics Solutions

This repository serves as a portfolio showcasing end-to-end Machine Learning projects. Each project addresses a distinct business problem, demonstrating skills in data cleaning, feature engineering, model training, evaluation, and interpretation.

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://scikit-learn.org/stable/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://seaborn.pydata.org/"><img src="https://img.shields.io/badge/Seaborn-31517E?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"></a>
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-orange?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"></a>
  <a href="https://xgboost.readthedocs.io/en/stable/"><img src="https://img.shields.io/badge/XGBoost-005E51?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook"></a>
</p>

---
## 🚀 New Notebook Added: Predictive Manufacturing Efficiency Analysis! 🚀

I've just added a comprehensive Jupyter Notebook, **main.ipynb**, to the repository! This notebook contains the complete workflow for predicting manufacturing production efficiency.

**Where to find it:**

*   **Local Path:** `D:\Omnie Solutions\Mark-1\Manufacturing-Team\Notebooks\Main.ipynb`
*   **GitHub:** [Predictive-Analytics-Solutions/Manufacturing-Team/Notebooks/Main.ipynb](https://github.com/Predictive-Analytics-Solutions/Manufacturing-Team/blob/main/Notebooks/Main.ipynb)

**📘 What’s Inside main.ipynb:**

*   ✅ **Exploratory Data Analysis (EDA):**  Includes data cleaning, preprocessing (handling missing values, outliers, and correlations), and insightful visualizations.
*   ✅ **Feature Engineering:**  Covers encoding categorical features, scaling/normalization, and the creation of derived/interaction features.
*   ✅ **Model Training & Evaluation:**  We've implemented and compared Linear Regression, Random Forest, and XGBoost models, evaluating performance using MAE, RMSE, and R².  Hyperparameter tuning was performed using GridSearchCV.
*   ✅ **User Input Prediction:**  An interactive section allows you to input custom data and receive instant predictions from the model!

All the work is consolidated inside this single notebook, making it easy to follow, understand, and reproduce our results.  

Check it out and let us know what you think! ✨




---

## 📁 Project 1: Manufacturing Team Efficiency Prediction

### Problem Statement:
This project aims to understand and predict the daily operational efficiency of production teams within a manufacturing plant. By identifying the key factors influencing efficiency, management can make informed decisions to optimize production processes and improve overall performance.

### Dataset:
The dataset contains various operational metrics from a manufacturing plant, including `recordDate`, `fiscalQuarter`, `productionDept`, `dayOfWeek`, `team`, `plannedEfficiency`, `standardMinuteValue`, `workInProgress`, `overtimeMinutes`, `performanceBonus`, `idleMinutes`, `idleWorkers`, `styleChangeCount`, and `workerCount`, with `efficiencyScore` as the target variable.

### Key Steps:
1.  **Data Cleaning & Preprocessing**: Initial handling of raw data to ensure quality.
2.  **Feature Engineering (`Feature_Engg.ipynb`)**:
    *   **Categorical Encoding**: One-Hot Encoding for nominal features (`productionDept`, `dayOfWeek`) and Label Encoding for ordinal features (`styleChangeCount`, `fiscalQuarter`).
    *   **Binary Feature Creation**: `idleOccurred` from `idleMinutes`.
    *   **Derived Features**: `workPerWorker` and `overtimePerWorker` to capture team-level dynamics.
    *   **Missing Value Imputation**: Handling `NaN` values introduced during feature transformations or present in the original dataset.
    *   **Feature Scaling**: Applying `StandardScaler` to numerical features for model compatibility.
    *   **Train/Test Split**: Dividing the dataset for model training and unbiased evaluation.
3.  **Model Training & Evaluation (`model_training.ipynb`)**:
    *   **Model Selection**: Training several regression models (Linear Regression, Ridge, Random Forest, XGBoost).
    *   **Hyperparameter Tuning**: Using `RandomizedSearchCV` to optimize the performance of selected models (e.g., Random Forest).
    *   **Performance Metrics**: Evaluating models using R-squared (R2) and Mean Absolute Error (MAE).
    *   **Feature Importance Analysis**: Identifying the most influential features for predicting efficiency.

### Technologies Used:
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Category Encoders`, `XGBoost`, `Matplotlib`, `Seaborn`

---

## 🛒 Project 2: E-commerce Conversion Prediction

### Problem Statement:
Develop a model that can predict the likelihood of a conversion event (`MonetaryConversion`) based on a user’s browsing behavior, engagement metrics, and contextual factors. This project aims to encourage the exploration of how user engagement, device diversity, and time-based factors contribute to online conversion outcomes, including challenges posed by missing values and correlated variables common in real-world e-commerce analytics.

### Dataset: Retail Web Session Intelligence (RWSI)
The Retail Web Session Intelligence (RWSI) dataset simulates customer interactions on a digital retail platform. Each record represents an anonymized user session, capturing browsing patterns, engagement metrics, contextual attributes, and conversion outcomes. The goal is to build predictive and diagnostic models that help understand what drives successful purchase intent and user engagement.

**Feature Definitions:**
*   **`SessionID`**: Unique alphanumeric identifier for each session.
*   **`AdClicks`**: Number of ad banners clicked during the session (0–4), proxy for ad engagement.
*   **`InfoSectionCount`**: Number of times a user accessed informational or support sections.
*   **`InfoSectionTime`**: Total time (in seconds) spent in informational/help sections.
*   **`HelpPageVisits`**: Count of dedicated help or guidance pages visited.
*   **`HelpPageTime`**: Cumulative time spent on help pages.
*   **`ItemBrowseCount`**: Number of product pages viewed in the session, proxy for product discovery.
*   **`ItemBrowseTime`**: Total time spent on product-related pages.
*   **`ExitRateFirstPage`**: Ratio of sessions that ended after the first page view.
*   **`SessionExitRatio`**: Overall exit probability based on pages viewed vs. total exits.
*   **`PageEngagementScore`**: Derived score indicating page value/interactivity.
*   **`HolidayProximityIndex`**: Index (0–1) representing closeness to major holidays/campaigns.
*   **`VisitMonth`**: Encoded month of visit (1–12) for seasonality.
*   **`UserPlatformID`**: Encoded identifier for the user’s operating platform.
*   **`WebClientCode`**: Encoded browser identifier.
*   **`MarketZone`**: Encoded global region or market area.
*   **`TrafficSourceCode`**: Encoded numeric tag for inbound traffic type.
*   **`UserCategory`**: Encoded user classification (e.g., New, Returning, Loyal).
*   **`IsWeekendVisit`**: Boolean indicator (0/1) if the session occurred on a weekend.
*   **`MonetaryConversion` (Target Variable)**: Binary target (1 = transaction, 0 = no conversion).

### Key Accomplishments & Learning Objectives:
By completing this project, the following objectives were met:
*   **Digital Behavior Data Understanding**: Explored session-level features such as browsing patterns, engagement metrics, and contextual variables.
*   **Exploratory Data Analysis (EDA)**: Identified trends, correlations, and anomalies; visualized how behavior differs between converting and non-converting sessions.
*   **Missing Value Handling**: Implemented strategies to address missing values effectively.
*   **Feature Engineering**: Created new features to enhance model predictive power.
*   **Predictive Model Building**: Developed classification models to predict `MonetaryConversion`.
*   **Performance Evaluation**: Assessed model performance using appropriate metrics for classification tasks.
*   **Model Interpretation**: Interpreted model results to understand factors differentiating high-intent shoppers from casual browsers.
*   **Communication of Findings**: Summarized key insights and actionable recommendations.

### Technologies Used:
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Category Encoders`, (Common classification models like `Logistic Regression`, `Random Forest Classifier`, `XGBoost Classifier`), `Matplotlib`, `Seaborn`
---

## 📁 Project Structure
```bash
Mark-1/
├── .gitignore
├── README.md
├── requirements.txt
├── DATA/
│   ├── RWSI.xlsx
│   ├── manufacturing_data.csv
│   └── ... (other raw datasets)
├── Manufacturing-Team/
│   ├── Data/
│   │   ├── cleaned_manufacturing_data.csv
│   │   ├── Train-Test/
│   │   │   ├── X_train.csv
│   │   │   ├── X_test.csv
│   │   │   ├── y_train.csv
│   │   │   └── y_test.csv
│   │   └── ...
│   ├── Models/
│   │   ├── Linear_Regression.pkl
│   │   ├── Random_Forest_Regressor.pkl
│   │   └── XGBoost_Regressor.pkl
│   └── Notebooks/
│       ├── EDA.ipynb
│       ├── Feature_Engg.ipynb
│       ├── Model_Training.ipynb
│       └── Models.ipynb
└── Retail-Web-Session-Intelligence/
    ├── Data/
    │   ├── cleaned_session_data.csv
    │   ├── data_final_dataset_for_training.csv
    │   └── ...
    ├── Models/
    │   ├── Best_random_forest_model.pkl
    │   ├── logistic_regression_model.pkl
    │   ├── preprocessor.pkl
    │   └── Random_forest_model.pkl
    └── Notebooks/
        ├── RWSI_EDA.ipynb
        ├── RWSI_Feature_Engg.ipynb
        └── RWSI_Model_Training.ipynb

```

## 🚀 Getting Started

To run these projects locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HarshitWaldia/Predictive-Analytics-Solutions.git
    cd ml-project-portfolio
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Navigate into the desired project directory** (e.g., `manufacturing_efficiency_prediction/`).
5.  **Open and run the Jupyter notebooks** (`.ipynb` files) in sequential order.
    ```bash
    jupyter notebook
    ```

---

## 🤝 Contribution

Feel free to fork this repository, open issues, or submit pull requests. Any feedback or suggestions for improvement are welcome!

---

## 👨‍💻 Author

**Harshit Waldia**

*   GitHub: [@HarshitWaldia](https://github.com/HarshitWaldia)
*   LinkedIn: [Harshit Waldia](https://www.linkedin.com/in/harshitwaldia/)

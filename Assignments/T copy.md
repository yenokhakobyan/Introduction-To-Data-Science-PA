# AI Exam Task: IBM Telco Customer Churn Analysis

## Task Overview
You are a data scientist at a telecommunications company. Your mission is to analyze the **IBM Telco Customer Churn Dataset** and build a machine learning model that predicts which customers are likely to churn, enabling the company to take proactive retention measures.

## Business Context
- Dataset contains **7,043 customers** from a telecom company
- Customer acquisition cost: $500 per customer
- Average monthly revenue per customer: $64.76
- Retention campaign cost: $50 per customer
- Typical retention campaign success rate: 40%

## Dataset Information
**Source**: IBM Cognos Analytics Sample Data  
**File**: `data.csv`  
**Size**: 7,043 rows × 21 columns

### Dataset Features:

**Customer Demographics:**
- `customerID`: Unique identifier
- `gender`: Male/Female
- `SeniorCitizen`: 0 (No), 1 (Yes)
- `Partner`: Yes/No (has partner)
- `Dependents`: Yes/No (has dependents)

**Service Information:**
- `tenure`: Number of months with company
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service

**Account Information:**
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)
- `MonthlyCharges`: Monthly bill amount
- `TotalCharges`: Total amount charged
- `Churn`: **TARGET VARIABLE** - Yes (churned), No (retained)

---

# Part 1: Project Implementation (30%)

## 1. Data Preprocessing (5 points)

### Task 1: Data Cleaning Pipeline
**Create a comprehensive preprocessing pipeline that handles:**

**Data Quality Issues:**
- Missing values in TotalCharges column
- Inconsistent data types
- Categorical variables with special values ("No internet service", "No phone service")

**Implementation Requirements:**
- Write reusable preprocessing functions
- Handle the TotalCharges conversion issue
- Create appropriate encoding strategies

💡 **Hint**: `TotalCharges` may contain string values and missing data represented as spaces

### Task 1.2: Feature Engineering
**Create meaningful new features from existing data:**

**Suggested feature categories:**
- Tenure-based features (new customer, tenure groups)
- Service bundling features (total services, premium services)
- Financial features (average charges, price per service)
- Risk factors (payment risk, contract risk)

**Requirements:**
- Create at least 5 new meaningful features

---

## 2. Exploratory Data Analysis (5 points)

### Task 2.1: Comprehensive EDA Report
**Create a detailed EDA report including:**

**Statistical Analysis:**
- Descriptive statistics for all variables
- Correlation analysis between numerical variables

**Visualizations:**
- Some Distribution plots for numerical variables
- Some Count plots for categorical variables
- Correlation heatmaps
- Churn rate analysis by different segments

💡 **Hint**: Use `seaborn` and `matplotlib` for professional visualizations

---

## 3. Model Selection (5 points)

### Task 3.1: Model Comparison Framework
**Implement a systematic model comparison:**

**Models to compare:**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)
- Support Vector Machine

**Evaluation Framework:**
- Use cross-validation
- Multiple evaluation metrics

### Task 3.2: Hyperparameter Optimization
**For the best-performing model:**
- Implement Grid Search or Random Search
- Use cross-validation

💡 **Hint**: Consider using `GridSearchCV` or `RandomizedSearchCV` with appropriate scoring metrics

---

## 4. Model Training (5 points)

### Task 4.1: Final Model Training
**Train your selected model with optimal hyperparameters:**

**Requirements:**
- Use the full training dataset
- Implement proper validation strategy
- Save the trained model for deployment


## 6. Model Evaluation (5 points)

### Task 6.1: Performance Evaluation
**Conduct thorough model evaluation:**

**Technical Metrics:**
- Accuracy, Precision, Recall, F1-score
- ROC-AUC and Precision-Recall AUC

---

# Part 2: Technical Interview Questions (70%)

## 1. Key Concepts of Machine Learning (10 points)

### Question 1.1: Problem Classification
**Task**: Analyze this business problem and classify the type of machine learning task.

**What you need to determine:**
- What type of ML problem is this? (supervised/unsupervised, classification/regression)
- What are the main challenges you anticipate with this dataset?
- How would you handle class imbalance if it exists?

💡 **Hint**: Look at the target variable `Churn` and examine its distribution using `value_counts()`

### Question 1.2: Data Quality Assessment
**Task**: Identify and plan solutions for data quality issues in this dataset.

**What to investigate:**
- Are there missing values? Which columns?
- Are there any data type inconsistencies?
- How would you handle categorical variables with multiple categories (e.g., "No internet service")?

💡 **Hint**: Use `df.info()`, `df.isnull().sum()`, and examine the `TotalCharges` column carefully

---

## 2. Exploratory Data Analysis (10 points)

### Question 2.1: Univariate Analysis
**Task**: Perform comprehensive univariate analysis.

**Your analysis should include:**
- Distribution of the target variable (churn rate)
- Distribution of numerical variables (tenure, charges)
- Frequency analysis of categorical variables

💡 **Hint**: Use `df['Churn'].value_counts(normalize=True)` to get churn rate percentage

### Question 2.2: Bivariate Analysis
**Task**: Analyze relationships between features and churn.

**Investigate these relationships:**
- How does churn vary by contract type?
- What's the relationship between tenure and churn?
- Which payment methods have highest churn rates?
- How does internet service type affect churn?

💡 **Hint**: Use cross-tabulations with `pd.crosstab()` and group analysis with `df.groupby()`

### Question 2.3: Key Insights
**Task**: Identify insights from your EDA.

**Focus areas:**
- Contract-related patterns
- Payment method implications
- Service usage patterns
- Customer demographic effects

---

## 3. K-Nearest Neighbors (5 points)

### Question 3.1: KNN Suitability
**Task**: Evaluate whether KNN is appropriate for this dataset.

**Consider:**
- Mixed data types (numerical + categorical)
- Dataset size and computational efficiency
- Curse of dimensionality
- Required preprocessing steps

**If you implement KNN:**
- How would you handle categorical variables?
- What distance metric would you use?
- How would you select the optimal k value?

💡 **Hint**: Consider the challenges of distance calculation with mixed data types

---

## 4. Decision Trees (10 points)

### Question 4.1: Implementation
**Task**: Build a decision tree classifier for churn prediction.

**Requirements:**
- Proper preprocessing for categorical variables
- Hyperparameter tuning to prevent overfitting

---

## 5. Ensemble Methods (10 points)

### Question 5.1: Random Forest vs Gradient Boosting
**Task**: Compare Random Forest and Gradient Boosting (or XGBoost) performance.

**Implementation requirements:**
- Use proper cross-validation
- Compare multiple metrics (accuracy, precision, recall, F1, ROC-AUC)

---

## 6. Generalized Linear Models & Gradient Descent (10 points)

### Question 6.1: Logistic Regression Implementation
**Task**: Implement logistic regression with proper preprocessing.

**Requirements:**
- Feature scaling for numerical variables
- Appropriate encoding for categorical variables
- Regularization to prevent overfitting

### Question 6.2: Coefficient Interpretation
**Task**: Interpret the logistic regression coefficients for business insights.

**Analyze:**
- Which features increase/decrease churn probability?

💡 **Hint**: Positive coefficients increase churn probability; use `np.exp(coef)` for odds ratios

---

## 7. Support Vector Machines (5 points)

### Question 7.1: SVM Implementation
**Task**: Implement SVM for churn prediction.

**Consider:**
- Kernel selection (linear, RBF, polynomial)
- Hyperparameter tuning (C, gamma)
- Feature scaling requirements

**Compare SVM performance with tree-based methods and explain the differences.**

💡 **Hint**: SVM requires feature scaling and can be computationally expensive for large datasets

---

## 8. Logistic Regression Deep Dive (10 points)

### Question 8.1: Mathematical Understanding
**Task**: Demonstrate understanding of logistic regression mechanics.

**Explain:**
- Why logistic regression is suitable for this problem
- How the sigmoid function works

### Question 8.2: Model Diagnostics
**Task**: Perform comprehensive logistic regression diagnostics.

**Include:**
- ROC curve analysis
- Precision-recall interpretation

---

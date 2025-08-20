# Real-World AI Task: IBM Telco Customer Churn Analysis

## Task Overview
You are a data scientist at a telecommunications company using the **IBM Telco Customer Churn Dataset** to build a machine learning model that predicts which customers are likely to churn. This dataset contains real business patterns and is widely used in the industry for churn prediction modeling.

## Business Context
- Dataset contains **7,043 customers** from a telecom company
- Current churn rate: **26.5%** (realistic industry standard)
- Customer acquisition cost: $500 per customer
- Average monthly revenue per customer: $64.76
- Retention campaign cost: $50 per customer
- Success rate of retention campaigns: 40%

## Dataset Description
**Source**: IBM Cognos Analytics Sample Data
**File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows × 21 columns)

### Real Dataset Features:

**Customer Demographics:**
- `customerID`: Unique identifier (string)
- `gender`: Male/Female
- `SeniorCitizen`: 0 (No), 1 (Yes)
- `Partner`: Yes/No (has partner)
- `Dependents`: Yes/No (has dependents)

**Service Information:**
- `tenure`: Number of months with company (0-72)
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
- `MonthlyCharges`: Monthly bill amount ($18.25-$118.75)
- `TotalCharges`: Total amount charged (string with some missing values)

**Target Variable:**
- `Churn`: Yes (churned), No (retained)

## Part 1: Technical Interview Questions (70%)

### 1. Key Concepts of ML (10 points)

**Question 1:** "Looking at this IBM dataset, what type of machine learning problem is this and what challenges do you anticipate?"

**Expected Answer:**
- Binary classification problem (Churn: Yes/No)
- Supervised learning with labeled historical data
- **Key Challenges**:
  - Class imbalance (~26.5% churn rate)
  - Mixed data types (numerical + categorical)
  - Missing values in TotalCharges column
  - Multicollinearity between related services
  - Business interpretability requirements

**Question 2:** "How would you handle the specific data quality issues in this dataset?"

**Expected Answer:**
```python
# Handle TotalCharges data quality issue
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Missing TotalCharges: {df['TotalCharges'].isnull().sum()}")

# Strategy for missing values
# Option 1: Drop (only 11 rows)
# Option 2: Impute with median or calculate from tenure*MonthlyCharges
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
```

### 2. EDA (10 points)

**Question:** "Walk me through your EDA approach for this real dataset."

**Expected Key Findings:**
```python
# Load the real dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic statistics
print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
# Output: 26.5%

# Key insights you should discover:
# 1. Month-to-month customers have 42% churn rate
# 2. Electronic check payment has highest churn (45%)
# 3. Fiber optic customers churn more than DSL customers
# 4. New customers (tenure < 12) have 50%+ churn rate
# 5. Senior citizens have higher churn rates
# 6. Customers without partners/dependents churn more
```

**Critical EDA Insights:**
- **Contract Type**: Month-to-month (42% churn) vs Two-year (3% churn)
- **Payment Method**: Electronic check (45% churn) vs Credit card (15% churn)
- **Tenure Effect**: Strong negative correlation with churn
- **Internet Service**: Fiber optic customers have higher churn despite higher bills
- **Support Services**: Customers without OnlineSecurity/TechSupport churn more

### 3. Decision Trees (10 points)

**Question:** "Build a decision tree for this dataset and interpret the business rules it discovers."

**Expected Implementation:**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Preprocessing for decision tree
def preprocess_for_tree(df):
    # Create copy
    data = df.copy()
    
    # Handle TotalCharges
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['tenure'] * data['MonthlyCharges'], inplace=True)
    
    # Binary encoding for Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0})
    
    # Handle categorical columns with "No internet service" / "No phone service"
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    data['MultipleLines'] = data['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
    
    # One-hot encode remaining categorical variables
    data = pd.get_dummies(data, columns=['gender', 'Contract', 'PaymentMethod', 'InternetService'])
    
    # Encode target
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    return data.drop('customerID', axis=1)

# Build decision tree
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=200,
    min_samples_leaf=100,
    random_state=42
)

X = processed_data.drop('Churn', axis=1)
y = processed_data['Churn']
dt.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
```

**Expected Business Rules Discovery:**
- **Root Node**: Contract type (Month-to-month vs others)
- **Secondary Splits**: Tenure, TotalCharges, InternetService
- **Key Rule**: "If Contract is Month-to-month AND tenure < 10 months → 70% churn probability"

### 4. Ensemble Methods (10 points)

**Question:** "Compare Random Forest vs XGBoost performance on this dataset."

**Expected Analysis:**
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Random Forest - handles mixed data types well
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)

# XGBoost - often superior performance
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=2.77,  # Adjust for class imbalance (73.5/26.5)
    random_state=42
)

# Cross-validation comparison
rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
xgb_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='f1')

print(f"Random Forest F1: {rf_scores.mean():.3f} (+/- {rf_scores.std():.3f})")
print(f"XGBoost F1: {xgb_scores.mean():.3f} (+/- {xgb_scores.std():.3f})")
```

**Expected Results:**
- **Random Forest**: Better interpretability, faster training, ~0.62 F1-score
- **XGBoost**: Superior performance, better handling of complex patterns, ~0.68 F1-score
- **Business Trade-off**: XGBoost for prediction accuracy vs Random Forest for business insights

### 5. Logistic Regression (10 points)

**Question:** "Implement logistic regression and interpret the coefficients for business insights."

**Expected Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Feature scaling required for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression with regularization
lr = LogisticRegression(
    class_weight='balanced',
    C=1.0,
    penalty='l2',
    random_state=42,
    max_iter=1000
)

lr.fit(X_train_scaled, y_train)

# Coefficient interpretation
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0],
    'odds_ratio': np.exp(lr.coef_[0])
}).sort_values('coefficient', key=abs, ascending=False)

print("Top 5 Churn Drivers:")
print(coefficients.head())
```

**Expected Key Insights:**
- **Contract_Month-to-month**: Highest positive coefficient (increases churn odds by ~3x)
- **tenure**: Strong negative coefficient (each month reduces churn odds by 5%)
- **PaymentMethod_Electronic check**: Significant positive coefficient
- **InternetService_Fiber optic**: Positive coefficient (service quality issues?)
- **TotalCharges**: Negative coefficient (loyal high-value customers)

### 6. Model Evaluation (10 points)

**Question:** "How would you evaluate model performance for this business problem?"

**Expected Answer:**
```python
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# Business-focused evaluation
def evaluate_churn_model(model, X_test, y_test, threshold=0.5):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Standard metrics
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Business metrics
    tp = sum((y_test == 1) & (y_pred == 1))  # Correctly identified churners
    fp = sum((y_test == 0) & (y_pred == 1))  # False alarms
    fn = sum((y_test == 1) & (y_pred == 0))  # Missed churners
    
    campaign_cost = (tp + fp) * 50
    retention_revenue = tp * 0.4 * 64.76 * 12  # 40% success rate * monthly revenue * 12 months
    churn_cost = fn * 64.76 * 12  # Lost revenue from missed churners
    
    net_benefit = retention_revenue - campaign_cost - churn_cost
    
    print(f"Business Metrics:")
    print(f"Campaign Cost: ${campaign_cost:,.0f}")
    print(f"Retention Revenue: ${retention_revenue:,.0f}")
    print(f"Churn Cost: ${churn_cost:,.0f}")
    print(f"Net Benefit: ${net_benefit:,.0f}")
    
    return net_benefit

# Optimize threshold for maximum business value
thresholds = np.arange(0.1, 0.9, 0.05)
benefits = [evaluate_churn_model(model, X_test, y_test, t) for t in thresholds]
optimal_threshold = thresholds[np.argmax(benefits)]
print(f"Optimal Threshold: {optimal_threshold:.2f}")
```

### 7. Feature Engineering (10 points)

**Question:** "What additional features would you create from this dataset?"

**Expected Features:**
```python
def engineer_features(df):
    data = df.copy()
    
    # Tenure-based features
    data['is_new_customer'] = (data['tenure'] <= 12).astype(int)
    data['tenure_group'] = pd.cut(data['tenure'], 
                                bins=[0, 12, 24, 36, 72], 
                                labels=['New', 'Medium', 'Long', 'Veteran'])
    
    # Service-based features
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['total_services'] = sum([(data[col] == 'Yes').astype(int) for col in service_cols])
    data['has_premium_services'] = ((data['StreamingTV'] == 'Yes') | 
                                  (data['StreamingMovies'] == 'Yes')).astype(int)
    data['has_support_services'] = ((data['OnlineSecurity'] == 'Yes') | 
                                  (data['TechSupport'] == 'Yes')).astype(int)
    
    # Financial features
    data['avg_monthly_charge'] = data['TotalCharges'] / (data['tenure'] + 1)
    data['price_per_service'] = data['MonthlyCharges'] / (data['total_services'] + 1)
    data['high_value_customer'] = (data['MonthlyCharges'] > data['MonthlyCharges'].quantile(0.75)).astype(int)
    
    # Risk factors
    data['payment_risk'] = (data['PaymentMethod'] == 'Electronic check').astype(int)
    data['contract_risk'] = (data['Contract'] == 'Month-to-month').astype(int)
    data['demographic_risk'] = ((data['SeniorCitizen'] == 1) & 
                               (data['Partner'] == 'No') & 
                               (data['Dependents'] == 'No')).astype(int)
    
    return data
```

## Part 2: Project Implementation (30%)

### 1. Data Preprocessing (5 points)

**Task:** Create a complete preprocessing pipeline for the IBM dataset.

**Expected Implementation:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TelcoPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.column_transformer = None
        self.feature_names = None
    
    def fit_transform(self, df):
        data = df.copy()
        
        # Handle TotalCharges data quality issue
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Impute missing values (only 11 rows)
        data['TotalCharges'].fillna(data['tenure'] * data['MonthlyCharges'], inplace=True)
        
        # Feature engineering
        data = self.engineer_features(data)
        
        # Separate features by type
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                            'total_services', 'avg_monthly_charge', 'price_per_service']
        
        categorical_features = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
        
        binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                         'PaperlessBilling', 'is_new_customer', 'has_premium_services']
        
        # Create preprocessing pipeline
        from sklearn.preprocessing import OneHotEncoder
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features),
                ('bin', 'passthrough', binary_features)
            ])
        
        # Process service features separately (handle "No internet service")
        service_features = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_features:
            data[f'{col}_binary'] = data[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
        
        # Fit and transform
        X = data.drop(['customerID', 'Churn'], axis=1)
        y = data['Churn'].map({'Yes': 1, 'No': 0})
        
        X_processed = preprocessor.fit_transform(X)
        
        # Store feature names for later use
        feature_names = (numerical_features + 
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
                        binary_features +
                        [f'{col}_binary' for col in service_features])
        
        return pd.DataFrame(X_processed, columns=feature_names), y
    
    def engineer_features(self, df):
        # Add all the feature engineering from above
        # [Implementation details as shown in Feature Engineering section]
        return df
```

### 2. Model Selection and Training (10 points)

**Task:** Compare multiple algorithms and select the best performer using proper validation.

**Expected Approach:**
```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Define models with initial hyperparameters
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=2.77, random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
}

# Evaluation metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Cross-validation results
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Evaluating {name}...")
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    
    cv_results[name] = {
        'accuracy': scores['test_accuracy'].mean(),
        'precision': scores['test_precision'].mean(),
        'recall': scores['test_recall'].mean(),
        'f1': scores['test_f1'].mean(),
        'roc_auc': scores['test_roc_auc'].mean()
    }

# Display results
results_df = pd.DataFrame(cv_results).T
print(results_df.round(3))

# Hyperparameter tuning for best model (XGBoost)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [2.5, 2.77, 3.0]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.3f}")
```

### 3. Model Evaluation and Business Impact (10 points)

**Task:** Comprehensive evaluation with business context.

**Expected Deliverables:**
```python
# Final model evaluation
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Comprehensive evaluation report
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

print("=== MODEL PERFORMANCE REPORT ===")
print("\n1. Classification Report:")
print(classification_report(y_test, y_pred))

print("\n2. Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n3. ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n4. Top 10 Most Important Features:")
print(feature_importance.head(10))

# Business impact calculation
def calculate_business_impact(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    tn = sum((y_true == 0) & (y_pred == 0))
    
    # Business calculations
    customers_targeted = tp + fp
    successful_retentions = tp * 0.4  # 40% success rate
    campaign_cost = customers_targeted * 50
    retention_revenue = successful_retentions * 64.76 * 12
    lost_revenue = fn * 64.76 * 12
    
    net_benefit = retention_revenue - campaign_cost - lost_revenue
    roi = (net_benefit / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    return {
        'customers_targeted': customers_targeted,
        'successful_retentions': successful_retentions,
        'campaign_cost': campaign_cost,
        'retention_revenue': retention_revenue,
        'lost_revenue': lost_revenue,
        'net_benefit': net_benefit,
        'roi': roi
    }

business_impact = calculate_business_impact(y_test, y_pred_proba)
print("\n5. Business Impact Analysis:")
for key, value in business_impact.items():
    if 'cost' in key or 'revenue' in key or 'benefit' in key:
        print(f"{key}: ${value:,.0f}")
    elif 'roi' in key:
        print(f"{key}: {value:.1f}%")
    else:
        print(f"{key}: {value:.0f}")
```

### 4. Final Report and Recommendations (5 points)

**Executive Summary Template:**

```markdown
# Telecom Customer Churn Prediction - Executive Summary

## Problem Statement
Predicting customer churn using IBM's telecom dataset to enable proactive retention strategies.

## Key Findings
1. **Churn Rate**: 26.5% of customers churned (industry-typical rate)
2. **Primary Risk Factors**:
   - Month-to-month contracts (42% churn rate)
   - Electronic check payments (45% churn rate)
   - New customers with tenure < 12 months (50%+ churn)
   - Fiber optic customers (higher churn despite premium service)

## Model Performance
- **Algorithm**: XGBoost Classifier
- **Accuracy**: 82.3%
- **F1-Score**: 0.68
- **ROC-AUC**: 0.85
- **Business ROI**: 150%

## Business Impact
- **Monthly Savings**: $45,000 from prevented churn
- **Campaign Efficiency**: Target 1,200 customers to retain 480
- **Annual Revenue Protection**: $540,000

## Recommendations
1. **Contract Strategy**: Incentivize annual contracts for month-to-month customers
2. **Payment Method**: Encourage automatic payment methods
3. **New Customer Onboarding**: Enhanced support for first 12 months
4. **Fiber Service**: Investigate and improve fiber optic service quality
5. **Targeted Retention**: Focus on high-risk segments identified by the model
```

## Success Criteria

**Technical Interview (70%):**
- Demonstrate understanding of real-world data challenges
- Explain business context behind technical decisions
- Show proficiency with industry-standard dataset
- Interpret results for business stakeholders

**Project Implementation (30%):**
- Handle real data quality issues effectively
- Build production-ready preprocessing pipeline
- Achieve performance benchmarks (F1 > 0.65, ROC-AUC > 0.80)
- Provide actionable business recommendations

## Expected Benchmarks (IBM Dataset)
- **Baseline Accuracy**: 73.5% (predicting majority class)
- **Good Performance**: F1-score > 0.60, ROC-AUC > 0.80
- **Excellent Performance**: F1-score > 0.68, ROC-AUC > 0.85

This real-world task using the IBM dataset provides authentic experience with industry-standard data and business problems, making it excellent preparation for both technical interviews and actual data science work.
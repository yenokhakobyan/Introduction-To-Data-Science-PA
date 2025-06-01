# K-Nearest Neighbors Algorithm: Comprehensive Learning Assignment

## Assignment Overview

Welcome to this comprehensive exploration of the K-Nearest Neighbors (KNN) algorithm! This assignment is designed to take you on a complete journey from understanding the mathematical foundations to implementing sophisticated machine learning models. 

The beauty of KNN lies in its simplicity and intuitive nature - it makes predictions based on the principle that similar data points tend to have similar labels. However, beneath this simplicity lies a rich landscape of implementation details, optimization strategies, and evaluation techniques that you'll master through this assignment.

**Learning Journey:**
You'll start by working through calculations by hand to build deep intuition, then progress to implementing the algorithm from scratch to understand every component. Finally, you'll apply your knowledge to real-world datasets and master the art of model evaluation and hyperparameter tuning. Advanced topics and tasks are optional  **(Optional)**.

**Part 4 is mainly advanced topics and is not required but highlt recommended.** 

**Required Libraries:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
```

---

## Part 1: Manual KNN Calculations - Building Intuition (5 Questions)

Understanding KNN begins with grasping how distances work in feature space and how voting mechanisms determine final predictions. These manual exercises will build the foundational intuition you need.

### Question 1: Understanding Distance Metrics

You are given a small dataset representing houses with two features: size (in hundreds of square feet) and age (in years). Each house is labeled as either "Expensive" or "Affordable".

**Training Data:**
- House A: Size=20, Age=5 → "Expensive"  
- House B: Size=35, Age=15 → "Expensive"
- House C: Size=15, Age=25 → "Affordable"
- House D: Size=25, Age=10 → "Expensive"
- House E: Size=10, Age=30 → "Affordable"
- House F: Size=18, Age=20 → "Affordable"

**Your Task:** A new house has Size=22 and Age=12. Calculate the Euclidean distance from this new house to each training house. Show your complete calculations step by step, and then rank all training houses by their distance to the new house.

**Space for your calculations:**

```python
# You can use this space to verify your manual calculations
# But first, do the calculations by hand!

```

### Question 2: The Voting Mechanism

Using your distance calculations from Question 1, predict whether the new house (Size=22, Age=12) should be classified as "Expensive" or "Affordable" using different values of K.

**Your Task:** Make predictions for K=1, K=3, and K=5. For each K value, explain which neighbors are selected and how the final vote is determined. Discuss what happens when there's a tie in voting.

**Analysis Questions:**
- How does the prediction change as K increases?
- Which K value seems most appropriate for this dataset and why?
- What would happen if we used K=6 (an even number)?

### Question 3: Manhattan vs Euclidean Distance

Using the same dataset from Question 1, calculate the Manhattan distance (also called L1 distance) from the new house to each training house.

**Manhattan Distance Formula:** |x₁ - x₂| + |y₁ - y₂|

**Your Task:** Compare the rankings you get using Manhattan distance versus Euclidean distance. Do the nearest neighbors change? How might this affect the final classification?

### Question 4: Feature Scaling Impact

Consider a modified version of the housing dataset where we add a third feature - price (in thousands of dollars):

- House A: Size=20, Age=5, Price=450 → "Expensive"
- House B: Size=35, Age=15, Price=380 → "Expensive"  
- House C: Size=15, Age=25, Price=180 → "Affordable"

New house: Size=22, Age=12, Price=320

**Your Task:** Calculate the Euclidean distance with and without considering the price feature. Notice how the price feature (with much larger numerical values) dominates the distance calculation. Explain why feature scaling would be important in this scenario.

### Question 5: Regression with KNN

KNN can also be used for regression tasks. Given these houses with their actual sale prices:

- House A: Size=20, Age=5 → Price=$420K
- House B: Size=35, Age=15 → Price=$380K
- House C: Size=15, Age=25 → Price=$180K
- House D: Size=25, Age=10 → Price=$350K
- House E: Size=10, Age=30 → Price=$150K

**Your Task:** For a new house with Size=22 and Age=12, predict its price using K=3 KNN regression. Calculate distances, identify the 3 nearest neighbors, and determine the predicted price using the mean of the neighbors' prices.

---

## Part 2: Implementing KNN from Scratch (7-Step Implementation)

Now you'll build a complete KNN classifier from the ground up. This implementation will help you understand every component of the algorithm and give you the flexibility to modify it for different scenarios.

### Step 1: Distance Calculation Functions

Create functions to calculate different distance metrics. Understanding various distance measures is crucial because different datasets may benefit from different approaches.

```python
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    point1, point2: array-like, the two points
    
    Returns:
    float: The Euclidean distance
    """
    # Your implementation here
    pass

def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.
    
    Parameters:
    point1, point2: array-like, the two points
    
    Returns:
    float: The Manhattan distance
    """
    # Your implementation here
    pass

def minkowski_distance(point1, point2, p=2):
    """
    Calculate the Minkowski distance between two points.
    When p=1, it's Manhattan distance. When p=2, it's Euclidean distance.
    
    Parameters:
    point1, point2: array-like, the two points
    p: int, the order of the Minkowski distance
    
    Returns:
    float: The Minkowski distance
    """
    # Your implementation here
    pass

# Test your distance functions
test_point1 = np.array([1, 2, 3])
test_point2 = np.array([4, 5, 6])

print("Testing distance functions:")
print(f"Euclidean: {euclidean_distance(test_point1, test_point2)}")
print(f"Manhattan: {manhattan_distance(test_point1, test_point2)}")
print(f"Minkowski (p=3): {minkowski_distance(test_point1, test_point2, p=3)}")
```

### Step 2: Finding K Nearest Neighbors

Implement the core logic to find the K nearest neighbors to a given point. This step is the heart of the KNN algorithm.

```python
def find_k_nearest_neighbors(X_train, y_train, query_point, k, distance_func=euclidean_distance):
    """
    Find the k nearest neighbors to a query point.
    
    Parameters:
    X_train: array-like, shape (n_samples, n_features), training data
    y_train: array-like, shape (n_samples,), training labels
    query_point: array-like, shape (n_features,), point to find neighbors for
    k: int, number of neighbors to find
    distance_func: function, distance metric to use
    
    Returns:
    list: indices of k nearest neighbors
    list: distances to k nearest neighbors
    list: labels of k nearest neighbors
    """
    # Your implementation here
    # Hint: Calculate distances to all training points, sort, and select top k
    pass

# Test your function with a simple dataset
X_simple = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_simple = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
query = np.array([3, 3])

neighbors_idx, distances, neighbor_labels = find_k_nearest_neighbors(X_simple, y_simple, query, k=3)
print(f"Nearest neighbors indices: {neighbors_idx}")
print(f"Distances: {distances}")
print(f"Neighbor labels: {neighbor_labels}")
```

### Step 3: Classification Prediction

Implement the voting mechanism for classification tasks. Consider how to handle ties in voting and different weighting schemes.

```python
def knn_classify(X_train, y_train, query_point, k, distance_func=euclidean_distance, weighted=False):
    """
    Classify a query point using KNN.
    
    Parameters:
    X_train: array-like, training features
    y_train: array-like, training labels
    query_point: array-like, point to classify
    k: int, number of neighbors to consider
    distance_func: function, distance metric
    weighted: bool, whether to use distance-weighted voting
    
    Returns:
    prediction: the predicted class
    confidence: confidence score (proportion of votes for winning class)
    """
    # Your implementation here
    # Steps:
    # 1. Find k nearest neighbors
    # 2. Count votes (with or without distance weighting)
    # 3. Return most frequent class and confidence
    pass

# Test classification
prediction, confidence = knn_classify(X_simple, y_simple, query, k=3)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
```

### Step 4: Regression Prediction

Implement KNN for regression tasks, where instead of voting, you average the target values of nearest neighbors.

```python
def knn_regress(X_train, y_train, query_point, k, distance_func=euclidean_distance, weighted=False):
    """
    Predict a continuous value using KNN regression.
    
    Parameters:
    X_train: array-like, training features
    y_train: array-like, training target values
    query_point: array-like, point to predict for
    k: int, number of neighbors to consider
    distance_func: function, distance metric
    weighted: bool, whether to use distance-weighted averaging
    
    Returns:
    prediction: predicted continuous value
    """
    # Your implementation here
    # For weighted averaging, closer neighbors should have more influence
    pass

# Create test regression data
X_reg = np.array([[1], [2], [3], [4], [5], [6]])
y_reg = np.array([10, 15, 12, 18, 20, 25])
query_reg = np.array([3.5])

prediction = knn_regress(X_reg, y_reg, query_reg, k=3)
print(f"Regression prediction: {prediction:.2f}")
```

### Step 5: Complete KNN Class Implementation

Create a complete class that encapsulates all KNN functionality, similar to scikit-learn's interface.

```python
class KNeighborsClassifier:
    """
    Custom implementation of K-Nearest Neighbors Classifier.
    """
    
    def __init__(self, n_neighbors=5, distance_metric='euclidean', weighted=False):
        """
        Initialize the KNN classifier.
        
        Parameters:
        n_neighbors: int, number of neighbors to use
        distance_metric: str, distance metric ('euclidean', 'manhattan', 'minkowski')
        weighted: bool, whether to use distance-weighted voting
        """
        # Your implementation here
        pass
    
    def fit(self, X, y):
        """
        Store the training data.
        
        Parameters:
        X: array-like, training features
        y: array-like, training labels
        """
        # Your implementation here
        pass
    
    def predict(self, X):
        """
        Predict classes for test data.
        
        Parameters:
        X: array-like, test features
        
        Returns:
        predictions: array of predicted classes
        """
        # Your implementation here
        pass
    
    def predict_proba(self, X):
        """
        Predict class probabilities for test data.
        
        Parameters:
        X: array-like, test features
        
        Returns:
        probabilities: array of class probabilities
        """
        # Your implementation here
        pass

# Test your complete implementation
custom_knn = KNeighborsClassifier(n_neighbors=3)
# Add testing code here
```

### Step 6: Handling Edge Cases and Optimizations

Enhance your implementation to handle various edge cases and add optimizations.

```python
def preprocess_data(X, y, scale_features=True):
    """
    Preprocess data for KNN algorithm.
    
    Parameters:
    X: array-like, features
    y: array-like, labels
    scale_features: bool, whether to scale features
    
    Returns:
    X_processed: preprocessed features
    y_processed: preprocessed labels
    scaler: fitted scaler object (if scaling was applied)
    """
    # Your implementation here
    # Handle missing values, feature scaling, categorical encoding
    pass

def optimize_k_selection(X_train, X_val, y_train, y_val, k_range):
    """
    Find optimal K value using validation set.
    
    Parameters:
    X_train, X_val: training and validation features
    y_train, y_val: training and validation labels
    k_range: range of k values to test
    
    Returns:
    best_k: optimal k value
    scores: validation scores for each k
    """
    # Your implementation here
    pass

# Test preprocessing and optimization functions
```

### Step 7: Comprehensive Testing and Visualization

Create comprehensive tests and visualizations to understand your implementation's behavior.

```python
def visualize_knn_decision_boundary(X, y, k_values, custom_knn_class):
    """
    Visualize KNN decision boundaries for different K values.
    Only works for 2D data.
    
    Parameters:
    X: 2D array of features
    y: array of labels
    k_values: list of k values to visualize
    custom_knn_class: your KNN implementation
    """
    # Your implementation here
    # Create subplots showing decision boundaries for different K values
    pass

def compare_distance_metrics(X, y, distance_metrics):
    """
    Compare performance of different distance metrics.
    
    Parameters:
    X: features
    y: labels
    distance_metrics: list of distance metric names
    
    Returns:
    comparison_results: dictionary with performance metrics
    """
    # Your implementation here
    pass

# Run comprehensive tests
print("Running comprehensive tests of your KNN implementation...")
# Add your test cases here
```

---

## Part 3: Real-World Applications (2 Tasks)

Now you'll apply your understanding to real datasets, experiencing the challenges and nuances of practical machine learning.

### Task 1: Wine Quality Classification

The wine dataset contains chemical measurements of wines from different cultivars. This is a classic multi-class classification problem that will test your KNN implementation.

```python
# Load and explore the wine dataset
wine_data = load_wine()
X_wine, y_wine = wine_data.data, wine_data.target

print("Wine Dataset Information:")
print(f"Number of samples: {X_wine.shape[0]}")
print(f"Number of features: {X_wine.shape[1]}")
print(f"Number of classes: {len(np.unique(y_wine))}")
print(f"Feature names: {wine_data.feature_names}")
print(f"Class names: {wine_data.target_names}")

# Your tasks:
# 1. Perform exploratory data analysis
# 2. Implement data preprocessing (scaling, train/test split)
# 3. Apply your custom KNN implementation
# 4. Compare with sklearn's KNeighborsClassifier
# 5. Experiment with different K values and distance metrics
# 6. Analyze which features are most important for classification

# Start your implementation here:

```

### Task 2: Boston Housing Price Prediction (Regression)

For this task, create a regression dataset and apply KNN regression to predict continuous values.

```python
# Create a regression dataset
X_reg, y_reg = make_classification(n_samples=1000, n_features=8, n_informative=6, 
                                   n_redundant=2, n_clusters_per_class=1, 
                                   random_state=42)

# Convert to regression problem by creating continuous target
y_reg = np.dot(X_reg, np.random.random(X_reg.shape[1])) + np.random.normal(0, 0.1, X_reg.shape[0])

print("Regression Dataset Information:")
print(f"Number of samples: {X_reg.shape[0]}")
print(f"Number of features: {X_reg.shape[1]}")
print(f"Target range: {y_reg.min():.2f} to {y_reg.max():.2f}")

# Your tasks:
# 1. Implement KNN regression (extend your class or create new one)
# 2. Compare different K values for regression
# 3. Evaluate using appropriate regression metrics (MSE, MAE, R²)
# 4. Visualize predictions vs actual values
# 5. Analyze the effect of feature scaling on regression performance

# Start your implementation here:

```

---

## Part 4: Advanced Topics and Hyperparameter Tuning (15 Tasks)

This section will master the sophisticated aspects of KNN and model evaluation. These tasks will transform you from a basic user to an expert practitioner.

### Tasks 1-3: Understanding Cross-Validation from Scratch

#### Task 1: Implement K-Fold Cross-Validation Manually

```python
def manual_k_fold_cv(X, y, k_folds, model_class, **model_params):
    """
    Implement k-fold cross-validation from scratch.
    
    Parameters:
    X: features
    y: labels
    k_folds: number of folds
    model_class: class to instantiate for each fold
    **model_params: parameters for model initialization
    
    Returns:
    cv_scores: list of scores for each fold
    mean_score: average score across folds
    std_score: standard deviation of scores
    """
    # Your implementation here
    # Remember to shuffle data and create balanced folds
    pass

# Test your implementation
```

#### Task 2: Stratified Cross-Validation Implementation

```python
def stratified_k_fold_cv(X, y, k_folds, model_class, **model_params):
    """
    Implement stratified k-fold cross-validation to ensure balanced class distribution.
    
    Parameters:
    X: features
    y: labels
    k_folds: number of folds
    model_class: class to instantiate
    **model_params: model parameters
    
    Returns:
    cv_scores: scores for each fold
    fold_distributions: class distribution in each fold
    """
    # Your implementation here
    pass

# Compare regular vs stratified cross-validation
```

#### Task 3: Leave-One-Out Cross-Validation

```python
def leave_one_out_cv(X, y, model_class, **model_params):
    """
    Implement Leave-One-Out Cross-Validation.
    
    This is particularly interesting for KNN since removing one point
    can significantly change local neighborhoods.
    """
    # Your implementation here
    pass

# Analyze when LOOCV is preferable to k-fold CV
```

### Tasks 4-6: Comprehensive Accuracy Metrics Implementation

#### Task 4: Classification Metrics from Scratch

```python
def calculate_confusion_matrix(y_true, y_pred, labels=None):
    """
    Calculate confusion matrix manually.
    """
    # Your implementation here
    pass

def calculate_precision_recall_f1(y_true, y_pred, average='macro'):
    """
    Calculate precision, recall, and F1-score from scratch.
    Handle different averaging strategies: 'macro', 'micro', 'weighted'
    """
    # Your implementation here
    pass

def calculate_specificity_sensitivity(y_true, y_pred):
    """
    Calculate specificity and sensitivity (recall) for binary classification.
    """
    # Your implementation here
    pass

# Test with multi-class data
```

#### Task 5: ROC Curve and AUC Implementation

```python
def calculate_roc_curve(y_true, y_scores):
    """
    Calculate ROC curve points manually.
    
    Parameters:
    y_true: true binary labels
    y_scores: predicted probabilities or scores
    
    Returns:
    fpr: false positive rates
    tpr: true positive rates
    thresholds: decision thresholds
    """
    # Your implementation here
    pass

def calculate_auc(fpr, tpr):
    """
    Calculate Area Under Curve using trapezoidal rule.
    """
    # Your implementation here
    pass

# Create ROC curves for different K values in KNN
```

#### Task 6: Regression Metrics Implementation

```python
def regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics from scratch.
    
    Returns:
    dict with MSE, RMSE, MAE, R², Adjusted R², MAPE
    """
    # Your implementation here
    pass

def calculate_prediction_intervals(X_train, y_train, X_test, k, confidence=0.95):
    """
    Calculate prediction intervals for KNN regression.
    Use the variance among k nearest neighbors to estimate uncertainty.
    """
    # Your implementation here
    pass
```

### Tasks 7-10: Advanced Hyperparameter Optimization

#### Task 7: Grid Search Implementation

```python
def custom_grid_search_cv(X, y, model_class, param_grid, cv_folds=5, scoring='accuracy'):
    """
    Implement grid search with cross-validation from scratch.
    
    Parameters:
    X, y: data
    model_class: model to optimize
    param_grid: dictionary of parameters to search
    cv_folds: number of cross-validation folds
    scoring: scoring metric
    
    Returns:
    best_params: best parameter combination
    best_score: best cross-validation score
    all_results: detailed results for all combinations
    """
    # Your implementation here
    # Generate all parameter combinations
    # For each combination, perform cross-validation
    # Track best performance
    pass

# Example usage for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'distance_metric': ['euclidean', 'manhattan'],
    'weighted': [True, False]
}
```

#### Task 8: Random Search Implementation

```python
def custom_random_search_cv(X, y, model_class, param_distributions, n_iter=100, cv_folds=5):
    """
    Implement randomized search for hyperparameter optimization.
    Often more efficient than grid search for high-dimensional parameter spaces.
    """
    # Your implementation here
    pass

# Compare grid search vs random search efficiency
```


#### Task 10: Nested Cross-Validation

```python
def nested_cross_validation(X, y, model_class, param_grid, outer_cv=5, inner_cv=3):
    """
    Implement nested cross-validation for unbiased model evaluation.
    
    Outer loop: Model evaluation
    Inner loop: Hyperparameter selection
    
    Returns:
    outer_scores: unbiased performance estimates
    selected_params: parameters selected in each outer fold
    """
    # Your implementation here
    pass

# This gives you an unbiased estimate of model performance
```

### Tasks 11-13: Advanced KNN Variations

#### Task 11: Weighted KNN Implementation

```python
def distance_weighted_knn(X_train, y_train, X_test, k, distance_func, weight_func='inverse'):
    """
    Implement distance-weighted KNN where closer neighbors have more influence.
    
    Weight functions:
    - 'inverse': 1/distance
    - 'gaussian': exp(-distance²/2σ²)
    - 'linear': 1 - distance/max_distance
    """
    # Your implementation here
    pass

# Compare weighted vs unweighted KNN performance
```

#### Task 12: Locally Weighted KNN

```python
def locally_weighted_knn(X_train, y_train, X_test, bandwidth):
    """
    Implement locally weighted KNN (also known as locally weighted regression).
    Instead of using k neighbors, use all points within a certain bandwidth.
    """
    # Your implementation here
    pass
```

#### Task 13: KNN with Feature Selection

```python
def knn_with_feature_selection(X, y, feature_selection_method='correlation', k_features=5):
    """
    Combine KNN with feature selection techniques.
    
    Methods:
    - 'correlation': Select features most correlated with target
    - 'mutual_info': Use mutual information
    - 'recursive': Recursive feature elimination with KNN
    """
    # Your implementation here
    pass

# Analyze how feature selection affects KNN performance
```

### Tasks 14-15: Advanced Analysis and Comparison

#### Task 14: Curse of Dimensionality Analysis

```python
def analyze_curse_of_dimensionality():
    """
    Demonstrate how KNN performance degrades in high dimensions.
    Create datasets with increasing dimensionality and measure:
    1. Distance concentration (all points become equidistant)
    2. Classification performance
    3. Computational time
    """
    # Your implementation here
    # Create synthetic datasets with 2, 5, 10, 20, 50, 100 dimensions
    # Show how nearest neighbors become less meaningful
    pass

def visualize_distance_concentration(n_dimensions_list, n_samples=1000):
    """
    Visualize how distances between random points concentrate as dimensionality increases.
    """
    # Your implementation here
    pass
```

#### Task 15: Comprehensive Model Comparison

```python
def comprehensive_knn_analysis(datasets, models_to_compare):
    """
    Perform a comprehensive comparison of:
    1. Your custom KNN vs sklearn KNN
    2. Different distance metrics
    3. Different K values
    4. KNN vs other algorithms (SVM, Random Forest, etc.)
    5. Performance across different dataset characteristics
    
    Create visualizations showing:
    - Learning curves
    - Validation curves
    - Performance vs dataset size
    - Performance vs dimensionality
    - Computational complexity analysis
    """
    # Your implementation here
    # This is your capstone analysis bringing everything together
    pass

# Final comprehensive analysis and conclusions
```

---

## Submission Guidelines

**What to Submit:**
1. Complete Jupyter notebook with all implementations and analysis
2. Well-commented code demonstrating understanding
3. Detailed explanations of your findings and insights
4. Visualizations supporting your analysis
5. Discussion of when KNN is appropriate vs inappropriate

**Evaluation Criteria:**
- Correctness of manual calculations and implementations
- Quality of code structure and documentation  
- Depth of analysis and insights
- Understanding demonstrated through explanations
- Creativity in exploration and visualization

**Bonus Challenges:**
- Implement KNN for time series data
- Create an interactive visualization of KNN decision boundaries
- Analyze KNN performance on imbalanced datasets
- Implement approximate nearest neighbor algorithms for efficiency

Remember: The goal is not just to complete the tasks, but to develop deep understanding of how KNN works, when to use it, and how to optimize it for different scenarios. Take time to experiment, visualize results, and think critically about what you observe.

Good luck with your comprehensive exploration of the K-Nearest Neighbors algorithm!
# Decision Trees Algorithm: Comprehensive Learning Assignment

## Assignment Overview

Welcome to your comprehensive exploration of Decision Trees! This assignment will take you on a fascinating journey through one of the most intuitive and interpretable machine learning algorithms. Decision trees mirror how humans naturally make decisions - by asking a series of yes/no questions that progressively narrow down to a conclusion.

What makes decision trees particularly beautiful for learning is their transparency. Unlike many machine learning algorithms that operate as "black boxes," decision trees show you exactly how they arrive at their predictions. You can trace the path from root to leaf and understand every decision point. This interpretability makes them not only powerful tools for prediction but also valuable instruments for understanding your data and discovering hidden patterns.

Throughout this assignment, you'll discover how seemingly simple concepts like "which question should I ask first?" lead to sophisticated mathematical frameworks involving entropy, information gain, and optimization. You'll see how the recursive nature of tree building creates complex decision boundaries from simple binary splits, and you'll grapple with fundamental machine learning challenges like overfitting and the bias-variance tradeoff.

**Learning Journey:**
You'll begin by manually constructing decision trees to build intuition about how splits are chosen and how trees make predictions. Then you'll implement the algorithm from scratch, understanding every component from information gain calculations to tree traversal. Finally, you'll master advanced concepts like pruning, ensemble methods, and hyperparameter optimization that separate novice users from expert practitioners **(Optional)**.

**Part 4 is mainly advanced topics and is not required but highlt recommended.** 


**Required Libraries:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import graphviz
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# For visualizing trees
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
```

---

## Part 1: Manual Decision Tree Construction - Building Intuition (5 Questions)

Understanding decision trees begins with grasping how we choose the best questions to ask and how we measure the quality of our splits. These manual exercises will build the foundational intuition you need to understand why certain mathematical concepts like entropy and information gain are central to tree construction.

### Question 1: Understanding Information and Entropy

You are analyzing a small dataset of weather conditions to predict whether people will play tennis outside. This classic example helps illustrate how decision trees quantify uncertainty and information.

**Training Data:**
| Day | Outlook | Temperature | Humidity | Wind | Play Tennis |
|-----|---------|-------------|----------|------|-------------|
| 1   | Sunny   | Hot         | High     | Weak | No          |
| 2   | Sunny   | Hot         | High     | Strong | No        |
| 3   | Overcast| Hot         | High     | Weak | Yes         |
| 4   | Rain    | Mild        | High     | Weak | Yes         |
| 5   | Rain    | Cool        | Normal   | Weak | Yes         |
| 6   | Rain    | Cool        | Normal   | Strong | No        |
| 7   | Overcast| Cool        | Normal   | Strong | Yes        |
| 8   | Sunny   | Mild        | High     | Weak | No          |
| 9   | Sunny   | Cool        | Normal   | Weak | Yes         |
| 10  | Rain    | Mild        | Normal   | Weak | Yes         |
| 11  | Sunny   | Mild        | Normal   | Strong | Yes        |
| 12  | Overcast| Mild        | High     | Strong | Yes        |
| 13  | Overcast| Hot         | Normal   | Weak | Yes         |
| 14  | Rain    | Mild        | High     | Strong | No        |

**Your Task:** Calculate the entropy of the entire dataset with respect to the "Play Tennis" attribute. Remember that entropy measures the uncertainty or "impurity" in your data. A pure set (all same class) has entropy 0, while maximum uncertainty occurs when classes are equally distributed.

**Entropy Formula:** H(S) = -Σ(p_i * log₂(p_i)) where p_i is the proportion of examples belonging to class i.

**Guiding Questions to Consider:**
- How many examples are there for each class (Yes/No)?
- What does it mean intuitively when entropy is high versus low?
- How would entropy change if all examples had the same label?

**Space for your calculations:**

```python
# You can use this space to verify your manual calculations
# But first, work through the entropy calculation by hand!

```

### Question 2: Information Gain and Feature Selection

Now you'll determine which attribute would be the best choice for the root node of your decision tree. This involves calculating information gain for each possible attribute.

**Your Task:** Calculate the information gain for splitting on each attribute (Outlook, Temperature, Humidity, Wind). Information gain measures how much uncertainty is reduced by knowing the value of an attribute.

**Information Gain Formula:** Gain(S,A) = H(S) - Σ((|Sv|/|S|) * H(Sv))
where Sv is the subset of S for which attribute A has value v.

**Steps to follow:**
1. For each attribute, group the examples by the attribute's values
2. Calculate the entropy of each subset
3. Calculate the weighted average entropy after splitting
4. Subtract from the original entropy to get information gain

**Detailed Analysis Required:**
- Which attribute gives the highest information gain?
- Intuitively explain why this attribute is the best choice for the first split
- What would happen if you chose an attribute with lower information gain?

**Advanced Thinking:** Consider what happens when an attribute perfectly separates the classes versus when it provides no separation at all. How do these scenarios relate to the information gain values you calculate?

### Question 3: Constructing the Complete Tree

Using your information gain calculations from Question 2, manually construct the complete decision tree for the tennis dataset.

**Your Task:** 
1. Choose the root node based on your information gain calculations
2. For each branch from the root, recursively apply the same process to the remaining data
3. Continue until you reach pure leaf nodes or run out of attributes
4. Draw the complete tree structure showing all internal nodes and leaf nodes

**Tree Construction Guidelines:**
- At each internal node, show the attribute being tested
- Label each branch with the attribute value that leads down that path
- Label each leaf with the predicted class and the supporting examples
- Track how the entropy decreases as you move down the tree

**Critical Thinking Questions:**
- At what point do you stop splitting? What criteria determine when a node becomes a leaf?
- How does the tree structure reflect the importance of different attributes?
- Can you trace through the tree to classify a new example?

### Question 4: Handling Continuous Attributes

Modify the tennis dataset by replacing the categorical "Temperature" attribute with actual temperature values: Hot=85°F, Mild=75°F, Cool=65°F.

**Your Task:** Determine the best threshold for splitting on temperature. Unlike categorical attributes, continuous attributes require finding optimal cut-points.

**Process to Follow:**
1. Sort all examples by temperature value
2. Consider all possible split points (typically midpoints between consecutive values)
3. For each potential split, calculate the information gain
4. Choose the split that maximizes information gain

**New Dataset with Continuous Temperature:**
| Day | Temperature | Play Tennis |
|-----|-------------|-------------|
| 1   | 85          | No          |
| 2   | 85          | No          |
| 3   | 85          | Yes         |
| ... | ...         | ...         |

**Analysis Questions:**
- What temperature threshold gives the best split?
- How does handling continuous attributes change the tree construction process?
- What are the computational implications of continuous versus categorical attributes?

### Question 5: Gini Impurity Alternative

Calculate the same splits using Gini impurity instead of entropy, and compare the results.

**Gini Impurity Formula:** Gini(S) = 1 - Σ(p_i²) where p_i is the proportion of examples in class i.

**Your Task:**
1. Recalculate the impurity of the original dataset using Gini instead of entropy
2. Recalculate the impurity reduction (analogous to information gain) for each attribute
3. Compare the attribute rankings using Gini versus entropy
4. Discuss when you might prefer one measure over the other

**Comparative Analysis:**
- Do Gini and entropy give the same attribute rankings?
- Which measure is easier to compute, and why might this matter?
- How do the two measures handle extreme cases (very pure or very mixed nodes)?

**Conceptual Understanding:** Both entropy and Gini impurity measure node "impurity," but they have different mathematical properties. Entropy has a stronger theoretical foundation in information theory, while Gini is computationally simpler. Understanding when to use each measure is an important practical skill.

---

## Part 2: Implementing Decision Trees from Scratch (7-Step Implementation)

Now you'll build a complete decision tree implementation from the ground up. This hands-on construction will deepen your understanding of every algorithmic component and give you the flexibility to modify the algorithm for specific needs.

### Step 1: Node Structure and Basic Classes

Begin by defining the fundamental building blocks of your decision tree. The elegance of decision trees lies in their recursive structure - each subtree is itself a complete decision tree.

```python
class TreeNode:
    """
    Represents a single node in the decision tree.
    
    A node can be either:
    1. Internal node: contains a decision rule (attribute + threshold/value)
    2. Leaf node: contains a prediction (class label or regression value)
    """
    
    def __init__(self, is_leaf=False):
        """
        Initialize a tree node.
        
        Parameters:
        is_leaf: bool, whether this node is a leaf (terminal) node
        """
        # Basic node properties
        self.is_leaf = is_leaf
        
        # For internal nodes - decision rule
        self.feature_index = None      # Which feature to split on
        self.threshold = None          # Split threshold for continuous features
        self.feature_value = None      # Split value for categorical features
        
        # For leaf nodes - prediction
        self.prediction = None         # Class label or regression value
        self.class_counts = None       # For classification: count of each class
        
        # Tree structure
        self.left = None               # Left child (True/<=threshold branch)
        self.right = None              # Right child (False/>threshold branch)
        self.children = {}             # For categorical splits: value -> child mapping
        
        # Metadata for analysis
        self.samples = 0               # Number of training samples at this node
        self.impurity = 0.0           # Impurity measure at this node
        self.depth = 0                # Depth of this node in the tree
    
    def __repr__(self):
        """String representation for debugging."""
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction}, samples={self.samples})"
        else:
            return f"Node(feature={self.feature_index}, threshold={self.threshold}, samples={self.samples})"

# Test your node implementation
root = TreeNode()
leaf = TreeNode(is_leaf=True)
print("Node classes created successfully!")
print(f"Root node: {root}")
print(f"Leaf node: {leaf}")
```

### Step 2: Impurity Measures Implementation

Implement the mathematical foundations for measuring node quality. These functions determine how "pure" or "mixed" a set of examples is, which drives the entire tree construction process.

```python
def calculate_entropy(y):
    """
    Calculate the entropy of a set of labels.
    
    Entropy measures uncertainty: 0 for pure sets, maximum for uniform distribution.
    This is the foundation of information-theoretic tree building.
    
    Parameters:
    y: array-like, class labels
    
    Returns:
    float: entropy value
    """
    # Your implementation here
    # Steps:
    # 1. Count occurrences of each class
    # 2. Calculate proportions
    # 3. Apply entropy formula: -Σ(p_i * log₂(p_i))
    # 4. Handle edge case where p_i = 0 (log is undefined)
    pass

def calculate_gini_impurity(y):
    """
    Calculate the Gini impurity of a set of labels.
    
    Gini impurity measures the probability of misclassifying a randomly
    chosen element if it were randomly labeled according to the distribution.
    
    Parameters:
    y: array-like, class labels
    
    Returns:
    float: Gini impurity value
    """
    # Your implementation here
    # Formula: 1 - Σ(p_i²)
    pass

def calculate_mse(y):
    """
    Calculate Mean Squared Error for regression trees.
    
    For regression, we measure impurity as the variance of target values.
    Lower variance means more homogeneous predictions.
    
    Parameters:
    y: array-like, target values
    
    Returns:
    float: MSE value
    """
    # Your implementation here
    # Formula: (1/n) * Σ(y_i - ȳ)²
    pass

# Test your impurity functions
test_labels_pure = np.array([1, 1, 1, 1])
test_labels_mixed = np.array([1, 0, 1, 0])
test_labels_skewed = np.array([1, 1, 1, 0])

print("Testing impurity measures:")
print(f"Pure set entropy: {calculate_entropy(test_labels_pure)}")
print(f"Mixed set entropy: {calculate_entropy(test_labels_mixed)}")
print(f"Skewed set entropy: {calculate_entropy(test_labels_skewed)}")

# Add tests for Gini and MSE
```

### Step 3: Feature Splitting Logic

Implement the core logic for finding the best way to split data. This is where the algorithm decides which questions to ask and how to ask them.

```python
def find_best_split_categorical(X_column, y, impurity_func):
    """
    Find the best split for a categorical feature.
    
    For categorical features, we try splitting on each possible value,
    creating one branch for that value and another for all other values.
    
    Parameters:
    X_column: array-like, values of a single categorical feature
    y: array-like, target labels
    impurity_func: function, impurity measure to use
    
    Returns:
    best_value: value to split on
    best_impurity_reduction: improvement in impurity
    left_indices: indices of samples going to left child
    right_indices: indices of samples going to right child
    """
    # Your implementation here
    # Steps:
    # 1. Get unique values in the feature
    # 2. For each value, create binary split (value vs. not value)
    # 3. Calculate weighted impurity after split
    # 4. Choose split with maximum impurity reduction
    pass

def find_best_split_continuous(X_column, y, impurity_func):
    """
    Find the best split threshold for a continuous feature.
    
    We consider all possible thresholds (typically midpoints between
    consecutive sorted values) and choose the one that maximizes
    information gain.
    
    Parameters:
    X_column: array-like, values of a single continuous feature
    y: array-like, target labels
    impurity_func: function, impurity measure to use
    
    Returns:
    best_threshold: threshold value for split
    best_impurity_reduction: improvement in impurity
    left_indices: indices of samples with value <= threshold
    right_indices: indices of samples with value > threshold
    """
    # Your implementation here
    # Steps:
    # 1. Sort data by feature values
    # 2. Consider splits between consecutive unique values
    # 3. For each potential threshold, calculate impurity reduction
    # 4. Return best threshold and corresponding split
    pass

def find_best_feature_split(X, y, feature_types, impurity_func):
    """
    Find the best feature and split point across all features.
    
    This function orchestrates the search across all possible features
    and returns the overall best split for building the tree.
    
    Parameters:
    X: array-like, shape (n_samples, n_features), feature matrix
    y: array-like, target values
    feature_types: list, 'categorical' or 'continuous' for each feature
    impurity_func: function, impurity measure
    
    Returns:
    best_feature: index of best feature to split on
    best_split_info: dictionary with split details
    """
    # Your implementation here
    # Combine categorical and continuous splitting logic
    # Return comprehensive information about the best split found
    pass

# Test splitting functions with sample data
sample_X = np.array([[1, 2.5], [2, 1.8], [1, 3.2], [3, 2.1], [2, 2.9]])
sample_y = np.array([0, 0, 1, 1, 1])
feature_types = ['categorical', 'continuous']

print("Testing split finding...")
# Add your test code here
```

### Step 4: Tree Building Algorithm

Implement the recursive tree construction algorithm. This is the heart of decision tree learning, where the magic of automatic rule discovery happens.

```python
class DecisionTreeClassifier:
    """
    Custom implementation of Decision Tree Classifier.
    
    This implementation provides full control over the tree building process
    and helps you understand every decision made during construction.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 impurity_measure='entropy', feature_types=None):
        """
        Initialize the decision tree classifier.
        
        Parameters:
        max_depth: int or None, maximum depth of the tree
        min_samples_split: int, minimum samples required to split a node
        min_samples_leaf: int, minimum samples required at a leaf node
        impurity_measure: str, 'entropy' or 'gini'
        feature_types: list, type of each feature ('categorical' or 'continuous')
        """
        # Your implementation here
        # Store hyperparameters and initialize tree structure
        pass
    
    def _build_tree(self, X, y, depth=0, node_indices=None):
        """
        Recursively build the decision tree.
        
        This is the core recursive algorithm that creates the tree structure
        by repeatedly finding the best splits and creating child nodes.
        
        Parameters:
        X: feature matrix for current subset
        y: target values for current subset  
        depth: current depth in the tree
        node_indices: indices of samples at this node
        
        Returns:
        TreeNode: the constructed node (internal or leaf)
        """
        # Your implementation here
        # Algorithm:
        # 1. Check stopping criteria (depth, sample count, purity)
        # 2. If stopping, create leaf node with majority class
        # 3. Otherwise, find best split using your splitting functions
        # 4. Create internal node and recursively build children
        # 5. Return the constructed node
        pass
    
    def fit(self, X, y):
        """
        Build the decision tree from training data.
        
        Parameters:
        X: array-like, shape (n_samples, n_features), training features
        y: array-like, shape (n_samples,), training labels
        """
        # Your implementation here
        # Prepare data and call _build_tree
        pass
    
    def _predict_sample(self, x, node):
        """
        Predict the class for a single sample by traversing the tree.
        
        Parameters:
        x: single sample (feature vector)
        node: current node in traversal
        
        Returns:
        prediction: predicted class label
        """
        # Your implementation here
        # Recursive traversal from root to leaf
        pass
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        
        Parameters:
        X: array-like, shape (n_samples, n_features), test features
        
        Returns:
        predictions: array of predicted class labels
        """
        # Your implementation here
        pass

# Test your decision tree implementation
print("Testing custom Decision Tree...")
# Create simple test data and verify your implementation works
```

### Step 5: Tree Visualization and Interpretation

Implement functions to visualize and interpret your decision trees. Understanding what the tree has learned is crucial for both debugging and gaining insights from your model.

```python
def print_tree(node, feature_names=None, class_names=None, indent="", max_depth=10):
    """
    Print a text representation of the decision tree.
    
    This function creates a human-readable representation that shows
    the decision rules learned by the tree.
    
    Parameters:
    node: TreeNode, root of tree/subtree to print
    feature_names: list, names of features for readable output
    class_names: list, names of classes for readable output
    indent: str, current indentation level
    max_depth: int, maximum depth to print (prevents overly long output)
    """
    # Your implementation here
    # Recursively print tree structure with proper indentation
    # Show decision rules at internal nodes and predictions at leaves
    pass

def visualize_tree_structure(node, feature_names=None, class_names=None):
    """
    Create a visual representation of the tree structure.
    
    Generate a matplotlib-based visualization showing the tree structure,
    decision boundaries, and class distributions at each node.
    """
    # Your implementation here
    # Create a graphical representation of the tree
    # This is more complex but very valuable for understanding
    pass

def extract_decision_rules(node, feature_names=None, path="", rules_list=None):
    """
    Extract all decision rules from the tree as text.
    
    Convert the tree into a set of if-then rules that can be easily
    understood and potentially used outside the tree structure.
    
    Parameters:
    node: TreeNode, current node
    feature_names: list, feature names
    path: str, current path conditions
    rules_list: list, accumulator for rules
    
    Returns:
    list: all decision rules as strings
    """
    # Your implementation here
    # Extract rules by traversing all root-to-leaf paths
    pass

# Test visualization functions
print("Testing tree visualization...")
# Build a simple tree and test your visualization functions
```

### Step 6: Regression Tree Implementation

Extend your implementation to handle regression problems. This demonstrates how the same algorithmic framework adapts to different types of prediction tasks.

```python
class DecisionTreeRegressor:
    """
    Custom implementation of Decision Tree Regressor.
    
    Regression trees predict continuous values rather than discrete classes.
    The main differences are in the impurity measure (MSE instead of entropy)
    and prediction method (mean instead of majority vote).
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the decision tree regressor.
        """
        # Your implementation here
        # Similar to classifier but adapted for regression
        pass
    
    def _build_tree(self, X, y, depth=0):
        """
        Build regression tree using MSE as splitting criterion.
        
        Key differences from classification:
        - Use MSE to measure impurity
        - Leaf predictions are mean of target values
        - Consider variance reduction instead of information gain
        """
        # Your implementation here
        pass
    
    def fit(self, X, y):
        """Train the regression tree."""
        # Your implementation here
        pass
    
    def predict(self, X):
        """Predict continuous values."""
        # Your implementation here
        pass

# Test regression tree
print("Testing regression tree...")
# Create regression data and test your implementation
```

### Step 7: Advanced Features and Optimizations

Add sophisticated features that improve tree performance and provide more control over the learning process.

```python
def implement_pruning(tree_root, X_val, y_val, pruning_method='reduced_error'):
    """
    Implement tree pruning to reduce overfitting.
    
    Pruning removes subtrees that don't improve performance on validation data.
    This is crucial for creating trees that generalize well to new data.
    
    Parameters:
    tree_root: TreeNode, root of tree to prune
    X_val: validation features
    y_val: validation labels
    pruning_method: str, pruning strategy to use
    
    Returns:
    TreeNode: pruned tree root
    """
    # Your implementation here
    # Implement reduced error pruning or cost complexity pruning
    pass

def implement_feature_importance(tree_root, n_features):
    """
    Calculate feature importance scores based on tree structure.
    
    Feature importance measures how much each feature contributes
    to decreasing impurity across all splits in the tree.
    
    Parameters:
    tree_root: TreeNode, root of trained tree
    n_features: int, number of features
    
    Returns:
    array: importance scores for each feature
    """
    # Your implementation here
    # Traverse tree and accumulate impurity decreases for each feature
    pass

def implement_missing_value_handling(X, method='surrogate'):
    """
    Handle missing values in the dataset.
    
    Decision trees can handle missing values through surrogate splits
    or by treating missing as a separate category.
    
    Parameters:
    X: feature matrix with potential missing values
    method: str, method for handling missing values
    
    Returns:
    X_processed: processed feature matrix
    """
    # Your implementation here
    # Implement strategies for missing value handling
    pass

# Test advanced features
print("Testing advanced features...")
```

---

## Part 3: Real-World Applications (2 Tasks)

Now you'll apply decision trees to real datasets, experiencing how they perform on practical problems and understanding their strengths and limitations in different contexts.

### Task 1: Medical Diagnosis Classification

The breast cancer dataset provides an excellent example of how decision trees can be used for medical diagnosis. This binary classification problem has clear real-world importance and interpretability requirements.

```python
# Load and explore the breast cancer dataset
cancer_data = load_breast_cancer()
X_cancer, y_cancer = cancer_data.data, cancer_data.target

print("Breast Cancer Dataset Information:")
print(f"Number of samples: {X_cancer.shape[0]}")
print(f"Number of features: {X_cancer.shape[1]}")
print(f"Classes: {cancer_data.target_names}")
print(f"Class distribution: {np.bincount(y_cancer)}")

# Display feature information
feature_info = pd.DataFrame({
    'Feature': cancer_data.feature_names,
    'Mean': np.mean(X_cancer, axis=0),
    'Std': np.std(X_cancer, axis=0)
})
print("\nFirst 10 features:")
print(feature_info.head(10))

# Your comprehensive analysis tasks:

# 1. Exploratory Data Analysis
# Create visualizations showing:
# - Distribution of features between malignant and benign cases
# - Correlation matrix of features
# - Box plots comparing feature values across classes
# Start your EDA here:


# 2. Data Preprocessing
# Consider the following preprocessing steps:
# - Feature scaling (do decision trees need it?)
# - Handling potential outliers
# - Feature selection based on importance
# Implement preprocessing here:


# 3. Model Training and Comparison
# Train and compare:
# - Your custom decision tree implementation
# - Sklearn's DecisionTreeClassifier with different parameters
# - Different impurity measures (entropy vs gini)
# Implement model training here:


# 4. Interpretability Analysis
# This is where decision trees truly shine:
# - Extract and analyze the most important decision rules
# - Visualize the tree structure (limit depth for readability)
# - Identify which features are most crucial for diagnosis
# - Explain how the tree makes predictions for specific cases
# Implement interpretability analysis here:


# 5. Medical Validity Assessment
# Evaluate whether the tree's decisions align with medical knowledge:
# - Do the important features make medical sense?
# - Are the decision thresholds reasonable?
# - How might you present this model to medical professionals?
# Conduct medical validity assessment here:

```

### Task 2: Housing Price Prediction (Regression)

Create a comprehensive regression analysis using decision trees to predict housing prices. This task will deepen your understanding of how trees handle continuous target variables.

```python
# Create a realistic housing price dataset
np.random.seed(42)
n_samples = 1000

# Generate realistic housing features
square_feet = np.random.normal(2000, 600, n_samples)
bedrooms = np.random.poisson(3, n_samples) + 1
bathrooms = np.random.normal(2.5, 0.8, n_samples)
age = np.random.exponential(15, n_samples)
lot_size = np.random.lognormal(8, 0.5, n_samples)
garage = np.random.binomial(1, 0.7, n_samples)

# Create realistic neighborhood categories
neighborhoods = np.random.choice(['Downtown', 'Suburban', 'Rural', 'Waterfront'], 
                                n_samples, p=[0.3, 0.5, 0.15, 0.05])

# Generate realistic price based on features with some noise
base_price = (square_feet * 150 + 
              bedrooms * 10000 + 
              bathrooms * 15000 + 
              lot_size * 5 + 
              garage * 20000 - 
              age * 1000)

# Add neighborhood effects
neighborhood_effects = {'Downtown': 50000, 'Suburban': 0, 'Rural': -30000, 'Waterfront': 100000}
for i, neighborhood in enumerate(neighborhoods):
    base_price[i] += neighborhood_effects[neighborhood]

# Add realistic noise and ensure positive prices
price = base_price + np.random.normal(0, 25000, n_samples)
price = np.maximum(price, 50000)  # Minimum price floor

# Create feature matrix
X_housing = np.column_stack([square_feet, bedrooms, bathrooms, age, lot_size, garage])
feature_names = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age', 'Lot_Size', 'Garage']

# Add categorical neighborhood feature (you'll need to encode this)
housing_df = pd.DataFrame(X_housing, columns=feature_names)
housing_df['Neighborhood'] = neighborhoods
housing_df['Price'] = price

print("Housing Dataset Information:")
print(f"Number of samples: {len(housing_df)}")
print(f"Features: {list(housing_df.columns[:-1])}")
print(f"Price range: ${price.min():,.0f} - ${price.max():,.0f}")
print(f"Average price: ${price.mean():,.0f}")

# Your comprehensive regression analysis tasks:

# 1. Regression-Specific EDA
# Analyze relationships between features and target:
# - Scatter plots of each feature vs price
# - Correlation analysis
# - Distribution of price variable
# - Identify potential non-linear relationships
# Start your regression EDA here:


# 2. Feature Engineering for Trees
# Consider creating new features that trees might find useful:
# - Binning continuous variables
# - Creating interaction features
# - Price per square foot ratios
# Implement feature engineering here:


# 3. Regression Tree Implementation
# Apply your custom regression tree and sklearn's version:
# - Compare different max_depth values
# - Analyze how tree depth affects bias-variance tradeoff
# - Implement and test tree pruning
# Implement regression tree analysis here:


# 4. Regression Evaluation
# Use appropriate regression metrics:
# - MSE, MAE, R² score
# - Residual analysis
# - Cross-validation for robust evaluation
# - Learning curves to understand overfitting
# Implement comprehensive evaluation here:


# 5. Tree Interpretation for Regression
# Understand what the tree learned:
# - Which features are most important for price prediction?
# - What are the key price ranges and decision thresholds?
# - How do different neighborhoods get classified?
# - Extract actionable insights for real estate professionals
# Implement regression tree interpretation here:

```

---

## Part 4**: Advanced Topics and Hyperparameter Tuning (15 Tasks)

This section will transform you from a basic decision tree user into an expert practitioner who understands the nuances, limitations, and advanced techniques that separate professional machine learning from academic exercises.

### Tasks 1-3: Understanding Overfitting and Tree Complexity

#### Task 1: Empirical Analysis of Overfitting

Decision trees are notorious for overfitting. This task will help you understand exactly how and why this happens, and how to detect it.

```python
def analyze_overfitting_behavior(X, y, max_depths=range(1, 21), cv_folds=5):
    """
    Comprehensive analysis of how tree depth affects overfitting.
    
    This function will create learning curves and validation curves
    to visualize the bias-variance tradeoff in decision trees.
    
    Parameters:
    X, y: dataset
    max_depths: range of tree depths to test
    cv_folds: number of cross-validation folds
    
    Returns:
    analysis_results: comprehensive overfitting analysis
    """
    # Your implementation here
    # Create training and validation curves
    # Show how performance changes with tree depth
    # Identify the optimal depth for generalization
    pass

def demonstrate_overfitting_mechanisms():
    """
    Create visualizations showing specific ways decision trees overfit:
    1. Creating overly complex boundaries for simple problems
    2. Memorizing noise in the training data
    3. Building unnecessarily deep trees for small datasets
    """
    # Your implementation here
    # Create synthetic datasets that clearly show overfitting
    # Use 2D data so you can visualize decision boundaries
    pass

# Implement comprehensive overfitting analysis
print("Analyzing overfitting behavior...")
```

#### Task 2: Implementing Multiple Pruning Strategies

```python
def reduced_error_pruning(tree_root, X_val, y_val):
    """
    Implement reduced error pruning (post-pruning).
    
    This method removes subtrees that don't improve validation accuracy.
    It's a greedy algorithm that considers pruning each internal node.
    
    Returns:
    pruned_tree: TreeNode after pruning
    pruning_log: detailed log of pruning decisions
    """
    # Your implementation here
    # Algorithm:
    # 1. For each internal node, calculate validation accuracy if pruned
    # 2. If pruning improves or maintains accuracy, prune the subtree
    # 3. Repeat until no beneficial pruning is possible
    pass

def cost_complexity_pruning(tree_root, X_train, y_train, X_val, y_val):
    """
    Implement cost complexity pruning (also known as minimal error pruning).
    
    This method uses a complexity parameter α to balance tree size and accuracy.
    It's the method used by sklearn's DecisionTreeClassifier.
    
    Returns:
    pruning_path: sequence of pruned trees for different α values
    optimal_alpha: best α value based on validation performance
    """
    # Your implementation here
    # This is more complex but gives better theoretical guarantees
    pass

def compare_pruning_methods(X, y, test_size=0.3):
    """
    Compare different pruning strategies on the same dataset.
    
    Analyze:
    - Final tree size after pruning
    - Generalization performance
    - Computational cost of pruning
    """
    # Your implementation here
    pass
```

#### Task 3: Bias-Variance Decomposition

```python
def bias_variance_decomposition(X, y, n_bootstrap=100, test_size=0.3):
    """
    Perform bias-variance decomposition for decision trees.
    
    This analysis helps understand the fundamental tradeoff in tree complexity:
    - High bias (underfitting): overly simple trees
    - High variance (overfitting): overly complex trees
    
    Parameters:
    X, y: dataset
    n_bootstrap: number of bootstrap samples for analysis
    test_size: proportion of data for testing
    
    Returns:
    bias: bias component of the error
    variance: variance component of the error
    noise: irreducible error component
    """
    # Your implementation here
    # Use bootstrap sampling to estimate bias and variance
    # Show how tree depth affects each component
    pass

# Visualize bias-variance tradeoff
def visualize_bias_variance_tradeoff():
    """
    Create comprehensive visualizations showing:
    1. How bias decreases and variance increases with tree complexity
    2. The sweet spot that minimizes total error
    3. How dataset size affects the optimal complexity
    """
    # Your implementation here
    pass
```

### Tasks 4-6: Feature Importance and Selection

#### Task 4: Multiple Feature Importance Metrics

```python
def calculate_impurity_based_importance(tree_root, n_features):
    """
    Calculate feature importance based on impurity decrease.
    
    This is the standard method: features that create larger decreases
    in impurity when used for splitting are considered more important.
    """
    # Your implementation here
    pass

def calculate_permutation_importance(model, X, y, scoring_func, n_repeats=10):
    """
    Calculate permutation-based feature importance.
    
    This method measures importance by observing how much performance
    decreases when each feature's values are randomly shuffled.
    """
    # Your implementation here
    # For each feature:
    # 1. Randomly permute its values
    # 2. Measure performance decrease
    # 3. Repeat and average
    pass

def calculate_shap_values_approximation(tree_root, X_sample):
    """
    Implement a simplified version of SHAP (SHapley Additive exPlanations).
    
    SHAP values provide a unified framework for feature importance
    that satisfies certain mathematical properties.
    """
    # Your implementation here
    # This is advanced - focus on understanding the concept
    pass

def compare_importance_methods():
    """
    Compare different feature importance methods and discuss:
    - When they agree vs disagree
    - Computational costs
    - Interpretability for stakeholders
    """
    # Your implementation here
    pass
```

#### Task 5: Feature Selection Integration

```python
def recursive_feature_elimination_trees(X, y, n_features_to_select):
    """
    Implement recursive feature elimination using decision trees.
    
    This method iteratively removes the least important features
    and retrains the model until the desired number remains.
    """
    # Your implementation here
    pass

def mutual_information_feature_selection(X, y, k_best):
    """
    Select features based on mutual information with the target.
    
    Mutual information measures how much knowing a feature's value
    reduces uncertainty about the target variable.
    """
    # Your implementation here
    pass

def tree_based_feature_selection_pipeline():
    """
    Create a complete pipeline that:
    1. Ranks features by multiple criteria
    2. Selects optimal subset using cross-validation
    3. Evaluates impact on model performance
    """
    # Your implementation here
    pass
```

#### Task 6: Feature Interaction Analysis

```python
def detect_feature_interactions(tree_root, feature_names):
    """
    Analyze the decision tree to identify important feature interactions.
    
    Features that appear together in the same path from root to leaf
    are potentially interacting in their effect on the target.
    """
    # Your implementation here
    # Extract all root-to-leaf paths
    # Identify frequently co-occurring features
    pass

def create_interaction_features(X, feature_names, max_degree=2):
    """
    Create explicit interaction features for tree models.
    
    While trees can capture interactions implicitly, explicit interaction
    features can sometimes improve performance and interpretability.
    """
    # Your implementation here
    pass
```

### Tasks 7-9: Cross-Validation and Model Selection

#### Task 7: Advanced Cross-Validation Strategies

```python
def time_series_cross_validation(X, y, time_index, n_splits=5):
    """
    Implement time series cross-validation for temporal data.
    
    Unlike random splits, this respects temporal order:
    training always occurs before testing.
    """
    # Your implementation here
    # Create splits that respect temporal order
    # Avoid data leakage from future to past
    pass

def stratified_group_k_fold(X, y, groups, n_splits=5):
    """
    Implement stratified group k-fold cross-validation.
    
    This ensures that:
    1. Class proportions are maintained (stratified)
    2. Related samples stay together (grouped)
    """
    # Your implementation here
    pass

def nested_cross_validation_with_trees(X, y, param_grid, outer_cv=5, inner_cv=3):
    """
    Implement nested cross-validation for unbiased performance estimation.
    
    Outer loop: Estimate model performance
    Inner loop: Select hyperparameters
    
    This provides unbiased estimates of how well your model will
    perform on truly unseen data.
    """
    # Your implementation here
    pass
```

#### Task 8: Custom Scoring Functions

```python
def cost_sensitive_scoring(y_true, y_pred, cost_matrix):
    """
    Implement cost-sensitive evaluation for imbalanced problems.
    
    In many real-world applications, different types of errors
    have different costs (e.g., false negatives in medical diagnosis).
    """
    # Your implementation here
    pass

def profit_based_scoring(y_true, y_pred_proba, profit_matrix):
    """
    Implement profit-based evaluation for business applications.
    
    Optimize for business metrics rather than traditional ML metrics.
    """
    # Your implementation here
    pass

def fairness_aware_scoring(y_true, y_pred, sensitive_attribute):
    """
    Implement fairness-aware evaluation metrics.
    
    Measure whether the model makes fair predictions across
    different demographic groups.
    """
    # Your implementation here
    pass
```

#### Task 9: Automated Hyperparameter Tuning

```python
def bayesian_optimization_trees(X, y, param_space, n_calls=50):
    """
    Implement Bayesian optimization for hyperparameter tuning.
    
    This is more efficient than grid search for expensive evaluations.
    Use Gaussian processes to model the objective function.
    """
    # Your implementation here
    # This is advanced - you may use existing libraries
    pass

def multi_objective_optimization(X, y, objectives=['accuracy', 'tree_size']):
    """
    Optimize for multiple objectives simultaneously.
    
    In practice, we often want to balance performance with interpretability,
    speed, or other criteria.
    """
    # Your implementation here
    pass
```

### Tasks 10-12: Ensemble Methods and Advanced Trees

#### Task 10: Bootstrap Aggregating (Bagging)

```python
def implement_bagging_from_scratch(base_estimator_class, n_estimators=10, max_samples=1.0):
    """
    Implement bootstrap aggregating (bagging) from scratch.
    
    Bagging reduces variance by training multiple trees on bootstrap
    samples and averaging their predictions.
    """
    class BaggingClassifier:
        def __init__(self, base_estimator_class, n_estimators, max_samples):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            # Your implementation here
            # Create bootstrap samples and train estimators
            pass
        
        def predict(self, X):
            # Your implementation here
            # Average predictions from all estimators
            pass
    
    return BaggingClassifier(base_estimator_class, n_estimators, max_samples)

# Test your bagging implementation
```

#### Task 11: Random Forest from Scratch

```python
def implement_random_forest_from_scratch(n_estimators=10, max_features='sqrt', max_depth=None):
    """
    Implement Random Forest from scratch.
    
    Random Forest combines bagging with random feature selection
    to create even more diverse trees.
    """
    class RandomForestClassifier:
        def __init__(self, n_estimators, max_features, max_depth):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            # Your implementation here
            # Implement bootstrap sampling + random feature selection
            pass
        
        def predict(self, X):
            # Your implementation here
            pass
        
        def feature_importances_(self):
            # Your implementation here
            # Aggregate feature importance across all trees
            pass
    
    return RandomForestClassifier(n_estimators, max_features, max_depth)

def analyze_random_forest_behavior():
    """
    Analyze how Random Forest improves upon single trees:
    1. Variance reduction through averaging
    2. Bias introduction through feature randomness
    3. Out-of-bag error estimation
    """
    # Your implementation here
    pass
```

#### Task 12: Gradient Boosting Fundamentals

```python
def implement_gradient_boosting_trees(n_estimators=10, learning_rate=0.1, max_depth=3):
    """
    Implement a simplified version of Gradient Boosting Trees.
    
    Unlike bagging, boosting trains trees sequentially,
    with each tree learning from the mistakes of previous trees.
    """
    class SimpleGradientBoostingClassifier:
        def __init__(self, n_estimators, learning_rate, max_depth):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            # Your implementation here
            # Sequential training with residual fitting
            pass
        
        def predict(self, X):
            # Your implementation here
            pass
    
    return SimpleGradientBoostingClassifier(n_estimators, learning_rate, max_depth)

def compare_ensemble_methods():
    """
    Compare bagging, random forest, and gradient boosting:
    - Bias-variance characteristics
    - Training time and parallelizability
    - Interpretability
    - Performance on different types of problems
    """
    # Your implementation here
    pass
```

### Tasks 13-15: Advanced Analysis and Real-World Considerations

#### Task 13: Handling Imbalanced Datasets

```python
def class_weight_adjustment(y, method='balanced'):
    """
    Calculate class weights for imbalanced datasets.
    
    Adjust the tree building process to give more importance
    to minority classes.
    """
    # Your implementation here
    pass

def cost_sensitive_tree_building(X, y, cost_matrix):
    """
    Modify tree building to consider misclassification costs.
    
    Instead of optimizing accuracy, optimize expected cost.
    """
    # Your implementation here
    pass

def smote_with_trees(X, y, sampling_strategy='auto'):
    """
    Combine SMOTE (Synthetic Minority Oversampling Technique) with trees.
    
    Generate synthetic examples of minority classes to balance the dataset.
    """
    # Your implementation here
    pass

def threshold_optimization_for_imbalanced_data():
    """
    Optimize classification thresholds for imbalanced problems.
    
    The default 0.5 threshold is often suboptimal for imbalanced data.
    """
    # Your implementation here
    pass
```

#### Task 14: Computational Efficiency and Scalability

```python
def analyze_computational_complexity():
    """
    Analyze the computational complexity of decision tree algorithms:
    
    1. Training complexity: O(n * m * log(n)) where n=samples, m=features
    2. Prediction complexity: O(depth)
    3. Memory complexity: O(nodes)
    
    Measure actual runtime vs theoretical complexity.
    """
    # Your implementation here
    # Create datasets of varying sizes and measure performance
    pass

def implement_early_stopping(X_train, y_train, X_val, y_val, patience=5):
    """
    Implement early stopping for tree building.
    
    Stop growing the tree when validation performance stops improving.
    """
    # Your implementation here
    pass

def parallel_tree_building():
    """
    Explore parallelization opportunities in decision trees:
    
    1. Feature evaluation can be parallelized
    2. Subtree building can be parallelized
    3. Ensemble methods are embarrassingly parallel
    """
    # Your implementation here
    pass
```

#### Task 15: Production Deployment Considerations

```python
def model_serialization_and_deployment():
    """
    Implement model serialization for production deployment.
    
    Consider:
    1. Model size and memory requirements
    2. Prediction speed requirements
    3. Model versioning and updates
    4. A/B testing frameworks
    """
    # Your implementation here
    pass

def model_monitoring_and_drift_detection():
    """
    Implement monitoring for deployed tree models.
    
    Monitor:
    1. Feature distribution drift
    2. Performance degradation
    3. Concept drift
    4. Adversarial inputs
    """
    # Your implementation here
    pass

def explainable_ai_for_stakeholders():
    """
    Create stakeholder-friendly explanations of tree decisions.
    
    Different audiences need different types of explanations:
    1. Technical teams: detailed tree structure
    2. Business users: high-level rules
    3. End users: simple explanations for individual predictions
    4. Regulators: audit trails and fairness metrics
    """
    # Your implementation here
    pass

def comprehensive_model_evaluation_framework():
    """
    Create a comprehensive evaluation framework covering:
    
    1. Statistical performance metrics
    2. Business impact metrics  
    3. Fairness and bias assessment
    4. Robustness testing
    5. Interpretability evaluation
    6. Computational efficiency metrics
    """
    # Your implementation here
    # This is your capstone analysis bringing everything together
    pass
```


---

## Submission Guidelines

**What to Submit:**
1. Complete Jupyter notebook with all implementations, analysis, and visualizations
2. Well-documented code demonstrating deep understanding
3. Comprehensive explanations of your findings and insights
4. Executive summary highlighting key learnings
5. Discussion of decision trees' place in the machine learning landscape

**Evaluation Criteria:**
- **Correctness (25%)**: Accurate manual calculations and implementations
- **Code Quality (20%)**: Clean, well-documented, efficient code
- **Analysis Depth (25%)**: Thorough exploration and meaningful insights
- **Understanding (20%)**: Clear explanations demonstrating comprehension
- **Innovation (10%)**: Creative extensions and novel applications

**Bonus Challenges:**
- Implement decision trees for multi-output problems
- Create an interactive decision tree visualization tool
- Analyze decision trees for time series data
- Implement online/incremental decision tree learning
- Compare decision trees with neural networks on the same problems

**Professional Development:**
This assignment prepares you for real-world machine learning by emphasizing:
- Implementation skills that deepen algorithmic understanding
- Proper experimental methodology and evaluation practices
- Communication skills for technical and non-technical audiences
- Critical thinking about model limitations and appropriate use cases

Remember: The goal extends beyond completing tasks to developing the deep, practical understanding that enables you to use decision trees effectively in your future work. Take time to experiment, ask questions, and connect concepts to build lasting expertise.

**Final Reflection Questions:**
- When would you choose decision trees over other algorithms?
- How do decision trees fit into modern machine learning pipelines?
- What are the most important lessons you've learned about interpretable machine learning?
- How has implementing algorithms from scratch changed your understanding?

Good luck with your comprehensive exploration of decision trees!
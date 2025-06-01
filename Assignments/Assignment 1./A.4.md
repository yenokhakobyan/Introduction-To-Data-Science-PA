# Perceptron Model: Comprehensive Learning Assignment

## Assignment Overview

Welcome to your comprehensive exploration of the Perceptron! This assignment will take you on a remarkable journey through one of the most foundational algorithms in machine learning and artificial intelligence. The perceptron, invented by Frank Rosenblatt in 1957, represents humanity's first serious attempt to create artificial neurons that could learn from experience, much like biological neurons in our brains.

What makes the perceptron so captivating from a learning perspective is its elegant simplicity combined with profound implications. At its core, the perceptron is just a mathematical function that takes inputs, multiplies them by weights, sums everything up, and makes a binary decision. Yet this simple mechanism can learn to classify data, recognize patterns, and serve as the building block for the most sophisticated neural networks powering today's artificial intelligence revolution.

Think of the perceptron as learning to draw a line that separates different types of data points, much like how you might draw a line on a map to separate different territories. But unlike a static line drawn once, the perceptron's line moves and adjusts based on every mistake it makes, gradually finding the best position to separate the data. This process of learning from errors is fundamental to how both artificial and biological neural systems improve their performance over time.

The beauty of studying the perceptron lies in its transparency. Unlike complex neural networks that can seem like mysterious black boxes, you can understand exactly how a perceptron makes every decision and how it learns from every example. You can visualize its decision boundary, trace through its learning process step by step, and gain intuitive understanding of concepts like weights, biases, learning rates, and convergence that form the foundation of all neural network architectures.

Throughout this assignment, you'll discover how the perceptron connects to fundamental concepts in linear algebra, optimization theory, and computational neuroscience. You'll see how its limitations led to important theoretical developments like the multi-layer perceptron and modern deep learning, while its strengths continue to make it valuable for certain types of problems today.

**Your Learning Journey:**
You'll begin by working through perceptron calculations by hand to build deep intuition about how weights update and decisions form. Then you'll implement the complete learning algorithm from scratch, understanding every component from initialization to convergence. Next, you'll explore the perceptron's capabilities and limitations through carefully designed experiments and real-world applications. Finally, you'll master advanced concepts like multi-class classification, ensemble methods, and the connections to modern neural networks.

**Part 4 is mainly advanced topics and is not required but highlt recommended.** 

**Required Libraries and Mathematical Foundation:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import Perceptron as SklearnPerceptron
import math
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Enhanced plotting setup for visualizing perceptron behavior
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# Mathematical constants and helper functions for clarity
EPSILON = 1e-10  # Small constant to prevent numerical issues
```

---

## Part 1: Manual Perceptron Calculations - Building Deep Intuition (5 Questions)

Understanding the perceptron begins with grasping how it processes information and learns from mistakes at the most fundamental level. These manual exercises will build the crucial mathematical intuition you need to understand why the perceptron works, when it succeeds, and why it sometimes fails. Working through calculations by hand reveals the elegant simplicity underlying this powerful learning algorithm.

### Question 1: Understanding the Perceptron Decision Function

The perceptron makes decisions by computing a weighted sum of its inputs and comparing this sum to a threshold. This process mirrors how biological neurons integrate signals from their inputs and fire when the total stimulation exceeds a certain level.

**Mathematical Foundation:**
The perceptron's decision function is: f(x) = sign(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
where w₁, w₂, ..., wₙ are weights, x₁, x₂, ..., xₙ are input features, and b is the bias term.

**Training Data:**
You have a small 2D dataset representing whether customers will purchase a product based on their age and income:

| Customer | Age (x₁) | Income (x₂) | Purchase (y) |
|----------|----------|-------------|--------------|
| A        | 25       | 30          | 0 (No)       |
| B        | 35       | 50          | 1 (Yes)      |
| C        | 45       | 70          | 1 (Yes)      |
| D        | 20       | 25          | 0 (No)       |
| E        | 40       | 60          | 1 (Yes)      |

**Initial Perceptron Parameters:**
Start with weights w₁ = 0.1, w₂ = 0.2, and bias b = -10.

**Your Tasks:**
Calculate the perceptron's output for each customer using the decision function. Show your complete calculations step by step, including the weighted sum and the final binary decision.

**Deep Analysis Questions:**
For each customer, explain what the weighted sum represents intuitively. How does the bias term affect the decision boundary? If the weighted sum is exactly zero, what challenge does this present for classification?

**Geometric Interpretation:**
The decision boundary of a perceptron is always a straight line (or hyperplane in higher dimensions). Using your calculated results, describe where this line would be positioned relative to the data points. Which customers are on which side of the decision boundary?

**Space for your calculations:**

```python
# You can use this space to verify your manual calculations
# But work through each step by hand first to build intuition!

```

### Question 2: The Perceptron Learning Rule in Action

Now you'll experience how the perceptron learns by adjusting its weights when it makes mistakes. This learning process is the heart of the perceptron algorithm and demonstrates how artificial neurons can improve their performance through experience.

**Learning Rule Foundation:**
When the perceptron makes an error, it updates its weights using:
- w_new = w_old + η × (y_true - y_predicted) × x
- b_new = b_old + η × (y_true - y_predicted)
where η (eta) is the learning rate.

**Your Task:**
Using the same dataset from Question 1, manually perform one complete epoch of perceptron learning with learning rate η = 0.5. For each customer:

1. Calculate the current prediction using your current weights
2. Compare with the true label
3. If there's an error, update the weights using the learning rule
4. Show the weight values after each update

**Critical Thinking Questions:**
Why does the learning rule add the input values when the prediction was too low (false negative) and subtract them when the prediction was too high (false positive)? How does this mathematical relationship ensure the perceptron moves toward better classification?

**Learning Rate Analysis:**
What would happen if you used a very large learning rate like η = 5.0? What about a very small learning rate like η = 0.01? Discuss the trade-offs between learning speed and stability.

**Convergence Intuition:**
After your manual epoch, predict whether the perceptron would eventually converge to perfect classification on this dataset. What mathematical property of the data determines whether convergence is possible?

### Question 3: Geometric Understanding of Weight Updates

The perceptron learning process can be understood geometrically as the gradual rotation and shifting of the decision boundary line. This geometric perspective provides crucial intuition about how the algorithm works.

**Geometric Setup:**
Consider a simplified 2D case where you're trying to separate two classes of points. Your current decision boundary is defined by the line: 0.3x₁ + 0.4x₂ - 2 = 0.

**Scenario Analysis:**
You encounter a misclassified point at (6, 2) that should be classified as positive (y = 1) but your perceptron currently predicts negative. With learning rate η = 0.2, calculate the new decision boundary after the weight update.

**Your Tasks:**
Calculate the new weights and bias after the learning update. Then determine the equation of the new decision boundary line. Describe how the line has moved geometrically - did it rotate, shift, or both?

**Visualization Challenge:**
Sketch both the original and updated decision boundaries on a coordinate plane. Mark the misclassified point and show how the boundary adjustment helps correct this particular error.

**Deeper Geometric Insight:**
The weight vector (w₁, w₂) is always perpendicular to the decision boundary. Explain why this is true and how weight updates cause the boundary to rotate toward better classification of the error point.

### Question 4: Understanding Linear Separability

Linear separability is the fundamental concept that determines whether a perceptron can successfully learn to classify a dataset. This question explores this crucial limitation through hands-on analysis.

**Concept Foundation:**
A dataset is linearly separable if there exists a straight line (or hyperplane) that can perfectly separate all points of one class from all points of the other class. The perceptron can only learn to perfectly classify linearly separable datasets.

**Dataset Analysis Tasks:**
Analyze these three small datasets to determine which are linearly separable:

**Dataset 1:**
Points: (1,1)→Class A, (2,2)→Class A, (3,1)→Class B, (4,2)→Class B

**Dataset 2:**  
Points: (1,1)→Class A, (3,3)→Class A, (2,1)→Class B, (2,3)→Class B

**Dataset 3:**
Points: (0,0)→Class A, (2,2)→Class A, (0,2)→Class B, (2,0)→Class B

**Your Analysis:**
For each dataset, attempt to draw a straight line that separates the classes. If successful, identify the dataset as linearly separable. If unsuccessful, explain why no such line exists.

**Mathematical Exploration:**
For the linearly separable datasets you identified, can you find the weights and bias for a perceptron that would classify them perfectly? For the non-separable datasets, what happens when you try to apply the perceptron learning algorithm?

**Real-World Implications:**
Many real-world problems are not linearly separable. Discuss what this means for the practical applicability of the basic perceptron. How might you modify or extend the perceptron to handle non-linearly separable data?

### Question 5: Multi-Class Classification with Multiple Perceptrons

While a single perceptron can only perform binary classification, multiple perceptrons can work together to solve multi-class problems. This question explores the strategies for extending perceptron learning to more complex classification scenarios.

**Problem Setup:**
You need to classify flowers into three species (Setosa, Versicolor, Virginica) based on two measurements: petal length and petal width.

**Training Data:**
| Flower | Petal Length | Petal Width | Species    |
|--------|--------------|-------------|------------|
| 1      | 1.4          | 0.2         | Setosa     |
| 2      | 4.7          | 1.4         | Versicolor |
| 3      | 5.4          | 2.3         | Virginica  |
| 4      | 1.5          | 0.1         | Setosa     |
| 5      | 4.5          | 1.5         | Versicolor |
| 6      | 5.6          | 1.8         | Virginica  |

**Multi-Class Strategy Analysis:**

**Approach 1: One-vs-Rest**
Design three binary perceptrons: one to distinguish Setosa from others, one for Versicolor vs others, and one for Virginica vs others. For each perceptron, show how you would encode the training labels and what decision each perceptron would learn.

**Approach 2: One-vs-One**
Design three binary perceptrons for pairwise classification: Setosa vs Versicolor, Setosa vs Virginica, and Versicolor vs Virginica. Explain how you would combine their predictions to make final classification decisions.

**Critical Analysis:**
Compare the two approaches. What happens if multiple perceptrons claim a test point belongs to their class? What if no perceptron claims it? How would you resolve these conflicts?

**Confidence and Uncertainty:**
Unlike probability-based classifiers, perceptrons make hard binary decisions. How might you extract confidence measures from perceptron outputs to help with multi-class decision making?

**Computational Considerations:**
Which approach would require more training time and why? Which would be faster for making predictions on new data? Consider both the number of perceptrons needed and the complexity of the decision-making process.

---

## Part 2: Implementing the Perceptron from Scratch (7-Step Implementation)

Now you'll build a complete perceptron implementation from the ground up. This hands-on construction will deepen your understanding of every algorithmic component and give you the foundation to understand more complex neural network architectures. Each step builds naturally on the previous ones, creating a comprehensive learning experience.

### Step 1: Basic Perceptron Structure and Initialization

Begin by creating the fundamental structure of your perceptron class. Understanding how to properly initialize a perceptron sets the foundation for successful learning and reveals important considerations about starting conditions.

```python
class Perceptron:
    """
    A complete implementation of the Perceptron learning algorithm.
    
    The perceptron is a linear binary classifier that learns by adjusting
    weights based on classification errors. This implementation provides
    full control over the learning process and detailed insight into
    how the algorithm converges to solutions.
    """
    
    def __init__(self, learning_rate=0.1, max_iterations=1000, random_state=None):
        """
        Initialize the perceptron with learning parameters.
        
        The initialization process is crucial for perceptron learning.
        Different initialization strategies can affect convergence speed
        and the final solution found by the algorithm.
        
        Parameters:
        learning_rate: float, controls how much weights change with each update
                      Larger values learn faster but may overshoot optimal solutions
        max_iterations: int, maximum number of training epochs to prevent
                       infinite loops on non-separable data
        random_state: int, seed for reproducible weight initialization
        """
        # Your implementation here
        # Consider these design decisions:
        # 1. How should weights be initialized? (zeros, small random values, etc.)
        # 2. Should the bias be treated separately or as an additional weight?
        # 3. What learning rate provides good balance of speed and stability?
        # 4. How will you track the learning process for analysis?
        pass
    
    def _initialize_weights(self, n_features):
        """
        Initialize weights and bias for the perceptron.
        
        Weight initialization can significantly impact learning behavior.
        Small random weights often work better than zeros because they
        break symmetry and provide more diverse gradient directions.
        
        Parameters:
        n_features: int, number of input features
        """
        # Your implementation here
        # Experiment with different initialization strategies:
        # 1. All zeros: simple but may learn slowly
        # 2. Small random values: often converges faster
        # 3. Xavier/Glorot initialization: scales with input size
        # 4. Consider the impact of bias initialization
        pass
    
    def _add_bias_term(self, X):
        """
        Add bias term to input data for mathematical convenience.
        
        Adding a bias column of ones allows us to treat the bias
        as just another weight, simplifying the mathematical
        operations during learning and prediction.
        
        Parameters:
        X: array-like, input features
        
        Returns:
        X_with_bias: array with bias column added
        """
        # Your implementation here
        # This helper function prepares data for weight-based calculations
        pass

# Test your initialization
print("Testing Perceptron Initialization...")
perceptron = Perceptron(learning_rate=0.1, random_state=42)
# Create simple test data to verify initialization works correctly
```

### Step 2: Forward Propagation and Decision Making

Implement the forward pass of the perceptron, where inputs are transformed into predictions. This step embodies the core computational process that transforms raw data into classification decisions.

```python
def predict_single(self, x):
    """
    Make a prediction for a single input sample.
    
    This method implements the core perceptron decision function:
    output = sign(weights · inputs + bias)
    
    Understanding this step is crucial because it shows how
    linear combinations of features create decision boundaries.
    
    Parameters:
    x: array-like, single input sample
    
    Returns:
    prediction: int, predicted class (0 or 1, or -1 and 1 depending on convention)
    """
    # Your implementation here
    # Steps:
    # 1. Calculate the weighted sum (dot product of weights and inputs)
    # 2. Add the bias term
    # 3. Apply the activation function (sign function or step function)
    # 4. Return the binary prediction
    pass

def predict(self, X):
    """
    Make predictions for multiple input samples.
    
    This method applies the single prediction function to
    entire datasets efficiently, enabling batch processing.
    
    Parameters:
    X: array-like, shape (n_samples, n_features), input data
    
    Returns:
    predictions: array, predicted classes for all samples
    """
    # Your implementation here
    # Consider vectorized operations for efficiency
    pass

def decision_function(self, X):
    """
    Calculate the decision function values without applying threshold.
    
    The decision function returns the raw weighted sum before
    applying the sign function. This is useful for understanding
    confidence and for visualizing decision boundaries.
    
    Parameters:
    X: array-like, input data
    
    Returns:
    decision_values: array, raw decision function outputs
    """
    # Your implementation here
    # This provides the "confidence" of predictions
    pass

# Test forward propagation
print("Testing Forward Propagation...")
# Create simple test cases to verify prediction logic
```

### Step 3: The Learning Algorithm Core

Implement the heart of perceptron learning: the weight update rule that allows the perceptron to learn from its mistakes. This step reveals how simple mathematical rules can lead to intelligent behavior.

```python
def _update_weights(self, x, y_true, y_pred):
    """
    Update weights based on a single training example.
    
    This implements the core perceptron learning rule:
    w_new = w_old + learning_rate * (y_true - y_pred) * x
    
    The beauty of this rule is its simplicity and intuitive nature:
    - When prediction is correct, no update occurs
    - When prediction is wrong, weights move toward the correct answer
    
    Parameters:
    x: array-like, input features for one sample
    y_true: int, true class label
    y_pred: int, predicted class label
    """
    # Your implementation here
    # Key considerations:
    # 1. Only update when there's an error (y_true != y_pred)
    # 2. The direction of update depends on the type of error
    # 3. Learning rate scales the magnitude of updates
    # 4. Track whether any update occurred (for convergence detection)
    pass

def fit_single_epoch(self, X, y):
    """
    Perform one complete epoch of learning over the training data.
    
    An epoch processes every training example once, updating
    weights when errors occur. Multiple epochs are typically
    needed for the perceptron to converge to a solution.
    
    Parameters:
    X: array-like, training features
    y: array-like, training labels
    
    Returns:
    n_errors: int, number of misclassified examples in this epoch
    """
    # Your implementation here
    # Algorithm:
    # 1. Initialize error count
    # 2. For each training example:
    #    a. Make prediction
    #    b. Compare with true label
    #    c. Update weights if error occurred
    #    d. Count errors for convergence tracking
    # 3. Return total errors for this epoch
    pass

# Test learning mechanics
print("Testing Learning Algorithm...")
# Create linearly separable data to test learning
```

### Step 4: Complete Training Loop with Convergence Detection

Implement the full training process that repeatedly applies the learning rule until convergence or until maximum iterations are reached. This step demonstrates how local weight updates lead to global learning.

```python
def fit(self, X, y, verbose=False):
    """
    Train the perceptron on the provided dataset.
    
    This method orchestrates the complete learning process:
    - Initialize weights appropriately
    - Repeatedly apply learning rule across epochs
    - Monitor convergence and stop when solution is found
    - Track learning progress for analysis
    
    Parameters:
    X: array-like, shape (n_samples, n_features), training data
    y: array-like, shape (n_samples,), training labels
    verbose: bool, whether to print learning progress
    
    Returns:
    self: returns the trained perceptron for method chaining
    """
    # Your implementation here
    # Complete training algorithm:
    # 1. Validate and prepare input data
    # 2. Initialize weights for the given feature space
    # 3. Repeat for each epoch:
    #    a. Process all training examples
    #    b. Count classification errors
    #    c. Check for convergence (zero errors)
    #    d. Store progress for analysis
    # 4. Handle convergence vs. maximum iterations reached
    pass

def _prepare_data(self, X, y):
    """
    Prepare and validate training data.
    
    Data preparation includes format validation, adding bias terms,
    and ensuring labels are in the correct format for learning.
    
    Parameters:
    X: raw input features
    y: raw target labels
    
    Returns:
    X_prepared: processed features ready for training
    y_prepared: processed labels ready for training
    """
    # Your implementation here
    # Consider:
    # 1. Convert to numpy arrays if needed
    # 2. Validate data shapes and types
    # 3. Handle different label encodings (0/1 vs -1/1)
    # 4. Add bias terms to features
    pass

def get_training_history(self):
    """
    Return detailed training history for analysis.
    
    Training history helps understand the learning process:
    - How quickly did errors decrease?
    - Did the algorithm converge?
    - Were there oscillations or steady progress?
    
    Returns:
    history: dict containing epoch-by-epoch training metrics
    """
    # Your implementation here
    # Track metrics like: errors per epoch, weight changes, etc.
    pass

# Test complete training
print("Testing Complete Training Process...")
# Use both separable and non-separable data to test robustness
```

### Step 5: Visualization and Analysis Tools

Implement comprehensive visualization tools to understand perceptron behavior. Visualization is crucial for building intuition about how perceptrons work and debugging when they don't.

```python
def plot_decision_boundary(self, X, y, title="Perceptron Decision Boundary"):
    """
    Visualize the perceptron's decision boundary for 2D data.
    
    Decision boundary visualization reveals how the perceptron
    divides the feature space and helps identify whether the
    classification problem is linearly separable.
    
    Parameters:
    X: array-like, 2D input features
    y: array-like, target labels
    title: str, plot title
    """
    # Your implementation here
    # Steps:
    # 1. Create a mesh grid covering the data space
    # 2. Calculate decision function values for all grid points
    # 3. Plot contour line where decision function equals zero
    # 4. Overlay training data points with class-based colors
    # 5. Highlight any misclassified points
    pass

def plot_learning_curve(self):
    """
    Plot the learning curve showing error reduction over epochs.
    
    Learning curves reveal the convergence behavior:
    - Smooth decrease indicates good learning
    - Oscillations suggest learning rate may be too high
    - Plateaus indicate convergence or non-separable data
    """
    # Your implementation here
    # Plot errors per epoch and highlight convergence point
    pass

def visualize_weight_evolution(self):
    """
    Visualize how weights change during training.
    
    Weight evolution shows the learning trajectory and
    helps understand how the decision boundary moves
    toward the optimal position.
    """
    # Your implementation here
    # Plot weight values over epochs, showing the path to convergence
    pass

def analyze_predictions(self, X, y):
    """
    Provide detailed analysis of perceptron predictions.
    
    Prediction analysis includes:
    - Classification accuracy and error breakdown
    - Confidence distribution (decision function values)
    - Identification of difficult examples
    
    Parameters:
    X: input features
    y: true labels
    
    Returns:
    analysis: dict containing comprehensive prediction analysis
    """
    # Your implementation here
    pass

# Test visualization tools
print("Testing Visualization Tools...")
# Create diverse datasets to test visualization capabilities
```

### Step 6: Multi-Class Extensions

Extend your perceptron to handle multi-class classification problems using strategies like one-vs-rest and one-vs-one. This demonstrates how simple binary classifiers can solve complex problems.

```python
class MultiClassPerceptron:
    """
    Multi-class extension of the perceptron using ensemble strategies.
    
    Multi-class perceptrons combine multiple binary perceptrons
    to solve problems with more than two classes. This approach
    shows how simple building blocks create sophisticated systems.
    """
    
    def __init__(self, strategy='one_vs_rest', learning_rate=0.1, 
                 max_iterations=1000, random_state=None):
        """
        Initialize multi-class perceptron.
        
        Parameters:
        strategy: str, multi-class strategy ('one_vs_rest' or 'one_vs_one')
        """
        # Your implementation here
        # Consider how many binary perceptrons you'll need for each strategy
        pass
    
    def fit(self, X, y, verbose=False):
        """
        Train multi-class perceptron using chosen strategy.
        
        Training process differs based on strategy:
        - One-vs-rest: train each perceptron to separate one class from all others
        - One-vs-one: train perceptrons for every pair of classes
        
        Parameters:
        X: training features
        y: training labels (can be any number of classes)
        """
        # Your implementation here
        # Algorithm depends on chosen strategy
        pass
    
    def _fit_one_vs_rest(self, X, y):
        """Implement one-vs-rest training strategy."""
        # Your implementation here
        # For each class, create binary problem: this class vs all others
        pass
    
    def _fit_one_vs_one(self, X, y):
        """Implement one-vs-one training strategy."""
        # Your implementation here
        # For each pair of classes, create binary classifier
        pass
    
    def predict(self, X):
        """
        Make multi-class predictions using trained ensemble.
        
        Prediction aggregation differs by strategy:
        - One-vs-rest: class with highest confidence wins
        - One-vs-one: majority vote among pairwise classifiers
        """
        # Your implementation here
        pass
    
    def predict_with_confidence(self, X):
        """
        Make predictions with confidence scores.
        
        Confidence helps identify uncertain predictions and
        reveals when the multi-class problem may be challenging.
        """
        # Your implementation here
        pass

# Test multi-class functionality
print("Testing Multi-Class Perceptron...")
# Create multi-class datasets to verify both strategies work
```

### Step 7: Advanced Features and Optimizations

Add sophisticated features that enhance perceptron performance and provide insights into learning dynamics. These extensions bridge the gap between basic perceptrons and modern neural networks.

```python
def add_regularization(self, regularization_type='l2', reg_strength=0.01):
    """
    Add regularization to prevent overfitting and improve generalization.
    
    Regularization modifies the learning process to prefer simpler
    models by penalizing large weights. This helps perceptrons
    generalize better to new data.
    
    Parameters:
    regularization_type: str, type of regularization ('l1', 'l2', 'elastic_net')
    reg_strength: float, strength of regularization penalty
    """
    # Your implementation here
    # Modify weight updates to include regularization terms
    pass

def implement_adaptive_learning_rate(self, strategy='decay'):
    """
    Implement adaptive learning rate strategies.
    
    Adaptive learning rates can improve convergence by starting
    with large steps and gradually reducing step size as the
    solution is approached.
    
    Parameters:
    strategy: str, adaptation strategy ('decay', 'bold_driver', 'adagrad')
    """
    # Your implementation here
    pass

def add_momentum(self, momentum_factor=0.9):
    """
    Add momentum to weight updates for faster convergence.
    
    Momentum helps the perceptron maintain direction across
    updates, leading to smoother convergence and ability to
    escape local oscillations.
    
    Parameters:
    momentum_factor: float, momentum coefficient (0 to 1)
    """
    # Your implementation here
    # Track previous weight changes and incorporate into current updates
    pass

def implement_early_stopping(self, validation_split=0.2, patience=10):
    """
    Implement early stopping to prevent overfitting.
    
    Early stopping monitors validation performance and stops
    training when performance stops improving, preventing
    the model from memorizing training data.
    
    Parameters:
    validation_split: float, fraction of data for validation
    patience: int, epochs to wait before stopping
    """
    # Your implementation here
    pass

def create_ensemble_perceptron(self, n_estimators=10, voting='hard'):
    """
    Create ensemble of perceptrons for improved performance.
    
    Ensemble methods combine multiple perceptrons trained on
    different subsets of data or with different initializations
    to create more robust predictions.
    
    Parameters:
    n_estimators: int, number of perceptrons in ensemble
    voting: str, ensemble combination strategy ('hard' or 'soft')
    """
    # Your implementation here
    pass

# Test advanced features
print("Testing Advanced Features...")
# Verify each enhancement improves performance on appropriate datasets
```

---

## Part 3: Real-World Applications with Perceptrons (2 Tasks)

Now you'll apply your perceptron implementations to meaningful real-world problems, experiencing how this fundamental algorithm performs in practical scenarios and understanding its place in the broader machine learning landscape.

### Task 1: Email Spam Classification - Text-Based Binary Classification

Email spam detection represents a classic application where perceptrons can demonstrate their effectiveness on high-dimensional, sparse data. This task will show you how to handle text data and deal with the challenges of real-world classification problems.

```python
# Create a realistic email spam dataset
def create_email_spam_dataset(n_samples=2000, n_features=1000, random_state=42):
    """
    Generate a realistic email spam classification dataset.
    
    This function simulates the characteristics of real email data:
    - High dimensionality (many possible words)
    - Sparsity (most emails use only a subset of vocabulary)
    - Class imbalance (more legitimate emails than spam)
    - Feature correlation (related words appear together)
    """
    np.random.seed(random_state)
    
    # Create base feature matrix with sparsity
    X = np.random.binomial(1, 0.05, (n_samples, n_features)).astype(float)
    
    # Define spam indicator words (higher probability in spam emails)
    spam_words = np.random.choice(n_features, size=50, replace=False)
    # Define legitimate indicator words
    legit_words = np.random.choice(n_features, size=80, replace=False)
    
    # Generate labels with realistic class distribution
    spam_rate = 0.3  # 30% spam emails
    y = np.random.binomial(1, spam_rate, n_samples)
    
    # Enhance feature values based on email type
    for i in range(n_samples):
        if y[i] == 1:  # Spam email
            # Increase probability of spam words
            X[i, spam_words] += np.random.binomial(1, 0.4, len(spam_words))
            # Add some noise and correlated features
            correlated_spam = np.random.choice(spam_words, size=10)
            X[i, correlated_spam] += np.random.binomial(1, 0.3, len(correlated_spam))
        else:  # Legitimate email
            # Increase probability of legitimate words
            X[i, legit_words] += np.random.binomial(1, 0.3, len(legit_words))
            # Reduce spam word presence
            X[i, spam_words] *= np.random.binomial(1, 0.1, len(spam_words))
    
    # Normalize features to represent word frequency
    X = np.clip(X, 0, 1)  # Binary presence/absence
    
    # Create feature names for interpretability
    feature_names = [f'word_{i}' for i in range(n_features)]
    spam_feature_names = [f'spam_word_{i}' for i in range(len(spam_words))]
    legit_feature_names = [f'legit_word_{i}' for i in range(len(legit_words))]
    
    return X, y, feature_names, spam_words, legit_words

# Load the email spam dataset
X_email, y_email, feature_names, spam_indicators, legit_indicators = create_email_spam_dataset(
    n_samples=2000, n_features=1000, random_state=42
)

print("Email Spam Dataset Information:")
print(f"Number of emails: {len(X_email)}")
print(f"Number of features (words): {X_email.shape[1]}")
print(f"Spam rate: {y_email.mean():.1%}")
print(f"Feature sparsity: {(X_email == 0).mean():.1%} of features are zero")
print(f"Average words per email: {X_email.sum(axis=1).mean():.1f}")

# Your comprehensive spam classification analysis:

# 1. Text Data Preprocessing and Feature Analysis
# Analyze the characteristics of text data that affect perceptron performance:
# - High dimensionality and sparsity challenges
# - Feature scaling considerations for text data
# - Identifying most discriminative features
# - Understanding the curse of dimensionality in text classification
# Start your text analysis here:


# 2. Perceptron Performance on High-Dimensional Data
# Apply your custom perceptron to email classification:
# - Compare different initialization strategies for high-dimensional data
# - Analyze convergence behavior on sparse, high-dimensional features
# - Experiment with different learning rates for text data
# - Study the effect of feature scaling on perceptron performance
# Implement high-dimensional analysis here:


# 3. Feature Selection and Dimensionality Reduction
# Explore techniques to improve perceptron performance on text:
# - Implement feature selection based on information gain or chi-square
# - Analyze which words are most important for spam detection
# - Study how feature reduction affects classification accuracy
# - Compare full feature set vs. selected features performance
# Implement feature selection analysis here:


# 4. Dealing with Class Imbalance
# Address the challenge of imbalanced spam/legitimate email distribution:
# - Analyze how class imbalance affects perceptron learning
# - Implement class weighting in your perceptron
# - Experiment with different evaluation metrics (precision, recall, F1)
# - Study the bias toward majority class in standard perceptron learning
# Implement imbalance handling here:


# 5. Comparison with Modern Text Classification
# Compare perceptron performance with more sophisticated methods:
# - Benchmark against sklearn's Perceptron on the same data
# - Compare with other linear methods (Logistic Regression, SVM)
# - Analyze computational efficiency for large-scale text classification
# - Discuss when perceptrons are still relevant for text problems
# Implement comparative analysis here:


# 6. Real-World Deployment Considerations
# Address practical considerations for spam filter deployment:
# - Online learning for adapting to new spam patterns
# - Handling concept drift as spam tactics evolve
# - Computational efficiency for real-time email filtering
# - Interpretability for understanding why emails are classified as spam
# Implement deployment analysis here:

```

### Task 2: Medical Diagnosis - Binary Classification with Interpretability Requirements

Medical diagnosis represents a domain where perceptrons can provide both accurate predictions and interpretable results. This task explores how the simplicity and transparency of perceptrons make them valuable in high-stakes applications where understanding the decision process is crucial.

```python
# Load and enhance a medical dataset for perceptron analysis
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
cancer_data = load_breast_cancer()
X_cancer_base, y_cancer = cancer_data.data, cancer_data.target

def enhance_medical_dataset(X_base, y_base, random_state=42):
    """
    Enhance the medical dataset to create realistic clinical scenarios.
    
    This function adds complexity that mirrors real medical diagnosis:
    - Missing values (common in clinical data)
    - Measurement noise (instrument variability)
    - Derived features (ratios and combinations used by clinicians)
    - Patient demographics that affect diagnosis
    """
    np.random.seed(random_state)
    n_samples = X_base.shape[0]
    
    # Add realistic measurement noise
    X_enhanced = X_base.copy()
    noise_scale = 0.05 * np.std(X_base, axis=0)
    noise = np.random.normal(0, noise_scale, X_base.shape)
    X_enhanced += noise
    
    # Create clinically relevant derived features
    # These represent the kind of feature engineering doctors do mentally
    
    # Ratios that might indicate malignancy
    area_perimeter_ratio = X_enhanced[:, 2] / X_enhanced[:, 3]  # area/perimeter
    texture_radius_ratio = X_enhanced[:, 1] / X_enhanced[:, 0]  # texture/radius
    
    # Interaction terms that capture combined effects
    size_texture_interaction = X_enhanced[:, 0] * X_enhanced[:, 1]  # radius * texture
    
    # Symmetry-related compound features
    symmetry_smoothness = X_enhanced[:, 8] * X_enhanced[:, 5]  # symmetry * smoothness
    
    # Add patient demographic features that might affect diagnosis
    # Age (older patients have different risk profiles)
    age = np.random.normal(58, 15, n_samples)
    age = np.clip(age, 25, 85)
    
    # Family history (binary: 0 = no family history, 1 = family history)
    family_history = np.random.binomial(1, 0.15, n_samples)
    
    # Previous benign findings (affects current diagnosis interpretation)
    previous_benign = np.random.binomial(1, 0.25, n_samples)
    
    # Combine all features
    derived_features = np.column_stack([
        area_perimeter_ratio, texture_radius_ratio, size_texture_interaction,
        symmetry_smoothness, age, family_history, previous_benign
    ])
    
    X_final = np.column_stack([X_enhanced, derived_features])
    
    # Create feature names for interpretability
    base_names = list(cancer_data.feature_names)
    derived_names = [
        'Area_Perimeter_Ratio', 'Texture_Radius_Ratio', 'Size_Texture_Interaction',
        'Symmetry_Smoothness', 'Age', 'Family_History', 'Previous_Benign'
    ]
    
    feature_names = base_names + derived_names
    
    # Introduce realistic missing values (5% missing rate)
    missing_mask = np.random.random(X_final.shape) < 0.05
    X_with_missing = X_final.copy()
    X_with_missing[missing_mask] = np.nan
    
    return X_with_missing, y_base, feature_names

# Create enhanced medical dataset
X_medical, y_medical, medical_feature_names = enhance_medical_dataset(
    X_cancer_base, y_cancer, random_state=42
)

print("Enhanced Medical Dataset Information:")
print(f"Number of patients: {X_medical.shape[0]}")
print(f"Number of features: {X_medical.shape[1]}")
print(f"Malignancy rate: {y_medical.mean():.1%}")
print(f"Missing value rate: {np.isnan(X_medical).mean():.1%}")

# Your comprehensive medical diagnosis analysis:

# 1. Medical Domain Analysis and Preprocessing
# Analyze data characteristics specific to medical diagnosis:
# - Handle missing values appropriately for clinical data
# - Understand the clinical significance of different features
# - Implement domain-appropriate feature scaling
# - Address the ethical implications of automated medical diagnosis
# Start medical domain analysis here:


# 2. Interpretable Model Development
# Develop perceptron models optimized for medical interpretability:
# - Train perceptrons that provide clear, interpretable decision rules
# - Analyze which features contribute most to malignancy prediction
# - Create visualizations that medical professionals can understand
# - Implement confidence measures for clinical decision support
# Implement interpretable model development here:


# 3. Clinical Performance Evaluation
# Evaluate perceptron performance using medical-relevant metrics:
# - Focus on sensitivity (recall) for detecting malignancies
# - Analyze specificity to minimize false positive anxiety
# - Implement cost-sensitive evaluation (false negatives are more costly)
# - Compare with clinical decision-making benchmarks
# Implement clinical evaluation here:


# 4. Feature Importance and Clinical Insights
# Extract actionable insights for medical professionals:
# - Identify which measurements are most predictive of malignancy
# - Analyze how derived features compare to raw measurements
# - Study the stability of feature importance across different patient groups
# - Validate findings against known medical literature
# Implement feature importance analysis here:


# 5. Robustness and Reliability Analysis
# Assess model robustness for clinical deployment:
# - Test sensitivity to measurement errors and noise
# - Analyze performance across different patient demographics
# - Implement uncertainty quantification for difficult cases
# - Study model behavior at decision boundaries
# Implement robustness analysis here:


# 6. Clinical Decision Support Integration
# Design perceptron integration into clinical workflow:
# - Create user-friendly interfaces for medical professionals
# - Implement alerts and recommendations based on perceptron output
# - Design explanations that support rather than replace clinical judgment
# - Address regulatory and ethical considerations for medical AI
# Implement clinical integration analysis here:

```

---

## Part 4: Advanced Perceptron Topics and Modern Connections (15 Tasks)

This section will elevate your understanding from basic perceptron usage to expert-level mastery of advanced concepts, theoretical foundations, and connections to modern machine learning. These tasks bridge the gap between classical perceptron theory and contemporary neural network architectures.

### Tasks 1-3: Theoretical Foundations and Mathematical Analysis

#### Task 1: Perceptron Convergence Theory and Proof Understanding

```python
def analyze_perceptron_convergence_theory():
    """
    Comprehensive analysis of the Perceptron Convergence Theorem.
    
    The Perceptron Convergence Theorem is one of the most important results
    in machine learning theory. Understanding it provides deep insight into
    when and why perceptrons work.
    """
    
    def empirical_convergence_validation(n_experiments=100):
        """
        Empirically validate the convergence theorem through simulation.
        
        The theorem states that if data is linearly separable, the perceptron
        algorithm will converge in a finite number of steps. We'll test this
        by creating separable data and measuring convergence behavior.
        """
        # Your implementation here
        # Steps:
        # 1. Generate linearly separable datasets with varying separability margins
        # 2. Track convergence time for each dataset
        # 3. Analyze relationship between margin size and convergence speed
        # 4. Validate that convergence always occurs for separable data
        # 5. Show non-convergence for non-separable data
        pass
    
    def margin_analysis():
        """
        Analyze the relationship between data margin and convergence speed.
        
        The margin is the distance from the closest point to the decision boundary.
        Larger margins generally lead to faster convergence.
        """
        # Your implementation here
        # Create datasets with different margins and measure convergence rates
        pass
    
    def convergence_bound_analysis():
        """
        Analyze the theoretical convergence bound: R²/γ² iterations.
        
        Where R is the maximum distance of any point from origin,
        and γ is the margin. This gives an upper bound on learning time.
        """
        # Your implementation here
        # Empirically verify this bound across different datasets
        pass
    
    def proof_visualization():
        """
        Create visualizations that illustrate key steps in the convergence proof.
        
        Show how the algorithm makes progress toward the optimal solution
        with each weight update, even when individual updates seem random.
        """
        # Your implementation here
        pass
    
    # Execute convergence analysis
    empirical_convergence_validation()
    margin_analysis()
    convergence_bound_analysis()
    proof_visualization()

# Run theoretical analysis
analyze_perceptron_convergence_theory()
```

#### Task 2: Geometric Interpretation and Decision Boundary Analysis

```python
def analyze_perceptron_geometry():
    """
    Deep geometric analysis of perceptron decision boundaries and learning.
    
    Understanding the geometry provides intuition about why perceptrons work
    and reveals connections to other linear classifiers.
    """
    
    def decision_boundary_evolution():
        """
        Visualize how decision boundaries evolve during learning.
        
        Show the geometric interpretation of weight updates as rotations
        and translations of the hyperplane separating classes.
        """
        # Your implementation here
        # Create animated visualizations showing boundary movement
        pass
    
    def weight_vector_analysis():
        """
        Analyze the geometric relationship between weights and decision boundary.
        
        The weight vector is always perpendicular to the decision boundary,
        and its magnitude affects the "confidence" of classifications.
        """
        # Your implementation here
        # Visualize weight vectors and their relationship to boundaries
        pass
    
    def margin_maximization_connection():
        """
        Explore connections between perceptron learning and margin maximization.
        
        While perceptrons don't explicitly maximize margins like SVMs,
        understanding this connection provides insight into solution quality.
        """
        # Your implementation here
        pass
    
    def high_dimensional_geometry():
        """
        Analyze perceptron behavior in high-dimensional spaces.
        
        In high dimensions, linear separability becomes more common,
        but visualization becomes challenging. Study this paradox.
        """
        # Your implementation here
        pass
    
    # Execute geometric analysis
    decision_boundary_evolution()
    weight_vector_analysis()
    margin_maximization_connection()
    high_dimensional_geometry()

# Run geometric analysis
analyze_perceptron_geometry()
```

#### Task 3: Error Analysis and Learning Dynamics

```python
def analyze_learning_dynamics():
    """
    Comprehensive analysis of perceptron learning dynamics and error behavior.
    
    Understanding how errors evolve during learning provides insight
    into algorithm behavior and guides hyperparameter selection.
    """
    
    def error_surface_analysis():
        """
        Analyze the error surface that perceptron learning navigates.
        
        Unlike smooth optimization problems, perceptron error surfaces
        are piecewise linear with discontinuous gradients.
        """
        # Your implementation here
        # Visualize error surfaces for simple 2D problems
        pass
    
    def learning_rate_sensitivity():
        """
        Comprehensive analysis of learning rate effects on convergence.
        
        Study how different learning rates affect:
        - Convergence speed
        - Solution stability
        - Oscillation behavior
        """
        # Your implementation here
        pass
    
    def initialization_impact():
        """
        Analyze how weight initialization affects learning trajectories.
        
        Different initializations can lead to different solutions
        and convergence paths, even for the same dataset.
        """
        # Your implementation here
        pass
    
    def noise_robustness_analysis():
        """
        Study perceptron robustness to different types of noise.
        
        Analyze behavior under:
        - Feature noise (measurement errors)
        - Label noise (annotation errors)
        - Adversarial perturbations
        """
        # Your implementation here
        pass
    
    # Execute learning dynamics analysis
    error_surface_analysis()
    learning_rate_sensitivity()
    initialization_impact()
    noise_robustness_analysis()

# Run learning dynamics analysis
analyze_learning_dynamics()
```

### Tasks 4-6: Advanced Perceptron Variants and Extensions

#### Task 4: Voted Perceptron and Averaged Perceptron

```python
def implement_advanced_perceptron_variants():
    """
    Implement and analyze advanced perceptron variants that improve generalization.
    
    These variants address limitations of the basic perceptron by considering
    multiple solutions or averaging across the learning trajectory.
    """
    
    class VotedPerceptron:
        """
        Voted Perceptron keeps track of all weight vectors found during learning.
        
        Instead of using only the final weights, it makes predictions by
        voting among all weight vectors encountered, weighted by their survival time.
        """
        
        def __init__(self, learning_rate=1.0, max_iterations=1000):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            """
            Train voted perceptron, storing all intermediate weight vectors.
            
            Track each weight vector and how long it survives (number of
            examples it classifies correctly before being updated).
            """
            # Your implementation here
            # Algorithm:
            # 1. Initialize weights
            # 2. For each training example:
            #    a. Make prediction with current weights
            #    b. If correct, increment survival count
            #    c. If incorrect, store current weights and survival count,
            #       then update weights
            # 3. Store all weight vectors with their survival counts
            pass
        
        def predict(self, X):
            """
            Make predictions using weighted voting among all weight vectors.
            
            Each stored weight vector votes on the prediction, weighted
            by how long it survived during training.
            """
            # Your implementation here
            pass
    
    class AveragedPerceptron:
        """
        Averaged Perceptron averages all weight vectors seen during training.
        
        This provides better generalization by reducing the variance
        of the final solution.
        """
        
        def __init__(self, learning_rate=1.0, max_iterations=1000):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            """
            Train averaged perceptron, maintaining running average of weights.
            
            Keep track of cumulative weight sum and average at the end.
            """
            # Your implementation here
            pass
        
        def predict(self, X):
            """Make predictions using averaged weights."""
            # Your implementation here
            pass
    
    def compare_perceptron_variants():
        """
        Comprehensive comparison of perceptron variants.
        
        Compare standard, voted, and averaged perceptrons on:
        - Generalization performance
        - Computational complexity
        - Memory requirements
        - Robustness to noise
        """
        # Your implementation here
        pass
    
    # Test advanced variants
    compare_perceptron_variants()

# Implement and test advanced variants
implement_advanced_perceptron_variants()
```

#### Task 5: Kernel Perceptron and Non-Linear Classification

```python
def implement_kernel_perceptron():
    """
    Implement kernel perceptron for non-linear classification.
    
    The kernel perceptron applies the kernel trick to enable
    non-linear decision boundaries while maintaining the
    simplicity of perceptron learning.
    """
    
    class KernelPerceptron:
        """
        Kernel Perceptron for non-linear classification.
        
        Uses kernel functions to implicitly map data to higher-dimensional
        spaces where linear separation becomes possible.
        """
        
        def __init__(self, kernel='rbf', gamma=1.0, degree=3, coef0=1.0, max_iterations=1000):
            """
            Initialize kernel perceptron.
            
            Parameters:
            kernel: str, kernel function ('linear', 'polynomial', 'rbf', 'sigmoid')
            gamma: float, kernel coefficient for RBF and polynomial kernels
            degree: int, degree for polynomial kernel
            coef0: float, independent term for polynomial and sigmoid kernels
            """
            # Your implementation here
            pass
        
        def _kernel_function(self, X1, X2):
            """
            Compute kernel function between two sets of points.
            
            The kernel function computes similarity between points
            in the transformed feature space without explicitly
            computing the transformation.
            """
            # Your implementation here
            # Implement different kernel functions:
            # - Linear: K(x,y) = x·y
            # - Polynomial: K(x,y) = (γx·y + r)^d
            # - RBF: K(x,y) = exp(-γ||x-y||²)
            # - Sigmoid: K(x,y) = tanh(γx·y + r)
            pass
        
        def fit(self, X, y):
            """
            Train kernel perceptron using dual formulation.
            
            In the dual formulation, we track coefficients α_i for each
            training example rather than explicit weight vectors.
            """
            # Your implementation here
            # Algorithm:
            # 1. Initialize α coefficients to zero
            # 2. For each training example x_i:
            #    a. Compute prediction using kernel evaluations
            #    b. If prediction is incorrect, increment α_i
            # 3. Store training data and coefficients for prediction
            pass
        
        def predict(self, X):
            """
            Make predictions using kernel evaluations.
            
            Prediction requires computing kernel function between
            test points and all training points with non-zero coefficients.
            """
            # Your implementation here
            pass
    
    def analyze_kernel_effects():
        """
        Analyze how different kernels affect decision boundaries.
        
        Study the impact of kernel choice and parameters on:
        - Decision boundary shape and complexity
        - Generalization performance
        - Computational requirements
        """
        # Your implementation here
        # Create non-linearly separable datasets and compare kernels
        pass
    
    def kernel_parameter_optimization():
        """
        Implement methods for optimizing kernel parameters.
        
        Kernel parameters significantly affect performance,
        requiring careful tuning for optimal results.
        """
        # Your implementation here
        pass
    
    # Test kernel perceptron
    analyze_kernel_effects()
    kernel_parameter_optimization()

# Implement and analyze kernel perceptron
implement_kernel_perceptron()
```

#### Task 6: Online Learning and Streaming Data

```python
def implement_online_perceptron_learning():
    """
    Implement perceptron variants optimized for online and streaming data.
    
    Online learning is crucial for applications where data arrives
    continuously and the model must adapt in real-time.
    """
    
    class OnlinePerceptron:
        """
        Online perceptron for streaming data with concept drift adaptation.
        """
        
        def __init__(self, learning_rate=0.1, decay_factor=0.99, adaptation_threshold=0.05):
            """
            Initialize online perceptron with drift detection.
            
            Parameters:
            decay_factor: float, exponential decay for old information
            adaptation_threshold: float, threshold for concept drift detection
            """
            # Your implementation here
            pass
        
        def partial_fit(self, x, y):
            """
            Update model with a single new example.
            
            Process one example at a time, updating weights immediately
            and detecting potential concept drift.
            """
            # Your implementation here
            # Steps:
            # 1. Make prediction with current weights
            # 2. Update weights if prediction is incorrect
            # 3. Update performance statistics
            # 4. Check for concept drift
            # 5. Adapt if drift is detected
            pass
        
        def detect_concept_drift(self):
            """
            Detect concept drift in streaming data.
            
            Use statistical methods to identify when the underlying
            data distribution has changed significantly.
            """
            # Your implementation here
            # Implement drift detection methods:
            # - Window-based error rate comparison
            # - Statistical significance tests
            # - Adaptive windowing (ADWIN)
            pass
        
        def adapt_to_drift(self):
            """
            Adapt model parameters when concept drift is detected.
            
            Strategies include:
            - Reset weights to handle dramatic shifts
            - Increase learning rate temporarily
            - Adjust window sizes for change detection
            """
            # Your implementation here
            pass
    
    class PassiveAggressivePerceptron:
        """
        Passive-Aggressive algorithm variant of perceptron.
        
        This variant is more aggressive in updating weights,
        making larger updates for larger margin violations.
        """
        
        def __init__(self, C=1.0, max_iterations=1000):
            """
            Initialize Passive-Aggressive perceptron.
            
            Parameters:
            C: float, aggressiveness parameter (higher = more aggressive updates)
            """
            # Your implementation here
            pass
        
        def fit(self, X, y):
            """
            Train using Passive-Aggressive updates.
            
            Update rule is more sophisticated than standard perceptron,
            taking into account the size of the margin violation.
            """
            # Your implementation here
            # PA update rule: w_new = w_old + τ * y * x
            # where τ = min(C, loss / ||x||²)
            pass
    
    def streaming_data_simulation():
        """
        Simulate realistic streaming data scenarios.
        
        Create datasets with:
        - Gradual concept drift
        - Sudden concept shift
        - Recurring concepts
        - Noise and outliers
        """
        # Your implementation here
        pass
    
    def online_performance_evaluation():
        """
        Evaluate online learning performance using appropriate metrics.
        
        Online evaluation requires different metrics than batch learning:
        - Cumulative regret
        - Sliding window accuracy
        - Adaptation speed after concept drift
        """
        # Your implementation here
        pass
    
    # Test online learning
    streaming_data_simulation()
    online_performance_evaluation()

# Implement and test online learning
implement_online_perceptron_learning()
```

### Tasks 7-9: Perceptron Ensembles and Meta-Learning

#### Task 7: Ensemble Methods with Perceptrons

```python
def implement_perceptron_ensembles():
    """
    Implement sophisticated ensemble methods using perceptron base learners.
    
    Ensemble methods can overcome individual perceptron limitations
    and improve robustness and accuracy.
    """
    
    class BaggingPerceptrons:
        """
        Bootstrap Aggregating with perceptron base learners.
        """
        
        def __init__(self, n_estimators=10, max_samples=1.0, max_features=1.0, 
                     bootstrap=True, random_state=None):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            """
            Train ensemble of perceptrons on bootstrap samples.
            
            Each perceptron sees a different view of the data,
            promoting diversity in the ensemble.
            """
            # Your implementation here
            pass
        
        def predict(self, X):
            """Make predictions using majority voting."""
            # Your implementation here
            pass
    
    class BoostingPerceptrons:
        """
        Adaptive Boosting with perceptron weak learners.
        """
        
        def __init__(self, n_estimators=10, learning_rate=1.0):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            """
            Train perceptrons sequentially with adaptive sample weighting.
            
            Each perceptron focuses on examples that previous
            perceptrons found difficult to classify.
            """
            # Your implementation here
            # Implement AdaBoost algorithm with perceptron base learners
            pass
        
        def predict(self, X):
            """Make predictions using weighted voting."""
            # Your implementation here
            pass
    
    class RandomSubspacePerceptrons:
        """
        Random Subspace method with perceptron base learners.
        
        Each perceptron trains on a random subset of features,
        promoting diversity in high-dimensional spaces.
        """
        
        def __init__(self, n_estimators=10, max_features=0.5):
            # Your implementation here
            pass
        
        def fit(self, X, y):
            """Train perceptrons on random feature subspaces."""
            # Your implementation here
            pass
    
    def analyze_ensemble_diversity():
        """
        Analyze diversity in perceptron ensembles.
        
        Study how different ensemble methods create diversity
        and how diversity relates to ensemble performance.
        """
        # Your implementation here
        pass
    
    # Test ensemble methods
    analyze_ensemble_diversity()

# Implement and test perceptron ensembles
implement_perceptron_ensembles()
```

#### Task 8: Meta-Learning and Adaptive Perceptrons

```python
def implement_meta_learning_perceptrons():
    """
    Implement meta-learning approaches for automatically configuring perceptrons.
    
    Meta-learning helps perceptrons adapt quickly to new tasks
    by learning from experience across multiple datasets.
    """
    
    class MetaPerceptron:
        """
        Meta-learning perceptron that adapts quickly to new tasks.
        """
        
        def __init__(self, meta_learning_rate=0.01, adaptation_steps=5):
            # Your implementation here
            pass
        
        def meta_train(self, task_datasets):
            """
            Train meta-learner across multiple tasks.
            
            Learn initialization and adaptation strategies that
            work well across diverse classification problems.
            """
            # Your implementation here
            # Implement Model-Agnostic Meta-Learning (MAML) for perceptrons
            pass
        
        def adapt_to_task(self, X_support, y_support):
            """
            Quickly adapt to a new task using few examples.
            
            Use meta-learned initialization and adaptation rules
            to achieve good performance with minimal training.
            """
            # Your implementation here
            pass
    
    class HyperparameterOptimizedPerceptron:
        """
        Perceptron with automated hyperparameter optimization.
        """
        
        def __init__(self, optimization_method='bayesian'):
            # Your implementation here
            pass
        
        def auto_configure(self, X, y, cv_folds=5):
            """
            Automatically find optimal hyperparameters for the dataset.
            
            Use techniques like Bayesian optimization or genetic algorithms
            to search the hyperparameter space efficiently.
            """
            # Your implementation here
            pass
    
    def few_shot_learning_analysis():
        """
        Analyze perceptron performance in few-shot learning scenarios.
        
        Study how quickly perceptrons can adapt to new tasks
        with very limited training data.
        """
        # Your implementation here
        pass
    
    # Test meta-learning approaches
    few_shot_learning_analysis()

# Implement and test meta-learning
implement_meta_learning_perceptrons()
```

#### Task 9: Neural Architecture Search for Perceptron Networks

```python
def implement_perceptron_architecture_search():
    """
    Implement architecture search for networks of perceptrons.
    
    Automatically discover optimal arrangements of perceptrons
    for solving complex problems.
    """
    
    class PerceptronNetworkSearchSpace:
        """
        Define search space for perceptron network architectures.
        """
        
        def __init__(self):
            # Your implementation
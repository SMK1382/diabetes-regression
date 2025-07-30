# Diabetes Regression Models

This repository contains two Python files implementing linear regression models to predict diabetes progression using the diabetes dataset from scikit-learn. One implementation is built from scratch using gradient descent, while the other uses scikit-learn's `LinearRegression` model.

## Files

1. **diabetes_regression_custom.py**  
   - Implements a linear regression model from scratch without using scikit-learn's `LinearRegression` class.  
   - Uses the **Gradient Descent** algorithm to optimize model parameters.  
   - Loads the diabetes dataset from scikit-learn for training and evaluation.  
   - Includes custom functions for model training, prediction, and evaluation using metrics like Mean Squared Error.  
   - Dependencies: `numpy`, `pandas`, `sklearn.datasets`, `sklearn.model_selection`, `sklearn.metrics`, `copy`.

2. **diabetes_regression_sklearn.py**  
   - Implements a linear regression model using scikit-learn's `LinearRegression` class.  
   - Loads the diabetes dataset from scikit-learn for training and evaluation.  
   - Optionally includes a `DecisionTreeClassifier` for additional experimentation (not used in the core regression task).  
   - Evaluates model performance using Mean Squared Error.  
   - Dependencies: `numpy`, `pandas`, `sklearn.datasets`, `sklearn.model_selection`, `sklearn.metrics`, `sklearn.linear_model`, `sklearn.tree`.

## Dataset
Both scripts use the **diabetes dataset** from scikit-learn, which includes 442 samples with 10 features (e.g., age, BMI, blood pressure) and a target variable representing disease progression after one year.

## Requirements
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
- Install dependencies using:
  ```
  pip install numpy pandas scikit-learn
  ```

## Usage
1. Run each script independently to train the respective model and view results:
   ```
   jupyter-lab diabetes_regression_custom.ipynb
   jupyter-lab diabetes_regression_sklearn.ipynb
   ```
2. Each script outputs model performance metrics (e.g., Mean Squared Error) and may include additional analysis or visualizations.

## Notes
- The custom implementation (`diabetes_regression_custom.py`) is designed for educational purposes, showcasing how linear regression can be implemented using gradient descent.  
- The scikit-learn implementation (`diabetes_regression_sklearn.py`) leverages optimized functions for practical use and includes an optional `DecisionTreeClassifier` for potential classification tasks.  
- Compare the performance of both models to understand the trade-offs between a custom implementation and scikit-learn's optimized version.

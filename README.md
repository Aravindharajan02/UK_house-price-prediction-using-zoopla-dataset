# UK_house-price-prediction-using-zoopla-dataset

## Overview
The project employs machine learning models to forecast property prices from Zoopla UK property listings data. It applies and compares several regression algorithms to determine the best performing model for real estate price prediction.

## Features
- Preprocessing and cleaning of Zoopla property listings data
- Feature engineering with categorical encoding
- Application of 7 various regression models:
  - Linear Regression
  - Random Forest
- Additional Trees
  - Artificial Neural Network (MLP)
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting
  - CatBoost

## Dataset
The dataset employed is the "Zoopla properties listing information.csv" dataset, which includes UK property listings with the features:
- Property size (sq ft)
- Number of bedrooms
- Number of bathrooms
- Number of receptions
- Tenure (Freehold/Leasehold)
- Price per size
- Address (postal code)
- Price (target variable)
- 
![zoopla hist](https://github.com/user-attachments/assets/72e8e68e-605b-4f89-a6bb-ee3be0225dc9)

![pairplot](https://github.com/user-attachments/assets/0156134d-a49b-4517-b182-d88ecbd5b193)

![heatmap_zoopla](https://github.com/user-attachments/assets/7c66449f-aa03-47a9-b02d-c806e5f35d06)

## Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations and array handling
- **scikit-learn**:
  - Machine learning models (LinearRegression, RandomForestRegressor, etc.)
  - Preprocessing tools (StandardScaler)
  - Model evaluation metrics (MAE, MSE, R²)
  - Train-test splitting functionality
- **catboost**: CatBoost gradient boosting algorithm implementation
- **seaborn & matplotlib**: Data visualization and plotting
- **warnings**: Suppression of redundant warnings

## Methodology
1. **Data Preprocessing**:
- Text extraction using regex for property size, price, and postal codes
   - Handling of categorical variables
   - Removal of missing values
   - Feature scaling with StandardScaler

2. **Feature Engineering**:
   - One-hot encoding for categorical features (tenure and address)
   - Extraction of postal codes from address information
- Price and property size text cleaning and conversion

3. **Model Training and Evaluation**:
   - 80/20 train-test split
   - Feature value standardization
   - Multiple regression model training
   - Evaluation using standard regression metrics

## Models Implemented
1. **Linear Regression**: Simple baseline model that defines linear associations between features and target
2. **Random Forest Regressor**: Ensemble approach using several decision trees to avoid overfitting and enhance prediction efficiency
3. **Extra Trees Regressor**: A variant of Random Forest with more randomization for enhanced generalization
4. **Artificial Neural Network (MLP)**: Multi-layer perceptron with two hidden layers containing 64 neurons each
5. **K-Nearest Neighbors**: Non-parametric model that predicts based on the k most similar attributes
6. **Gradient Boosting Regressor**: Method of sequential ensemble to construct trees for the purpose of eliminating mistakes made by the previous ones
7. **CatBoost Regressor**: Implementation of gradient boosting with automatic differentiation of categorical features

## Key Strengths
- **Comprehensive Model Comparison**: Compares several algorithms to determine the best method
- **Feature Engineering**: Deliberate preprocessing of text and categorical variables for better model performance
- **Standardized Evaluation**: Applies a variety of metrics (MAE, MSE, RMSE, R²) for a complete assessment of performance
- **Production-Ready**: Exposes complete ML pipeline from data cleaning to model checking
- **Ensemble Methods**: Employs sophisticated tree-based ensemble methods that generally work well on real estate datasets
- **Categorical Handling**: Extra care for categorical variables through one-hot encoding and custom algorithms

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- catboost
- seaborn
- matplotlib

## Installation
```bash
pip install pandas numpy scikit-learn catboost seaborn matplotlib
```

## Usage
1. Put the "Zoopla properties listing information.csv" file in the correct directory
2. Execute the notebook to:
   - Load and clean data
   - Engineer features and one-hot encode categorical variables
   - Train and compare several regression models
   - Compare model performances based on MAE, MSE, RMSE, and R²

## Model Evaluation
The notebook contrasts the following for every model:
- Mean Absolute Error (MAE): Average of the absolute difference between forecasted and actual prices
- Mean Squared Error (MSE): Average of squared differences between forecasted and actual prices
- Root Mean Squared Error (RMSE): Square root of MSE, gives error in the same unit as target
- R² Score: Ratio of variance in the dependent variable predicted by the model

## Future Improvements
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Feature importance analysis to determine most significant property characteristics
- Cross-validation implementation for better model assessment
- More sophisticated feature engineering (polynomial features, interaction terms)
- Ensemble methods to average model predictions
- Geographic visualization of price forecasts
- Time series analysis of price trends


## Author
Aravindharajan S S 
linkedin--- https://www.linkedin.com/in/aravindharajan-s-s/

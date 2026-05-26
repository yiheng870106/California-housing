# California Housing Regression Analysis

This project performs a regression analysis on the California Housing dataset using Python and scikit-learn. The goal is to predict median house values for California districts based on demographic, household, and geographic features.

The project compares three regression models:

- Linear Regression
- Polynomial Regression
- Ridge Regression with cross-validation

The final results show that Polynomial Regression performs best among the tested models.

## Dataset

This project uses the California Housing dataset from scikit-learn.

Dataset documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

The dataset contains 20,640 samples and 8 numeric predictive features.

### Features

| Feature | Description |
|---|---|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

### Target

The target variable is the median house value for California districts.

The target is expressed in units of $100,000. For example, a target value of 2.0 represents approximately $200,000.

## Project Structure

    California-housing/
    |
    |-- notebooks/
    |   |-- Setup.ipynb
    |   |-- Regression Analysis on California Housing Data.ipynb
    |
    |-- src/
    |   |-- __init__.py
    |   |-- tool.py
    |
    |-- results/
    |   |-- results.csv
    |
    |-- figures/
    |   |-- True vs. Predicted Linear regression.png
    |   |-- True vs. Predicted Polynomial regression.png
    |   |-- True vs. Predicted Ridge regression.png
    |
    |-- .gitignore
    |-- README.md

## Methods

### 1. Data Loading

The dataset is loaded using scikit-learn:

    from sklearn.datasets import fetch_california_housing

    cal = fetch_california_housing(as_frame=True)
    X, y = cal.data, cal.target

### 2. Train/Test Split

The dataset is split into training and testing sets:

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

The test set contains 20% of the data.

### 3. Models

This project evaluates three regression models.

#### Linear Regression

Linear Regression is used as the baseline model.

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

#### Polynomial Regression

Polynomial Regression is used to capture nonlinear relationships between features.

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly = PolynomialFeatures(degree=2, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

#### Ridge Regression

Ridge Regression is used to reduce overfitting through L2 regularization.

The best alpha is selected using cross-validation.

    from sklearn.linear_model import Ridge, RidgeCV

    alphas = [0.01, 0.1, 1, 10, 100, 1000]

    model_ridge_cv = RidgeCV(alphas=alphas, scoring="r2", cv=10)
    model_ridge_cv.fit(X_train, y_train)

    best_alpha = model_ridge_cv.alpha_

    model_ridge = Ridge(alpha=best_alpha)
    model_ridge.fit(X_train, y_train)

In this project, the best alpha was:

    10.0

## Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Meaning |
|---|---|
| R² | Proportion of variance explained by the model |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |

RMSE is especially useful because it is in the same unit as the target variable.

Since the California Housing target is measured in units of $100,000, an RMSE of 0.68 means the typical prediction error is about $68,000.

## Results

| Model | Data | R² | MAE | MSE | RMSE |
|---|---|---:|---:|---:|---:|
| Linear Regression | Train | 0.6126 | 0.5286 | 0.5179 | 0.7197 |
| Linear Regression | Test | 0.5758 | 0.5332 | 0.5559 | 0.7456 |
| Polynomial Regression | Train | 0.6853 | 0.4608 | 0.4207 | 0.6486 |
| Polynomial Regression | Test | 0.6457 | 0.4670 | 0.4643 | 0.6814 |
| Ridge Regression | Train | 0.6125 | 0.5287 | 0.5179 | 0.7197 |
| Ridge Regression | Test | 0.5764 | 0.5332 | 0.5550 | 0.7450 |

## Key Findings

Polynomial Regression achieved the best test performance.

Compared with Linear Regression, Polynomial Regression improved the test R² from 0.5758 to 0.6457 and reduced RMSE from 0.7456 to 0.6814.

This suggests that the relationship between the housing features and median house values is not purely linear.

Ridge Regression performed similarly to Linear Regression. This suggests that regularization alone did not significantly improve performance when using only the original features.

## Figures

The project generates true vs. predicted plots for each model.

These figures are saved in the `figures/` directory.

The red dashed line in each plot represents perfect prediction. Points closer to this line indicate better model performance.

## How to Run

This project was developed in Google Colab.

### Step 1: Clone or open the project

Clone the repository:

    git clone https://github.com/yiheng870106/California-housing.git

Move into the project folder:

    cd California-housing

### Step 2: Open the notebooks

Open the notebooks in Google Colab:

1. `notebooks/Setup.ipynb`
2. `notebooks/Regression Analysis on California Housing Data.ipynb`

### Step 3: Run setup

Run `Setup.ipynb` first. This notebook creates the project folders and helper files.

### Step 4: Run the regression analysis

Run `Regression Analysis on California Housing Data.ipynb`.

This notebook will:

- Load the California Housing dataset
- Split the data into train and test sets
- Train regression models
- Evaluate model performance
- Save results to `results/results.csv`
- Save figures to the `figures/` folder

## Requirements

The project uses the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn

If running locally, install the required libraries with:

    pip install numpy pandas matplotlib scikit-learn

Google Colab already includes most of these packages.

## Possible Improvements

Future improvements could include:

- Add exploratory data analysis
- Add correlation heatmap
- Add residual plots
- Add feature scaling with StandardScaler
- Use Pipeline for preprocessing and modeling
- Add cross-validation results for all models
- Compare more models, such as Random Forest and Gradient Boosting
- Add hyperparameter tuning with GridSearchCV
- Save trained models using joblib
- Add a requirements.txt file
- Make file paths more portable instead of relying on Google Drive paths
- Clear notebook outputs before committing to improve GitHub readability

## Conclusion

This project demonstrates how regression models can be used to predict California housing prices.

Among the tested models, Polynomial Regression achieved the best performance, indicating that nonlinear feature relationships are important for this dataset.

## Author

Yi-Heng Tsai

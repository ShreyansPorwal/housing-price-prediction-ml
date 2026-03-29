# Housing Price Prediction (Random Forest)

## Overview

This project builds a machine learning pipeline to predict California housing prices using structured census and geographic data.

The focus of this project was on feature engineering, model selection, and improving predictive performance through ensemble methods.

---

## My Contributions

* Conducted **exploratory data analysis (EDA)** to understand feature relationships and distributions
* Designed and implemented a **feature engineering pipeline**, including:

  * `rooms_per_household`
  * `bedrooms_per_room`
  * `population_per_household`
* Performed **feature selection** using correlation analysis and domain insights
* Built and tuned a **Random Forest regression model**
* Compared multiple models (Linear Regression, Ridge, Gradient Boosting, Random Forest)
* Evaluated performance using RMSE, MAE, and R²
* Contributed to project documentation and GitHub organization

---

## Dataset

* California housing dataset
* Features include:

  * Median income
  * Housing age
  * Geographic location
  * Engineered ratio-based features

---

## Methodology

### 1. Exploratory Data Analysis

* Correlation heatmaps
* Distribution analysis
* Identification of key predictors (median income, location)

### 2. Feature Engineering

* Created ratio-based features to improve signal:

  * rooms per household
  * bedrooms per room
  * population per household

### 3. Modeling

* Models evaluated:

  * Linear Regression
  * Ridge Regression
  * Random Forest
  * Gradient Boosting

### Final Model:

* **Random Forest**
* Achieved improved performance through feature selection and tuning

---

## Results

* Improved RMSE compared to baseline models
* Strong predictive performance with ensemble methods
* Feature engineering significantly improved model accuracy

---

## Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib / Seaborn

---

## Key Insights

* Median income is the strongest predictor of housing prices
* Engineered ratio features outperform raw features
* Tree-based models handle non-linearity effectively

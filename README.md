# Hotel Value Prediction — Machine Learning Regression Project

## Task
The objective of this project is to **predict the market value of hotels** based on various physical, locational, and structural characteristics of each property.  
This is a **supervised regression problem**, where the model learns from historical hotel data to predict property values for unseen hotels.  
The goal is to minimize the **Root Mean Squared Error (RMSE)** for the most accurate predictions.

---

## Dataset Description
The dataset contains comprehensive information about hotels, including:
- **Location attributes** (city, district, proximity, etc.)
- **Construction details** (year built, area, renovation details)
- **Facilities** (parking, lounge, pool, etc.)
- **Amenities and structural features**

The target variable is `HotelValue`, representing the **market value** of each hotel property.  
Predictions can assist **investors, developers, and hospitality analysts** in making data-driven investment decisions.

---

## EDA and Preprocessing

### 1. Importing Libraries & Loading Data
- Libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- Dataset loaded using `pd.read_csv()`

### 2. Data Overview
- Dataset contained **1200 rows and 81 columns**
- Checked for data types, missing values, and overall structure

### 3. Handling Duplicate Values
- Used `drop_duplicates()` — no duplicates found.

### 4. Handling Missing Values
- Columns with **<6% missing** → rows dropped.
- Columns with **>45% missing** → columns dropped.
- Moderate missing values (e.g., `RoadAccessLength`) handled using **group-wise median imputation** based on `PropertyClass` and `District`.
- Final dataset had **no missing values**.

### 5. Outlier Detection and Handling
- Outliers detected using **IQR method**.
- Values capped instead of removed.
- Rare categorical values (<1% frequency) identified for handling.

### 6. Scaling & Standardization
- Applied:
  - **Min-Max Scaling** → compressed values to [0, 1]
  - **Standardization (Z-score)** → centered data around mean 0, std 1
- Used `MinMaxScaler()` and `StandardScaler()` from scikit-learn.
- Visualized with KDE and boxplots.

### 7. Saving Preprocessed Data
- Final preprocessed dataset saved as `hotel_data_preprocessed.csv`.

### 8. Exploratory Data Analysis (EDA)
- **Target Variable (HotelValue)** was right-skewed → log-transformed.
- **Correlation Analysis** identified top 20 features influencing hotel value.
- **Linearity Assessment** showed mostly linear relationships.
- **Multicollinearity Check** → managed via PCA or feature elimination.
- **Outlier and Categorical Analysis** showed valid distribution tails and strong feature influence patterns.

**Summary:**
- Log transformation improved normality.
- PCA, correlation, and feature scaling improved model readiness.

---

## Models Used for Training

### 1. Decision Tree Regressor
- **Best Params:** `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=4`, `max_features=0.7`
- **Validation RMSE:** 30,404.07  
- **Kaggle Score:** **47,144.334**

### 2. Random Forest Regressor
- **Best Params:** `n_estimators=200`, `max_features='log2'`
- **Validation RMSE:** 24,787.88  
- **Kaggle Score:** **45,739.987**

### 3. Gradient Boosting Regressor
- **Best Params:** `max_depth=5`, `n_estimators=200`, `learning_rate=0.05`
- **Validation RMSE:** 20,716.46  
- **Kaggle Score:** **40,908.516**

### 4. XGBoost Regressor
- **Best Params:** `max_depth=4`, `n_estimators=700`, `learning_rate=0.03`
- **Validation RMSE:** 21,668.28  
- **Kaggle Score:** **41,597.766**

### 5. AdaBoost Regressor
- **Best Params:** `n_estimators=300`, `learning_rate=0.2`
- **Validation RMSE:** 25,289.24  
- **Kaggle Score:** **46,641.093**

### 6. Ridge Regression
- **R²:** 0.8879  
- **Validation RMSE:** 28,019.59  
- **Kaggle Score:** **33,895.045**

### 7. Lasso Regression
- **R²:** 0.8523  
- **Validation RMSE:** 32,161.34  
- **Kaggle Score:** **30,974.838**

### 8. ElasticNet Regression
- Applied PCA (95% variance retained) and log-transform.
- **Kaggle Score:** **19,999.523** (Best Performing Model)

### 9. K-Nearest Neighbors (KNN)
- **Best Params:** `n_neighbors=8`, `p=1`, `weights='distance'`
- **Validation RMSE:** 0.0000 (overfit)
- **Kaggle Score:** **45,263.419**

### 10. Bayesian Ridge Regression
- **Best Params:** `alpha_1=1e-08`, `alpha_2=1e-05`, `lambda_1=1e-05`, `lambda_2=1e-08`
- **R²:** 0.9090  
- **Validation RMSE:** 26,924.55  
- **Kaggle Score:** **29,325.086**

### 11. Maximum Likelihood Estimation (MLE)
- **R²:** 0.704  
- **MSE:** 1.32 × 10⁹  
- **Kaggle Score:** **455,146.781**

### 12. ElasticNet + LightGBM Blended Model
- Combines ElasticNet (linear) + LightGBM (non-linear)
- **MSE:** 7.81 × 10⁸  
- **R²:** 0.835  
- **Kaggle Score:** **24,769.301**

---

##  Final Rankings

| Rank | Model | Kaggle RMSE | Remarks |
|------|--------|--------------|----------|
| 1 | **ElasticNet Regression** | **19,999.523** | Best performer — balanced bias & variance; effective feature selection |
| 2 | **ElasticNet + LightGBM Blend** | 24,769.301 | Hybrid linear + non-linear learning; robust generalization |
| 3 | **Bayesian Ridge Regression** | 29,325.086 | Probabilistic regularization; handles uncertainty well |
| 4 | **Lasso Regression** | 30,974.838 | Sparse, interpretable model but mild underfitting |
| 5 | **Ridge Regression** | 33,895.045 | Stable but retains redundant predictors |
| 6 | **Gradient Boosting** | 40,908.516 | Captures non-linearities; slight overfitting |
| 7 | **XGBoost** | 41,597.766 | Efficient boosting but needs more tuning |
| 8 | **KNN Regression** | 45,263.419 | Overfit; poor generalization on high dimensions |
| 9 | **Random Forest** | 45,739.987 | Stable but lacks fine detail |
| 10 | **AdaBoost** | 46,641.093 | Weak learners limited performance |
| 11 | **Decision Tree** | 47,144.334 | Overfit, high variance |
| 12 | **Maximum Likelihood Estimation (MLE)** | 455,146.781 | Baseline; poor with non-linear & skewed data |

---

## Overall Insights

- **ElasticNet Outperformed All Models**
  - Dual regularization (L1 + L2) handled multicollinearity and noise effectively.
  - Provided the best generalization and lowest Kaggle RMSE.

- **Regularized Linear Models Were Most Reliable**
  - ElasticNet, Bayesian Ridge, Lasso, and Ridge performed consistently better than unregularized or distance-based methods.

- **Boosting Models Showed Strong Potential**
  - Gradient Boosting, XGBoost, and LightGBM performed well but needed deeper tuning for optimal results.

- **Simple Models Performed Poorly**
  - Decision Tree, KNN, and MLE models suffered from overfitting and weak generalization.

- **Impact of PCA & Feature Engineering**
  - PCA with `n_components=0.95` gave the best balance between dimensionality and performance.
  - Feature engineering (Age, YearsSinceRemod, and interaction terms) significantly boosted ElasticNet’s performance.

---

## Conclusion

The **ElasticNet Regression model** proved to be the most robust and accurate solution for hotel value prediction, achieving the **lowest Kaggle RMSE of 19,999.523**.  
Its hybrid L1-L2 penalty structure allowed it to generalize effectively while retaining interpretability — making it the **recommended model** for deployment in real-world hotel valuation tasks.

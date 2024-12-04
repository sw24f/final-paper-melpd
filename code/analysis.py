'''
import pandas as pd

nhl_edge = pd.read_csv('data\cleaned_nhl_edge.csv')
nst = pd.read_csv('data\cleaned_nst.csv')
nhl = pd.merge(nhl_edge, nst, on='Player', how='inner')  # 'inner' merge keeps only players present in both datasets
nhl = nhl[nhl['GP_x'] > 25]
# Identify collinearity
#print(nhl_edge.describe())
# Select only numerical columns
nhl_edge_numeric = nhl_edge.select_dtypes(include=['number'])
correlation_matrix = nhl_edge_numeric.corr()
#print(correlation_matrix)

import matplotlib.pyplot as plt
"""
# Scatter plot of shots vs goals
plt.scatter(nhl_edge['+/-'], nhl_edge['G'])
plt.xlabel('Plus/Minus')
plt.ylabel('Goals')
plt.title('Plus/Minus vs Goals')
plt.show()
"""

plus_minus_corr = nhl.select_dtypes(include=['number']).corr()['+/-']
print(plus_minus_corr)

print(nhl.columns)
'''
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load datasets
nhl_edge = pd.read_csv('data/cleaned_nhl_edge.csv')
nst = pd.read_csv('data/cleaned_nst.csv')
nst_indiv = pd.read_csv('data/cleaned_nst_indiv.csv')
nst = nst.drop_duplicates(subset='Player', keep='first')
nst_indiv = nst_indiv.drop_duplicates(subset='Player', keep='first')

# Merge datasets
nhl = pd.merge(nhl_edge, nst, on='Player', how='inner')  # Merge nhl_edge with nst
nhl = pd.merge(nhl, nst_indiv, on='Player', how='inner')  # Merge with nst_indiv

nhl['Team'] = nhl['Team_x'].combine_first(nhl['Team_y'])
nhl['Position'] = nhl['Position_x'].combine_first(nhl['Position_y'])
nhl['GP'] = nhl['GP_x'].combine_first(nhl['GP_y'])
nhl['TOI'] = nhl['TOI_x'].combine_first(nhl['TOI_y'])

# Clean up unnecessary columns
nhl.drop(columns=['Team_x', 'Team_y', 'Position_x', 'Position_y', 'GP_x', 'GP_y', 'TOI_x', 'TOI_y'], inplace=True)

# Filter for players with more than 25 games played
nhl = nhl[nhl['GP'] > 25]

print(nhl.columns)
# Select predictors and target
X = nhl[['GF%', 'PDO', 'SF%', 'Takeaways', 'SCF%']]
y = nhl['+/-']  # Target variable
'''
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split your data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to combine polynomial feature generation, scaling, and ridge regression
degree = 2  # Adjust the degree based on your observations
ridge_pipeline = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1109.752))  # Use your optimal lambda value
])

# Fit the pipeline to the training data
ridge_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = ridge_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

import matplotlib.pyplot as plt

# Plot predicted vs actual
plt.scatter(y_test, y_pred)
plt.axline((0, 0), slope=1, color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.legend()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()
'''

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Numerical analysis for multicollinearity
nhl_edge_numeric = nhl_edge.select_dtypes(include=['number'])  # Select only numerical columns

# Calculate correlation with plus/minus
plus_minus_corr = nhl.select_dtypes(include=['number']).corr()['+/-']
print("Correlation with +/-:")
print(plus_minus_corr)

# Variance Inflation Factor (VIF) for multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Find the best lambda (alpha) using RidgeCV
alphas = np.logspace(-6, 6, 200)  # Range of alphas to test
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)  # 5-fold CV
ridge_cv.fit(X_train, y_train)

best_alpha = ridge_cv.alpha_
print(f"\nOptimal Lambda (alpha): {best_alpha}")

# Fit ridge regression with the best alpha
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train, y_train)
cv_scores = cross_val_score(ridge, X_train, y_train, cv=5)
print("Cross-Validation Scores (Offensive/Defensive):", cv_scores)
y_pred = ridge.predict(X_test)

# Residuals
residuals = y_test - y_pred

# Residuals vs Fitted Values Plot
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# Breusch-Pagan Test for Homoscedasticity
_, p_value, _, _ = het_breuschpagan(residuals, sm.add_constant(X_test))
print("\nBreusch-Pagan p-value:", p_value)


# how plus/minus can be employed to assess offensive and defensive contribution with ridge regression/regular regression
'''
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Define target and features (Offensive vs. Defensive)
# Offensive variables: CF, FF, SF, GF, SCF, HDCF, HDGF, GF%, G, A, P/GP, CF%, SCF%, On-Ice SH%
# Defensive variables: CA, FA, SA, GA, SCF, HDCA, HDGA, On-Ice SV%
X = nhl[['GF%', 'SCF%', 'HDCF%', 'PDO', 'CF%', 'HDGF%', 'FF%']]
y = nhl['+/-']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge = Ridge(alpha=best_alpha)  # alpha is the regularization strength (lambda in ridge regression)
ridge.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(ridge, X_train, y_train, cv=5)
print("Cross-Validation Scores (Offensive/Defensive):", cv_scores)

# Predict on the test set
y_pred = ridge.predict(X_test)
'''

# Offensive alone
X_o = nhl[['CF', 'FF', 'SF', 'GF', 'SCF', 'HDCF', 'HDGF', 'GF%', 'G', 'A', 'P/GP', 'CF%', 'SCF%', 'On-Ice SH%', 'Rush Attempts']]
y_o = nhl['+/-']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_o_scaled = scaler.fit_transform(X_o)

# Split the data into training and test sets
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_o_scaled, y_o, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge_o = Ridge(alpha=1.0)  # alpha is the regularization strength (lambda in ridge regression)
ridge_o.fit(X_train_o, y_train_o)

# Evaluate the model using cross-validation
cv_scores_o = cross_val_score(ridge_o, X_train_o, y_train_o, cv=5)
print("Cross-Validation Scores (Offensive):", cv_scores_o)

# Predict on the test set
y_pred_o = ridge_o.predict(X_test_o)

# Defensive output
X_d = nhl[['SA', 'GA', 'SCF', 'HDCA', 'HDGA', 'On-Ice SV%', 'Hits', 'Shots Blocked']]
y_d = nhl['+/-']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_scaled_d = scaler.fit_transform(X_d)

# Split the data into training and test sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_scaled_d, y_d, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge_d = Ridge(alpha=1.0)  # alpha is the regularization strength (lambda in ridge regression)
ridge_d.fit(X_train_d, y_train_d)

# Evaluate the model using cross-validation
cv_scores_d = cross_val_score(ridge_d, X_train_d, y_train_d, cv=5)
print("Cross-Validation Scores (Defensive):", cv_scores_d)


# Team vs Individual performance
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Simulating team-based random effects (Group means for teams)
nhl['TeamMean_CF%'] = nhl.groupby('Team')['CF%'].transform('mean')
nhl['TeamMean_GF%'] = nhl.groupby('Team')['GF%'].transform('mean')

# Define features (team and player contributions)
X = nhl[['CF%', 'GF%', 'TeamMean_CF%', 'TeamMean_GF%']]
y = nhl['+/-']

# Fit a Mixed-Effects model with team as a random effect
md = MixedLM(y, X, groups=nhl['Team'])
mdf = md.fit()
print(mdf.summary())


# Plus minus vs other evaluators

X_corsi = nhl[['GF%', 'SCF%', 'HDCF%', 'PDO', 'On-Ice SH%', 'On-Ice SV%', 'G', 'A']]
y_cf = nhl['CF%']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_scaled_c = scaler.fit_transform(X_corsi)

# Split the data into training and test sets
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled_c, y_cf, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge_c = Ridge(alpha=1.0)  # alpha is the regularization strength (lambda in ridge regression)
ridge_c.fit(X_train_c, y_train_c)

# Evaluate the model using cross-validation
cv_scores_c = cross_val_score(ridge_c, X_train_c, y_train_c, cv=5)
print("Cross-Validation Scores (Corsi):", cv_scores_c)

# Predict on the test set
y_pred_c = ridge_c.predict(X_test_c)



X_fen = nhl[['GF%', 'SCF%', 'HDCF%', 'PDO', 'On-Ice SH%', 'On-Ice SV%', 'G', 'A']]
y_ff = nhl['FF%']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_scaled_f = scaler.fit_transform(X_fen)

# Split the data into training and test sets
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled_f, y_ff, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge_f = Ridge(alpha=1.0)  # alpha is the regularization strength (lambda in ridge regression)
ridge_f.fit(X_train_f, y_train_f)

# Evaluate the model using cross-validation
cv_scores_f = cross_val_score(ridge_f, X_train_f, y_train_f, cv=5)
print("Cross-Validation Scores (Fenwick):", cv_scores_f)

# Predict on the test set
y_pred_f = ridge_f.predict(X_test_f)
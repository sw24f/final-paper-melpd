import pandas as pd

nhl_edge = pd.read_csv('data\cleaned_nhl_edge.csv')
nst = pd.read_csv('data\Player Season Totals - Natural Stat Trick.csv')
nhl = pd.merge(nhl_edge, nst, on='Player', how='inner')  # 'inner' merge keeps only players present in both datasets

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
# how plus/minus can be employed to assess offensive and defensive contribution with ridge regression/regular regression

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Define target and features (Offensive vs. Defensive)
# Offensive variables: CF, FF, SF, GF, SCF, HDCF, HDGF, GF%, G, A, P/GP, CF%, SCF%, On-Ice SH%
# Defensive variables: CA, FA, SA, GA, SCF, HDCA, HDGA, On-Ice SV%
X = nhl[['GF%', 'SCF%', 'HDCF%', 'PDO', 'On-Ice SH%', 'On-Ice SV%']]
y = nhl['+/-']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge = Ridge(alpha=1.0)  # alpha is the regularization strength (lambda in ridge regression)
ridge.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(ridge, X_train, y_train, cv=5)
print("Cross-Validation Scores (Offensive/Defensive):", cv_scores)

# Predict on the test set
y_pred = ridge.predict(X_test)


# Offensive alone
X_o = nhl[['CF', 'FF', 'SF', 'GF', 'SCF', 'HDCF', 'HDGF', 'GF%', 'G', 'A', 'P/GP', 'CF%', 'SCF%', 'On-Ice SH%']]
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
X_d = nhl[['CA', 'FA', 'SA', 'GA', 'SCF', 'HDCA', 'HDGA', 'On-Ice SV%']]
y_d = nhl['+/-']  # Plus/minus as the target variable

# Scale the features for Ridge Regression
scaler = StandardScaler()
X_scaled_d = scaler.fit_transform(X_d)

# Split the data into training and test sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_scaled_d, y_d, test_size=0.2, random_state=42)

# Fit Ridge Regression Model
ridge = Ridge(alpha=1.0)  # alpha is the regularization strength (lambda in ridge regression)
ridge.fit(X_train_d, y_train_d)

# Evaluate the model using cross-validation
cv_scores_d = cross_val_score(ridge, X_train_d, y_train_d, cv=5)
print("Cross-Validation Scores (Defensive):", cv_scores_d)


# Team vs Individual performance
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Simulating team-based random effects (Group means for teams)
nhl['TeamMean_CF%'] = nhl.groupby('Team')['CF%'].transform('mean')
nhl['TeamMean_GF%'] = nhl.groupby('Team')['GF%'].transform('mean')

# Define features (team and player contributions)
X = nhl[['CF%', 'GF%', 'TeamMean_CF%', 'TeamMean_GF%']]
y = nhl['PlusMinus']

# Fit a Mixed-Effects model with team as a random effect
md = MixedLM(y, X, groups=nhl['Team'])
mdf = md.fit()
print(mdf.summary())


# Plus minus vs other evaluators
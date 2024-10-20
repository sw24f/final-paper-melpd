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

# how plus/minus can be employed to assess offensive and defensive contribution with ridge regression


# Team vs Individual performance


# Plus minus vs other evaluators
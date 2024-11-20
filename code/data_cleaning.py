"""
Player - Player name.
Team - Team or teams that the player has played for. Not displayed when filtering for specific teams.
Position - Position or positions that the player has been listed as playing by the NHL.
GP - Games played.
TOI - Total amount of time played.
Corsi - Any shot attempt (goals, shots on net, misses and blocks) outside of the shootout. Referred to as SAT by the NHL.
CF - Count of Corsi for that player's team while that player is on the ice.
CA - Count of Corsi against that player's team while that player is on the ice.
CF% - Percentage of total Corsi while that player is on the ice that are for that player's team. CF*100/(CF+CA)
Fenwick - any unblocked shot attempt (goals, shots on net and misses) outside of the shootout. Referred to as USAT by the NHL.
FF - Count of Fenwick for that player's team while that player is on the ice.
FA - Count of Fenwick against that player's team while that player is on the ice.
FF% - Percentage of total Fenwick while that player is on the ice that are for that player's team. FF*100/(FF+FA)
Shots - any shot attempt on net (goals and shots on net) outside of the shootout.
SF - Count of Shots for that player's team while that player is on the ice.
SA - Count of Shots against that player's team while that player is on the ice.
SF% - Percentage of total Shots while that player is on the ice that are for that player's team. SF*100/(SF+SA)
Goals - any goal, outside of the shootout.
GF - Count of Goals for that player's team while that player is on the ice.
GA - Count of Goals against that player's team while that player is on the ice.
GF% - Percentage of total Goals while that player is on the ice that are for that player's team. GF*100/(GF+GA)
Scoring Chances - a scoring chance, as originally defined by War-on-Ice
SCF - Count of Scoring Chances for that player's team while that player is on the ice.
SCA - Count of Scoring Chances against that player's team while that player is on the ice.
SCF% - Percentage of total Scoring Chances while that player is on the ice that are for that player's team. SCF*100/(SCF+SCA)
High Danger Scoring Chances - a scoring chance with a score of 3 or higher.
HDCF - Count of High Danger Scoring Chances for that player's team while that player is on the ice.
HDCA - Count of High Danger Scoring Chances against that player's team while that player is on the ice.
HDCF% - Percentage of total High Danger Scoring Chances while that player is on the ice that are for that player's team. HDCF*100/(HDCF+HDCA)
High Danger Goals - goals generated from High Danger Scoring Chances
HDGF - Count of Goals off of High Danger Scoring Chances for that player's team while that player is on the ice.
HDGA - Count of Goals off of High Danger Scoring Chances against that player's team while that player is on the ice.
HDGF% - Percentage of High Danger Goals while that player is on the ice that are for that player's team. HDGF*100/(HDGF+HDGA)
Medium Danger Scoring Chances - a scoring chance with a score of exactly 2.
MDCF - Count of Medium Danger Scoring Chances for that player's team while that player is on the ice.
MDCA - Count of Medium Danger Scoring Chances against that player's team while that player is on the ice.
MDCF% - Percentage of total Medium Danger Scoring Chances while that player is on the ice that are for that player's team. MDCF*100/(MDCF+MDCA)
Medium Danger Goals - goals generated from Medium Danger Scoring Chances
MDGF - Count of Goals off of Medium Danger Scoring Chances for that player's team while that player is on the ice.
MDGA - Count of Goals off of Medium Danger Scoring Chances against that player's team while that player is on the ice.
MDGF% - Percentage of Medium Danger Goals while that player is on the ice that are for that player's team. MDGF*100/(MDGF+MDGA)
Low Danger Scoring Chances - a scoring chance with a score of 1 or less. Does not include any attempts from the attacking team's neutral or defensive zone.
LDCF - Count of Low Danger Scoring Chances for that player's team while that player is on the ice.
LDCA - Count of Low Danger Scoring Chances against that player's team while that player is on the ice.
LDCF% - Percentage of total Low Danger Scoring Chances while that player is on the ice that are for that player's team. LDCF*100/(LDCF+LDCA)
Low Danger Goals - goals generated from Low Danger Scoring Chances
LDGF - Count of Goals off of Low Danger Scoring Chances for that player's team while that player is on the ice.
LDGA - Count of Goals off of Low Danger Scoring Chances against that player's team while that player is on the ice.
LDGF% - Percentage of Low Danger Goals while that player is on the ice that are for that player's team. LDGF*100/(LDGF+LDGA)
PDO
SH% - Percentage of Shots for that player's team while that player is on the ice that were Goals. GF*100/SF
SV% - Percentage of Shots against that player's team while that player is on the ice that were not Goals. GA*100/SA
PDO - Shooting percentage plus save percentage. (GF/SF)+(GA/SA)
Starts
Off. Zone Starts - Number of shifts for the player that started with an offensive zone faceoff.
Neu. Zone Starts - Number of shifts for the player that started with an neutral zone faceoff.
Def. Zone Starts - Number of shifts for the player that started with an defensive zone faceoff.
On The Fly Starts - Number of shifts for the player that started during play (without a faceoff).
Off. Zone Start % - Percentage of starts for the player that were Offensive Zone Starts, excluding Neutral Zone and On The Fly Starts. Off. Zone Starts*100/(Off. Zone Starts+Def. Zone Starts)
Faceoffs
Off. Zone Faceoffs - Number of faceoffs in the offensive zone for which the player was on the ice.
Neu. Zone Faceoffs - Number of faceoffs in the neutral zone for which the player was on the ice.
Def. Zone Faceoffs - Number of faceoffs in the defensive zone for which the player was on the ice.
Off. Zone Faceoff % - Percentage of faceoffs in the offensive zone for which the player was on the ice, excluding neutral zone faceoffs. Off. Zone Faceoffs*100/(Off. Zone Faceoffs+Def. Zone Faceoffs)
"""

import pandas as pd
from functools import reduce
nst = pd.read_csv('data\Player Season Totals - Natural Stat Trick.csv')
print(nst.head())

nst['SF%'].fillna(0, inplace=True)
nst['GF%'].fillna(0, inplace=True)
nst['SCF%'].fillna(0, inplace=True)
nst['HDCF%'].fillna(0, inplace=True)
nst['HDGF%'].fillna(0, inplace = True)
nst['On-Ice SH%'].fillna(0, inplace=True)
nst['On-Ice SV%'].fillna(0, inplace=True)
nst['PDO'].fillna(0, inplace=True)
nst['Off. Zone Start %'].fillna(0, inplace=True)
nst['Off. Zone Faceoff %'].fillna(0, inplace=True)
print(nst.dtypes)

edge1 = pd.read_excel('data\\Summary (1).xlsx')
edge2 = pd.read_excel('data\\Summary (2).xlsx')
edge3 = pd.read_excel('data\\Summary (3).xlsx')
edge4 = pd.read_excel('data\Summary (4).xlsx')
edge5 = pd.read_excel('data\Summary (5).xlsx')
edge6 = pd.read_excel('data\Summary (6).xlsx')
edge7 = pd.read_excel('data\Summary (7).xlsx')
edge8 = pd.read_excel('data\Summary (8).xlsx')
edge9 = pd.read_excel('data\Summary (9).xlsx')
edge10 = pd.read_excel('data\Summary (10).xlsx')
edge11 = pd.read_excel('data\Summary (11).xlsx')
edge12 = pd.read_excel('data\Summary (12).xlsx')
edge13 = pd.read_excel('data\Summary (13).xlsx')

edges = [edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge9, edge10, edge11, edge12, edge13]

import pandas as pd
from functools import reduce

# Helper function to resolve conflicts between two DataFrames during the merge
def merge_and_resolve_conflicts(left, right, on='Player', suffixes=('_x', '_y')):
    merged_df = pd.merge(left, right, on=on, how='outer', suffixes=suffixes)
    
    # Find columns with suffixes
    cols_x = [col for col in merged_df.columns if col.endswith(suffixes[0])]
    cols_y = [col for col in merged_df.columns if col.endswith(suffixes[1])]
    
    for col_x in cols_x:
        # Get the corresponding _y column name
        col_y = col_x.replace(suffixes[0], suffixes[1])
        
        if col_y in merged_df.columns:
            # Create the original column name (without suffixes)
            col_base = col_x.replace(suffixes[0], '')
            
            # Use combine_first() to keep non-null values from both columns
            merged_df[col_base] = merged_df[col_x].combine_first(merged_df[col_y])
            
            # Drop the suffixed columns
            merged_df.drop([col_x, col_y], axis=1, inplace=True)
    
    return merged_df

# Merging multiple DataFrames using reduce and custom conflict resolution
nhl_edge = reduce(lambda left, right: merge_and_resolve_conflicts(left, right), edges)

nhl_edge = nhl_edge.drop(['FOW%', 'PIM'], axis=1)

nhl_edge = nhl_edge[nhl_edge['GP'] > 15]


nhl_edge.fillna(0, inplace=True)

print(nhl_edge.head())
print(nhl_edge.dtypes)

# Function to convert 'Minutes:Seconds' to float
def convert_to_float(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes + (seconds / 60.0)

# Apply the function to the column
nhl_edge['TOI/GP'] = nhl_edge['TOI/GP'].apply(convert_to_float)

nhl_edge.fillna(0, inplace=True)

def clean_columns(df, columns):
    for col in columns:
        df[col] = df[col].replace('--', 0)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean the 'S' and 'S%' columns
nhl_edge = clean_columns(nhl_edge, ['S', 'S%'])

# Check the result
print(nhl_edge[['Player', 'S', 'S%']].head())

# View the result
#print(nhl_edge.head())
#print(nhl_edge.dtypes)

# Specify the path to save the CSV file
csv_file_path = 'data/cleaned_nhl_edge.csv'

# Save the nhl_edge DataFrame as a CSV file
#nhl_edge.to_csv(csv_file_path, index=False)
nst = clean_columns(nst, ['SF%', 'GF%', 'SCF%', 'HDCF%','HDGF%', 'On-Ice SH%', 'On-Ice SV%', 'PDO', 'Off. Zone Start %', 'Off. Zone Faceoff %'])
print(nst.dtypes)
nst.to_csv('data/cleaned_nst.csv', index=False)
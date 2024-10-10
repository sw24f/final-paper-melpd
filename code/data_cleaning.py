"""
Player - Player name.
Team - Team or teams that the player has played for. Not displayed when filtering for specific teams.
Position - Position or positions that the player has been listed as playing by the NHL.
GP - Games played.
TOI - Total amount of time played.
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
"""

import pandas as pd

nhl_stats = pd.read_csv('data\Player Season Totals - Natural Stat Trick.csv')
#print(nhl_stats.head())




print(nhl_stats.dtypes)
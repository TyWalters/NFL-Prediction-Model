# -*- coding: utf-8 -*-

# # NFL Betting Model

# packages
import pandas as pd
import numpy as np
import datetime
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn

# required machine learning packages
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb
import pickle
import os

file_path = os.path.dirname(os.path.abspath("__file__"))

# # Loading and Cleaning Data
# loading CSV files downloaded from Kaggle
df = pd.read_csv(file_path + "\\spreadspoke_scores.csv", encoding='utf-8')
teams = pd.read_csv(file_path + "\\nfl_teams.csv", encoding='utf-8')
games_elo = pd.read_csv(file_path + "\\nfl_games1.csv", encoding='utf-8')
games_elo18 = pd.read_csv(file_path + "\\nfl_games_2019_1.csv", encoding='utf-8')
games_elo = games_elo.append(games_elo18)


# replacing blank strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# removing rows from specific columns that have null values, resetting index and changing data types
df = df[(df.score_home.isnull() == False) & (df.team_favorite_id.isnull() == False) & (df.over_under_line.isnull() == False) &
        (df.schedule_season >= 1979)]


df.reset_index(drop=True, inplace=True)
df['over_under_line'] = df.over_under_line.astype(float)

# mapping team_id to the correct teams
df['team_home'] = df.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
df['team_away'] = df.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

# fix team_favorite_id for Colts in 1969 and 1971 SB
df.loc[(df.schedule_season == 1968) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'
df.loc[(df.schedule_season == 1970) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'

# creating home favorite and away favorite columns (fill na with 0's)
df.loc[df.team_favorite_id == df.team_home, 'home_favorite'] = 1
df.loc[df.team_favorite_id == df.team_away, 'away_favorite'] = 1
df.home_favorite.fillna(0, inplace=True)
df.away_favorite.fillna(0, inplace=True)

# creating over / under column (fill na with 0's)
df.loc[((df.score_home + df.score_away) > df.over_under_line), 'over'] = 1
df.over.fillna(0, inplace=True)

# stadium neutral and schedule playoff as boolean
df['stadium_neutral'] = df.stadium_neutral.astype(int)
df['schedule_playoff'] = df.schedule_playoff.astype(int)

# change data type of date columns
df['schedule_date'] = pd.to_datetime(df['schedule_date'])
games_elo['date'] = pd.to_datetime(games_elo['date'])

# fixing some schedule_week column errors and converting column to integer data type
df.loc[(df.schedule_week == '18'), 'schedule_week'] = '17'
df.loc[(df.schedule_week == 'Wildcard') | (df.schedule_week == 'WildCard'), 'schedule_week'] = '18'
df.loc[(df.schedule_week == 'Division'), 'schedule_week'] = '19'
df.loc[(df.schedule_week == 'Conference'), 'schedule_week'] = '20'
df.loc[(df.schedule_week == 'Superbowl') | (df.schedule_week == 'SuperBowl'), 'schedule_week'] = '21'
df['schedule_week'] = df.schedule_week.astype(int)

# removing extra columns that aren't necessary for analysis
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite',
       'over_under_line', 'weather_temperature',
       'weather_wind_mph', 'score_home', 'score_away',
       'stadium_neutral', 'home_favorite', 'away_favorite',
       'over']]

# Cleaning games_elo and df to merge correctly
wsh_map = {'WSH' : 'WAS'}
games_elo.loc[games_elo.team1 == 'WSH', 'team1'] = 'WAS' 
games_elo.loc[games_elo.team2 == 'WSH', 'team2'] = 'WAS'

# fix dates
df.loc[(df.schedule_date == '2016-09-19') & (df.team_home == 'MIN'), 'schedule_date'] = datetime.datetime(2016, 9, 18)
df.loc[(df.schedule_date == '2017-01-22') & (df.schedule_week == 21), 'schedule_date'] = datetime.datetime(2017, 2, 5)
df.loc[(df.schedule_date == '1990-01-27') & (df.schedule_week == 21), 'schedule_date'] = datetime.datetime(1990, 1, 28)
df.loc[(df.schedule_date == '1990-01-13'), 'schedule_date'] = datetime.datetime(1990, 1, 14)
games_elo.loc[(games_elo.date == '2016-01-09'), 'date'] = datetime.datetime(2016, 1, 10)
games_elo.loc[(games_elo.date == '2016-01-08'), 'date'] = datetime.datetime(2016, 1, 9)
games_elo.loc[(games_elo.date == '2016-01-16'), 'date'] = datetime.datetime(2016, 1, 17)
games_elo.loc[(games_elo.date == '2016-01-15'), 'date'] = datetime.datetime(2016, 1, 16)

# merge games_elo with df
df = df.merge(games_elo, left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'], how='left')

# merge to fix neutral games where team_home and team_away are switched
games_elo2 = games_elo.rename(columns={'team1' : 'team2', 'team2' : 'team1', 'elo1_pre' : 'elo2_pre', 'elo2_pre' : 'elo1_pre'})
games_elo2['qbelo_prob1'] = 1 - games_elo2.qbelo_prob1
df = df.merge(games_elo2, left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'], how='left')


# separating merged columns into x and y cols
x_cols = ['date_x', 'season_x', 'neutral_x', 'playoff_x', 'team1_x', 'team2_x', 'elo1_pre_x', 'elo2_pre_x',
          'qbelo_prob1_x', 'score1_x', 'score2_x']
y_cols = ['date_y', 'season_y', 'neutral_y', 'playoff_y', 'team1_y', 'team2_y', 'elo1_pre_y', 'elo2_pre_y',
          'qbelo_prob1_y', 'score1_y', 'score2_y']

# filling null values for games_elo merged cols
for x, y in zip(x_cols, y_cols):
    df[x] = df[x].fillna(df[y]) 

# removing y_cols from dataframe    
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
       'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away',
       'stadium_neutral', 'home_favorite', 'away_favorite', 'over', 'neutral_x', 'playoff_x',
         'elo1_pre_x', 'elo2_pre_x', 'qbelo_prob1_x']]

# remove _x ending from column names
df.columns = df.columns.str.replace('_x', '')


# creating result column df.loc[(df.score_home > df.score_away), 'result'
df['result'] = (df.score_home > df.score_away).astype(int)


# # Data Exploration

# all column names
print(df.columns)


# snapshot of data
print(df.head(10))


# null values by column
print(df.isnull().sum(axis=0))


# summary statistics
df.describe().transpose()


# some percentages to take into consideration when betting
home_win = "{:.2f}".format((sum((df.result == 1) & (df.stadium_neutral == 0)) / len(df)) * 100)
away_win = "{:.2f}".format((sum((df.result == 0) & (df.stadium_neutral == 0)) / len(df)) * 100)
under_line = "{:.2f}".format((sum((df.score_home + df.score_away) < df.over_under_line) / len(df)) * 100)
over_line = "{:.2f}".format((sum((df.score_home + df.score_away) > df.over_under_line) / len(df)) * 100)

favored = "{:.2f}".format((sum(((df.home_favorite == 1) & (df.result == 1)) | ((df.away_favorite == 1) & (df.result == 0)))
                           / len(df)) * 100)

cover = "{:.2f}".format((sum(((df.home_favorite == 1) & ((df.score_away - df.score_home) < df.spread_favorite)) | 
                             ((df.away_favorite == 1) & ((df.score_home - df.score_away) < df.spread_favorite))) 
                         / len(df)) * 100)

ats = "{:.2f}".format((sum(((df.home_favorite == 1) & ((df.score_away - df.score_home) > df.spread_favorite)) | 
                           ((df.away_favorite == 1) & ((df.score_home - df.score_away) > df.spread_favorite))) 
                       / len(df)) * 100)


# print all percentages
print("Number of Games: " + str(len(df)))
print("Home Straight Up Win Percentage: " + home_win + "%")
print("Away Straight Up Win Percentage: " + away_win + "%")
print("Under Percentage: " + under_line + "%")
print("Over Percentage: " + over_line + "%")
print("Favored Win Percentage: " + favored + "%")
print("Cover The Spread Percentage: " + cover + "%")
print("Against The Spread Percentage: " + ats + "%")


# # Creating More Features
# Features that will help predict whether the home team will win (straight up win in the current version the line doesn't matter)

# creating 2 separate dataframes with the home teams / scores and the away teams / scores
score = df.groupby(['schedule_season', 'schedule_week', 'team_home']).mean()[['score_home', 'score_away']].reset_index()
aw_score = df.groupby(['schedule_season', 'schedule_week', 'team_away']).mean()[['score_home', 'score_away']].reset_index()

# create total pts column
score['point_diff'] = score.score_home - score.score_away
aw_score['point_diff'] = aw_score.score_away - aw_score.score_home

# append the two dataframes
score = score.append(aw_score, ignore_index=True, sort=True)

# fill null values
score.team_home.fillna(score.team_away, inplace=True)

# sort by season and week 
score.sort_values(['schedule_season', 'schedule_week'], ascending = [True, True], inplace=True)

# removing unneeded columns & changing column name 
score = score[['schedule_season', 'schedule_week', 'team_home', 'point_diff']]
score.rename(columns={'team_home' : 'team'}, inplace=True)

# dictionary of dataframes - separate dataframe for each team
tm_dict = {}
for key in score.team.unique():
    tm_dict[key] = score[score.team == key].reset_index(drop=True)


# dataframe to populate
pts_diff = pd.DataFrame()

# for loop to create a rolling average of the previous games for each season
for yr in score.schedule_season.unique():
    for tm in score.team.unique():
        data = tm_dict[tm].copy()
        data = data[data.schedule_season == yr]
        
        data.loc[:, 'avg_pts_diff'] = data.point_diff.shift().expanding().mean()
        
        pts_diff = pts_diff.append(data)


# merging to df and changing column names
df = df.merge(pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']], 
              left_on=['schedule_season', 'schedule_week', 'team_home'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'avg_pts_diff' : 'hm_avg_pts_diff'}, inplace=True)

df = df.merge(pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']], 
              left_on=['schedule_season', 'schedule_week', 'team_away'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'avg_pts_diff' : 'aw_avg_pts_diff'}, inplace=True)


# average point differential over entire season
total_season = pts_diff.groupby(['schedule_season', 'team']).mean()['point_diff'].reset_index()


# adding schedule week for merge and adding one to the season for predictions
total_season['schedule_week'] = 1
total_season['schedule_season'] += 1


# differentials for start of the year
# total_season.to_csv("pointdiff2019.csv", index=False)


# cleaning of columns
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
       'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away', 'stadium_neutral', 'home_favorite',
       'away_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1_pre', 'elo2_pre', 'qbelo_prob1', 'over', 'result']]


# merge to have previous seasons average point differential
df = df.merge(total_season[['schedule_season', 'schedule_week', 'team', 'point_diff']], 
              left_on=['schedule_season', 'schedule_week', 'team_home'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'point_diff' : 'hm_avg_diff'}, inplace=True)

df = df.merge(total_season[['schedule_season', 'schedule_week', 'team', 'point_diff']], 
              left_on=['schedule_season', 'schedule_week', 'team_away'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'point_diff' : 'aw_avg_diff'}, inplace=True)

# fill null values
df.hm_avg_pts_diff.fillna(df.hm_avg_diff, inplace=True)
df.aw_avg_pts_diff.fillna(df.aw_avg_diff, inplace=True)


# cleaning of columns
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
       'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away', 'stadium_neutral', 'home_favorite',
       'away_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1_pre', 'elo2_pre', 'qbelo_prob1', 'over', 'result']]


# removing all rows with null values
df = df.dropna(how='any',axis=0) 


# # Feature and Model Testing 

# initial features possible for model
X = df[['schedule_season', 'schedule_week', 'over_under_line', 'spread_favorite', 'weather_temperature', 'weather_wind_mph',
        'home_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1_pre', 'elo2_pre', 'qbelo_prob1']]

y = df['result']


# base model
base = LDA()

# choose 5 best features
rfe = RFE(base, 5)
rfe = rfe.fit(X, y)

# features
print(rfe.support_)
print(rfe.ranking_)


# best 5 features chosen by the RFE base model
final_x = df[['spread_favorite', 'home_favorite', 'hm_avg_pts_diff', 'elo2_pre', 'qbelo_prob1']]



# prepare models
models = []

models.append(('LRG', LogisticRegression(solver='liblinear')))
models.append(('KNB', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('XGB', xgb.XGBClassifier(random_state=0)))
models.append(('RFC', RandomForestClassifier(random_state=0, n_estimators=100)))
models.append(('DTC', DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)))

# evaluate each model by average and standard deviations of roc auc 
results = []
names = []

for name, m in models:
    kfold = model_selection.KFold(n_splits=5, random_state=0)
    cv_results = model_selection.cross_val_score(m, final_x, y, cv=kfold, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# training and testing data (2017 and 2018)
train = df.copy()
test = df.copy()
train = train.loc[train['schedule_season'] < 2017]
test = test.loc[test['schedule_season'] > 2016]
X_train = train[['over_under_line', 'spread_favorite', 'home_favorite', 'hm_avg_pts_diff', 'qbelo_prob1']]
y_train = train['result']
X_test = test[['over_under_line', 'spread_favorite', 'home_favorite', 'hm_avg_pts_diff', 'qbelo_prob1']]
y_test = test['result']

# calibrate probabilities and fit model to training data
boost = xgb.XGBClassifier()
dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy')
lrg = LogisticRegression(solver='liblinear')
vote = VotingClassifier(estimators=[('boost', boost), ('dtc', dtc), ('lrg', lrg)], voting='soft')

model = CCV(vote, method='isotonic', cv=3)
model.fit(X_train, y_train)

# predict probabilities
predicted = model.predict_proba(X_test)[:,1]

# # Results
# ROC AUC Score higher is better while Brier Score the lower the better
print("Metrics" + "\t\t" + "My Model" + "\t" + "Elo Results")
print("ROC AUC Score: " +  "\t" + "{:.4f}".format(roc_auc_score(y_test, predicted)) + "\t\t" + "{:.4f}".format(roc_auc_score(test.result, test.qbelo_prob1)))
print("Brier Score: " + "\t" + "{:.4f}".format(brier_score_loss(y_test, predicted)) + "\t\t" + "{:.4f}".format(brier_score_loss(test.result, test.qbelo_prob1)))

# creating a column with the models probabilities to analyze vs elo fivethirtyeight
test.loc[:,'hm_prob'] = predicted
test = test[['schedule_season', 'schedule_week', 'team_home', 'team_away', 'qbelo_prob1', 'hm_prob', 'result']]

# calulate bets won (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_won'] = (((test.hm_prob >= 0.60) & (test.result == 1)) | ((test.hm_prob <= 0.40) & (test.result == 0))).astype(int)
test['elo_bet_won'] = (((test.qbelo_prob1 >= 0.60) & (test.result == 1)) | ((test.qbelo_prob1 <= 0.40) & (test.result == 0))).astype(int)

# calulate bets lost (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_lost'] = (((test.hm_prob >= 0.60) & (test.result == 0)) | ((test.hm_prob <= 0.40) & (test.result == 1))).astype(int)
test['elo_bet_lost'] = (((test.qbelo_prob1 >= 0.60) & (test.result == 0)) | ((test.qbelo_prob1 <= 0.40) & (test.result == 1))).astype(int)

# printing some quick overall results for my model
print("My Model Win Percentage: " + "{:.4f}".format(test.my_bet_won.sum() / (test.my_bet_lost.sum() + test.my_bet_won.sum())))
print("Total Number of Bets Won: " + str(test.my_bet_won.sum()))
print("Total Number of Bets Made: " + str((test.my_bet_lost.sum() + test.my_bet_won.sum())))
print("Possible Games: " + str(len(test)))

# printing some quick overall results for fivethirtyeight's ELO model
print("ELO Model Win Percentage: " + "{:.4f}".format(test.elo_bet_won.sum()/(test.elo_bet_lost.sum() + test.elo_bet_won.sum())))
print("Total Number of Bets Won: " + str(test.elo_bet_won.sum()))
print("Total Number of Bets Made: " + str((test.elo_bet_lost.sum() + test.elo_bet_won.sum())))
print("Possible Games: " + str(len(test)))

# creating week by week results
results_df = test.groupby(['schedule_season', 'schedule_week']).agg({'team_home' : 'count', 'my_bet_won' : 'sum', 
'elo_bet_won' : 'sum', 'my_bet_lost' : 'sum', 'elo_bet_lost' : 'sum'}).reset_index().rename(columns=
                                                                                            {'team_home' : 'total_games'})

# counting total bets for my model and the ELO model (prob >= 60% or prob <= 40%)
results_df['total_bets'] = results_df.my_bet_won + results_df.my_bet_lost
results_df['elo_total_bets'] = results_df.elo_bet_won + results_df.elo_bet_lost

# creating accuracy columns based on bets made not on total games
results_df['bet_accuracy'] = round((results_df.my_bet_won / results_df.total_bets) * 100, 2)
results_df['elo_bet_accuracy'] = round((results_df.elo_bet_won / results_df.elo_total_bets) * 100, 2)
results_df = results_df[['schedule_season', 'schedule_week', 'bet_accuracy', 'elo_bet_accuracy',
                         'total_bets', 'elo_total_bets', 'total_games']]


print(results_df)

# save model as pkl file for weekly predictions
pkl_filename = "NFLMoneyLine_model1.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

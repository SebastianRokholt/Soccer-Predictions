# Importing dependencies
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)


def preprocess(user_input):
    """
    Function for preprocessing the user input
    :param user_input: a Numpy array on the form [Team, Opponent, Venue]
    :return: a Numpy array with 40 features
    """

    """
    User input
    """
    # print(user_input)  # [array(['LSK Kvinner', 'Klepp', 'Home', '3'], dtype='<U11')]
    # If the user didn't specify the game number
    if len(user_input[0]) == 3:
        sample_df = pd.DataFrame(user_input, columns=["Team", "Opponent", "Venue"])
        # Setting the game number to zero (default value)
        sample_df["Game # in season"] = "0"
    else:
        sample_df = pd.DataFrame(user_input, columns=["Team", "Opponent", "Venue", "Game # in season"])

    """
    One Hot Encoding
    """
    teams = ['Arna-Bjørnar', 'Avaldsnes', 'Fart', 'Grand Bodø', 'IF Fløya',
             'Klepp', 'Kolbotn', 'LSK Kvinner', 'Lyn', 'Medkila', 'Røa',
             'Sandviken', 'Stabæk', 'Trondheims-Ørn', 'Vålerenga']
    # Getting the team and opponent names from the user input
    team_name = user_input[0][0]
    opponent_name = user_input[0][1]

    # Add dummy variables for Team and Opponent
    for team in teams:
        sample_df[f"Team_{team}"] = 0
        sample_df[f"Opponent_{team}"] = 0
    # Add and set dummy variables for Venue
    if sample_df["Venue"].iloc[0] == "Home":
        sample_df["Venue_Home"] = 1
        sample_df["Venue_Away"] = 0
    else:
        sample_df["Venue_Home"] = 0
        sample_df["Venue_Away"] = 1

    # Set the values for the dummy variables
    sample_df[f"Team_{team_name}"] = 1
    sample_df[f"Opponent_{opponent_name}"] = 1
    # Remove the original features for Team and Opponent
    sample_df.drop(columns=["Team", "Opponent", "Venue"], inplace=True)

    """
    Adding the average historical score difference to the sample
    """
    # Loading the training data
    games_df = pd.read_csv("../Prepared Data/games_prepared.csv")

    # Retrieve the most recent value for AHSD in the training data
    sample_df["Avg. Historical Score Diff"] = games_df.loc[(games_df["Season"] == 2019)
                                                           & (games_df[f"Team_{team_name}"] == 1)
                                                           & (games_df[f"Opponent_{opponent_name}"] == 1)]["Avg. Historical Score Diff"].iloc[-1]
    """
    Adding the features "Team's Goals per 90 min last season" and "Opponent's Goals per 90 min last season" to the sample
    """
    # Load the cleaned teams data for 2019
    teams_2019_df = pd.read_csv("../Prepared Data/2019_team-stats_prepared.csv")

    # The value to be imputed if the team didn't participate in Toppserien in 2019
    # The value is calculated in preparations.ipynb. It is half a standard deviation away from the mean.
    goals_2019_low_avg_score = 0.8286568783754502

    # Retrieve Team's Goals per 90 min last season
    try:
        team_avg_gls_season_2019 = teams_2019_df.loc[
            (teams_2019_df['Unnamed: 0_level_0'] == team_name), "Per 90 Minutes.1"].iloc[0]
        sample_df["Team's goals per 90 min last season"] = team_avg_gls_season_2019
    except IndexError:
        sample_df["Team's goals per 90 min last season"] = goals_2019_low_avg_score

    # Retrieve Opponent's Goals per 90 min last season
    try:
        opponents_avg_gls_season_2019 = teams_2019_df.loc[
            (teams_2019_df['Unnamed: 0_level_0'] == opponent_name), "Per 90 Minutes.1"].iloc[0]
        sample_df["Opponent's goals per 90 min last season"] = opponents_avg_gls_season_2019
    except IndexError:
        sample_df["Opponent's goals per 90 min last season"] = low_avg_score

    """
    Adding the feature "Team's average age" and "Opponent's average age" to the sample
    """
    try:
        team_avg_age = teams_2019_df.loc[
            (teams_2019_df['Unnamed: 0_level_0'] == team_name), 'Unnamed: 2_level_0'].iloc[0]
    except IndexError or KeyError:
        team_avg_age = 23.41  # Imputing mean value

    try:
        opponent_avg_age = teams_2019_df.loc[
            (teams_2019_df['Unnamed: 0_level_0'] == opponent_name), 'Unnamed: 2_level_0'].iloc[0]
    except IndexError or KeyError:
        opponent_avg_age = 23.41  # Imputing mean value

    sample_df["Team's Avg. Age"] = team_avg_age
    sample_df["Opponent's Avg. Age"] = opponent_avg_age

    """
    Adding the feature "Top 5 Team" and "Top 5 Opponent" to the sample
    """
    top_5_team = []
    top_5_opponent = []

    table_2019_df = pd.read_csv("../Prepared Data/2019_table_prepared.csv")
    team_placement = table_2019_df.loc[(table_2019_df["Squad"] == team_name), "Rk"]
    opponent_placement = table_2019_df.loc[(table_2019_df["Squad"] == opponent_name), "Rk"]

    # Calculating whether the team is a top 5 team
    try:
        if int(team_placement) < 6:
            top_5_team.append(1)
        else:
            top_5_team.append(0)

    except TypeError:
        # TypeError occurs when the team can't be found in the table
        # Which means they were recently promoted from 1st division
        top_5_team.append(0)

    # Calculating whether the opponent is a top 5 opponent
    try:
        if int(opponent_placement) < 6:
            top_5_opponent.append(1)
        else:
            top_5_opponent.append(0)

    except TypeError as err:
        # TypeError occurs when the opponent can't be found in the table
        # Which means they were recently promoted from 1st division
        top_5_opponent.append(0)

    # Creating the two new features with the calculated values
    sample_df["Top 5 Team"] = top_5_team
    sample_df["Top 5 Opponent"] = top_5_opponent

    """
    Adding the "Season" feature
    """
    # I don't actually know which season the model is predicting for, but I can assume that it is a season in the future.
    # Since the most recent data is from 2019, the optimal choice is to set the value to 2020 so that nothing unexpected occurs.
    sample_df["Season"] = 2020

    """
    Setting the order of the features
    """
    sample_df = sample_df[["Season", 'Team_Arna-Bjørnar', 'Team_Avaldsnes',
                           'Team_Fart', 'Team_Grand Bodø', 'Team_IF Fløya', 'Team_Klepp',
                           'Team_Kolbotn', 'Team_LSK Kvinner', 'Team_Lyn', 'Team_Medkila',
                           'Team_Røa', 'Team_Sandviken', 'Team_Stabæk', 'Team_Trondheims-Ørn',
                           'Team_Vålerenga', 'Opponent_Arna-Bjørnar', 'Opponent_Avaldsnes',
                           'Opponent_Fart', 'Opponent_Grand Bodø', 'Opponent_IF Fløya',
                           'Opponent_Klepp', 'Opponent_Kolbotn', 'Opponent_LSK Kvinner',
                           'Opponent_Lyn', 'Opponent_Medkila', 'Opponent_Røa',
                           'Opponent_Sandviken', 'Opponent_Stabæk', 'Opponent_Trondheims-Ørn',
                           'Opponent_Vålerenga', 'Venue_Away', 'Venue_Home', "Game # in season",
                           "Avg. Historical Score Diff",
                           "Team's goals per 90 min last season",
                           "Opponent's goals per 90 min last season", "Team's Avg. Age",
                           "Opponent's Avg. Age", "Top 5 Team", "Top 5 Opponent"]]

    """
    Converting dataframe features of datatype str to datatype int
    + converting sample from dataframe to numpy array and
    reshaping the array to 2D
    """
    sample_series = pd.to_numeric(sample_df.iloc[0])
    preprocessed_sample = np.array(sample_series).reshape(1, -1)

    return preprocessed_sample


# Test
# result = preprocess([np.array(['LSK Kvinner', 'Klepp', 'Home', '3'], dtype='<U11')])
# print(result)

#对games_table8.py的结果进行分析
#environment setup  C:\Users\Administrator\Desktop\simple eda\simple eda\EDA\player_table.ipynb
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from prefect import flow, task
import os
print("当前工作目录是：", os.getcwd())

@task(name="calculate_player_games")
def calculate_player_games(df):
    """_summary_
    count the number of each games each player played as White and Black,and compute the total

    Args:
        df (pd.Dataframe): DataFrame containing chess game data. Must include 'White', 'Black', and 'Event' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns: 'user', 'white_games', 'black_games', and 'total_games', 
          sorted by total games played in descending order.
    """    
    games_white = df.groupby('White').count()['Event'].sort_values(ascending=False).reset_index()
    games_black = df.groupby('Black').count()['Event'].sort_values(ascending=False).reset_index()
    
    games_white.columns = ['user', 'white_games']
    games_black.columns = ['user', 'black_games']
    
    df_games = pd.merge(games_white, games_black)
    df_games['total_games'] = df_games.white_games + df_games.black_games
    df_games = df_games.sort_values(by='total_games', ascending=False)
    
    return df_games

@task(name="calculate_player_wins")
def calculate_player_wins(df):
    """_summary_
     Calculate the number of wins each player achieved as White and as Black, and compute the total wins.

    Args:
        df (pd.DataFrame): DataFrame containing chess game data. Must include 'White', 'Black', 'Result', and 'Event' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns: 'user', 'wins_white', 'wins_black', and 'wins_total'.
                      Only games with a decisive result (not draws) are counted.
    """    
    df_wins_white = df[['White', 'Result', 'Event']].groupby(['White', 'Result']).count().reset_index()
    df_wins_white = df_wins_white[df_wins_white.Result=='1-0'].groupby('White').sum()['Event'].reset_index()
    df_wins_white.columns = ['user', 'wins_white']

    df_wins_black = df[['Black', 'Result', 'Event']].groupby(['Black', 'Result']).count().reset_index()
    df_wins_black = df_wins_black[df_wins_black.Result=='0-1'].groupby('Black').sum()['Event'].reset_index()
    df_wins_black.columns = ['user', 'wins_black']
    
    df_wins = pd.merge(df_wins_white, df_wins_black)
    df_wins['wins_total'] = df_wins['wins_white'] + df_wins['wins_black']
    
    return df_wins

@task(name="calculate_player_draws")
def calculate_player_draws(df):
    """ Calculate the number of draws for each player as White and as Black, and compute the total number of draws.

    Args:
        df (pd.DataFrame): DataFrame containing chess game data. Must include the columns 'White', 'Black', 'Result', and 'Event'.

    Returns:
        pd.DataFrame: A DataFrame with columns: 'user', 'draw_white', 'draw_black', and 'draw_total',
                      indicating draw statistics for each player.
    """
    df_draw_white = df[['White', 'Result', 'Event']].groupby(['White', 'Result']).count().reset_index()
    df_draw_white = df_draw_white[df_draw_white.Result=='1/2-1/2'].groupby('White').sum()['Event'].reset_index()
    df_draw_white.columns = ['user', 'draw_white']

    df_draw_black = df[['Black', 'Result', 'Event']].groupby(['Black', 'Result']).count().reset_index()
    df_draw_black = df_draw_black[df_draw_black.Result=='1/2-1/2'].groupby('Black').sum()['Event'].reset_index()
    df_draw_black.columns = ['user', 'draw_black']
    
    df_draw = pd.merge(df_draw_white, df_draw_black)
    df_draw['draw_total'] = df_draw['draw_white'] + df_draw['draw_black']
    
    return df_draw

@task(name="calculate_player_losses")
def calculate_player_losses(df):
    """ Calculate the number of losses for each player as White and as Black, and compute the total number of losses.

    Args:
        df (pd.DataFrame): DataFrame containing chess game data. Must include the columns 'White', 'Black', 'Result', and 'Event'.

    Returns:
        pd.DataFrame
    """    
    df_lose_white = df[['White', 'Result', 'Event']].groupby(['White', 'Result']).count().reset_index()
    df_lose_white = df_lose_white[df_lose_white.Result=='0-1'].groupby('White').sum()['Event'].reset_index()
    df_lose_white.columns = ['user', 'lose_white']

    df_lose_black = df[['Black', 'Result', 'Event']].groupby(['Black', 'Result']).count().reset_index()
    df_lose_black = df_lose_black[df_lose_black.Result=='1-0'].groupby('Black').sum()['Event'].reset_index()
    df_lose_black.columns = ['user', 'lose_black']
    
    df_lose = pd.merge(df_lose_white, df_lose_black)
    df_lose['lose_total'] = df_lose['lose_white'] + df_lose['lose_black']
    
    return df_lose

@task(name="calculate_player_elo")
def calculate_player_elo(df):
    """_summary_
    calculate average elo ratintg for players

    Args:
        df pd.Datafram):

    Returns:
         pd.DataFrame: A DataFrame with columns: 'user' and 'elo', 
         representing each player's average ELO rating.
    """ 
    white_elos = df[['White', 'WhiteElo']]
    black_elos = df[['Black', 'BlackElo']]
    
    white_elos.columns = ['user', 'elo']
    black_elos.columns = ['user', 'elo']
    
    df_elo = pd.concat([white_elos, black_elos])
    df_elo = df_elo.groupby('user').mean().reset_index()
    
    return df_elo

@task(name="calculate_player_accuracy")
def calculate_player_accuracy(df):
    """_summary_
    Calculate the average accuracy for each player, including overall accuracy 
    and phase-wise accuracies (opening, middlegame, endgame)

    Args:
        df (pd.DataFrame): DataFrame containing chess game data. Must include the following columns:
                           - 'White', 'Black'
                           - 'White Accuracy', 'Black Accuracy'
                           - 'White Beginning Accuracy', 'Black Beginning Accuracy'
                           - 'White Middle Accuracy', 'Black Middle Accuracy'
                           - 'White Endgame Accuracy', 'Black Endgame Accuracy'

    Returns:
        pd.DataFrame: A DataFrame with columns: 'user', 'accuracy', 
                      'accuracy_opening', 'accuracy_middlegame', and 'accuracy_endgame',
                      representing the average accuracy statistics for each player.
    """
    df_ea = pd.concat([
        df[['White', 'White Accuracy', 
            'White Beginning Accuracy', 'White Middle Accuracy', 'White Endgame Accuracy']]
        .rename(columns={
            'White':'user', 
            'White Accuracy':'accuracy', 
            'White Beginning Accuracy': 'accuracy_opening',
            'White Middle Accuracy': 'accuracy_middlegame',
            'White Endgame Accuracy': 'accuracy_endgame'
        }),
        
        df[['Black', 'Black Accuracy',
            'Black Beginning Accuracy', 'Black Middle Accuracy', 'Black Endgame Accuracy']]
        .rename(columns={
            'Black':'user',
            'Black Accuracy':'accuracy', 
            'Black Beginning Accuracy': 'accuracy_opening',
            'Black Middle Accuracy': 'accuracy_middlegame',
            'Black Endgame Accuracy': 'accuracy_endgame'
        })
    ], axis=0)
    
    df_ea = df_ea.dropna().groupby('user').mean().reset_index()
    
    return df_ea

@flow(name="comprehensive_player_analysis")
def comprehensive_player_analysis(df,output_dir, output_filename):
    """_summary_
     Perform a comprehensive analysis of each player's performance, including games played, 
    wins/draws/losses, ELO, accuracy, and derived metrics

    Args:
        df (pd.Dataframe):containing all chess games with fields for 
                          players, results, ELO, and accuracy.

    Returns:
        pd.DataFrame: Aggregated player-level statistics
    """    
    os.makedirs(output_dir,exist_ok=True)
    output_path=os.path.join(output_dir, output_filename)

    #calculate each index
    df_games = calculate_player_games(df)
    df_wins = calculate_player_wins(df)
    df_draws = calculate_player_draws(df)
    df_losses = calculate_player_losses(df)
    df_elo = calculate_player_elo(df)
    df_accuracy = calculate_player_accuracy(df)
    
    #merge
    df_all = df_games.merge(df_wins, on='user', how='left')
    df_all = df_all.merge(df_draws, on='user', how='left')
    df_all = df_all.merge(df_losses, on='user', how='left')
    df_all = df_all.merge(df_elo, on='user', how='left')
    df_final = df_all.merge(df_accuracy, on='user', how='left')
    df_final = df_final.fillna(0)
    
    #calculate win rate and the accuracy distance between whole-game accuracy and stage accuracy
    df_final['win_rate'] = df_final['wins_total'] / df_final['total_games']
    df_final['OpenAcc-meanAcc'] = df_final['accuracy_opening'] - df_final['accuracy']
    df_final['MiddleAcc-meanAcc'] = df_final['accuracy_middlegame'] - df_final['accuracy']
    df_final['EndAcc-meanAcc'] = df_final['accuracy_endgame'] - df_final['accuracy']
    
    df_final['OpenAcc-meanOpenAcc'] = df_final['accuracy_opening'] - df_final['accuracy_opening'].mean()
    df_final['MiddleAcc-meanMiddleAcc'] = df_final['accuracy_middlegame'] - df_final['accuracy_middlegame'].mean()
    df_final['EndAcc-meanEndAcc'] = df_final['accuracy_endgame'] - df_final['accuracy_endgame'].mean()
    
    df_final.to_csv(output_path, index=False)
    print(f"results have been saved to{output_path}")
    
    return df_final

if __name__ == "__main__":
    
    current_dir=Path(__file__).parent

    df = pd.read_csv(current_dir/"..\data\output\mpi1686.csv") 
    output_dir=current_dir/"..\data\output"
    output_dir=output_dir.resolve()
    output_filename="playerstats.csv"

    result = comprehensive_player_analysis(df,output_dir, output_filename)
    print(result)
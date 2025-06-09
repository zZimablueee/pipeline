# main.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import chess
import os
import sys
import chess.pgn
import chess.engine
import csv
import io
import traceback
import subprocess
import time
import asyncio
from tqdm import tqdm
from pathlib import Path
from stockfish import Stockfish
from prefect.tasks import task_input_hash
# from prefect.cache_policies import NO_CACHE
from prefect import flow, task, get_run_logger, serve
import numpy as np
from analysis import * 

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@task(name="analyze_chess_games")
def analyze_csv(input_csv_file_path, engine_path, threads, depth, is_verbose=False):
    """
    Analyze chess games in CSV file, calculate accuracy and CP loss for each game,
    and add results to the original DataFrame.

    Inputs
    ------
    input_csv_file_path (str): The file path of the input CSV file, containing raw data of chess games from PGN files. 
    engine_path (str): The path to Stockfish engine. 
    threads (int): Number of threads to use
    depth (int): The depth for Stockfish to evaluate moves.
    is_verbose (bool): Whether to output internal messages.

    Outputs
    -------
    result_dict (dict): ??
    original_df (pd.DataFrame): Dataframe of chess games, augmented with new features including accuracy and cp loss.
    """

    # Read CSV to DataFrame
    with open(input_csv_file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        delimiter = '\t' if '\t' in first_line else ','
    original_df = pd.read_csv(input_csv_file_path, delimiter=delimiter)
    
    engine = SimpleStockfishEngine(engine_path, threads)
    
    try:
        all_games, game_details = process_csv(input_csv_file_path, engine, depth, is_verbose)
        
        # Create empty lists with same length as original DataFrame
        white_cp_loss_list = [None] * len(original_df)
        white_accuracy_list = [None] * len(original_df)
        black_cp_loss_list = [None] * len(original_df)
        black_accuracy_list = [None] * len(original_df)
        
        # Add stage analysis result lists
        white_beginning_acc = [None] * len(original_df)
        white_beginning_std = [None] * len(original_df)
        white_middle_acc = [None] * len(original_df)
        white_middle_std = [None] * len(original_df)
        white_endgame_acc = [None] * len(original_df)
        white_endgame_std = [None] * len(original_df)
        
        black_beginning_acc = [None] * len(original_df)
        black_beginning_std = [None] * len(original_df)
        black_middle_acc = [None] * len(original_df)
        black_middle_std = [None] * len(original_df)
        black_endgame_acc = [None] * len(original_df)
        black_endgame_std = [None] * len(original_df)
        
        # Valid game statistics
        valid_games = 0
        total_avg_cp_white = total_avg_cp_black = 0.0
        total_acc_white = total_acc_black = 0.0
        
        for game in all_games:
            row_idx, white, black, avg_cp_white, acc_white, avg_cp_black, acc_black, stage_analysis = game
            
            # Fill results in corresponding index positions
            white_cp_loss_list[row_idx] = avg_cp_white
            white_accuracy_list[row_idx] = acc_white
            black_cp_loss_list[row_idx] = avg_cp_black
            black_accuracy_list[row_idx] = acc_black
            
            # Fill stage analysis results
            if stage_analysis is not None:
                # White
                white_beginning_acc[row_idx] = stage_analysis["beginning"]["white"]["accuracy"]
                white_beginning_std[row_idx] = stage_analysis["beginning"]["white"]["std"]
                white_middle_acc[row_idx] = stage_analysis["middle"]["white"]["accuracy"]
                white_middle_std[row_idx] = stage_analysis["middle"]["white"]["std"]
                white_endgame_acc[row_idx] = stage_analysis["endgame"]["white"]["accuracy"]
                white_endgame_std[row_idx] = stage_analysis["endgame"]["white"]["std"]
                
                # Black
                black_beginning_acc[row_idx] = stage_analysis["beginning"]["black"]["accuracy"]
                black_beginning_std[row_idx] = stage_analysis["beginning"]["black"]["std"]
                black_middle_acc[row_idx] = stage_analysis["middle"]["black"]["accuracy"]
                black_middle_std[row_idx] = stage_analysis["middle"]["black"]["std"]
                black_endgame_acc[row_idx] = stage_analysis["endgame"]["black"]["accuracy"]
                black_endgame_std[row_idx] = stage_analysis["endgame"]["black"]["std"]
            
            # Only count valid games
            if avg_cp_white is not None and acc_white is not None and avg_cp_black is not None and acc_black is not None:
                valid_games += 1
                total_avg_cp_white += avg_cp_white
                total_avg_cp_black += avg_cp_black
                total_acc_white += acc_white
                total_acc_black += acc_black
        
        # Add new columns to original DataFrame
        original_df['White CP Loss'] = white_cp_loss_list
        original_df['White Accuracy'] = white_accuracy_list
        original_df['Black CP Loss'] = black_cp_loss_list
        original_df['Black Accuracy'] = black_accuracy_list
        
        # Add stage analysis columns
        original_df['White Beginning Accuracy'] = white_beginning_acc
        original_df['White Beginning Std'] = white_beginning_std
        original_df['White Middle Accuracy'] = white_middle_acc
        original_df['White Middle Std'] = white_middle_std
        original_df['White Endgame Accuracy'] = white_endgame_acc
        original_df['White Endgame Std'] = white_endgame_std
        
        original_df['Black Beginning Accuracy'] = black_beginning_acc
        original_df['Black Beginning Std'] = black_beginning_std
        original_df['Black Middle Accuracy'] = black_middle_acc
        original_df['Black Middle Std'] = black_middle_std
        original_df['Black Endgame Accuracy'] = black_endgame_acc
        original_df['Black Endgame Std'] = black_endgame_std
        
        # Output statistics
        print(f"\nTotal games: {len(original_df)}")
        print(f"Successfully analyzed games: {valid_games}")
        print(f"Skipped/Failed games: {len(original_df) - valid_games}")
        
        summary = {}
        if valid_games > 0:
            print("\n===== Overall Statistics =====")
            print(f'Valid analyzed games: {valid_games}')
            print(f'White average: {total_avg_cp_white/valid_games:.1f} cp loss, {total_acc_white/valid_games:.1f}% accuracy')
            print(f'Black average: {total_avg_cp_black/valid_games:.1f} cp loss, {total_acc_black/valid_games:.1f}% accuracy')
            
            summary = {
                "total_games": valid_games,
                "avg_white_cp_loss": total_avg_cp_white/valid_games,
                "avg_white_accuracy": total_acc_white/valid_games,
                "avg_black_cp_loss": total_avg_cp_black/valid_games,
                "avg_black_accuracy": total_acc_black/valid_games
            }
        
        result_dict = {
            "game_results": game_details,  # Use detailed game info instead of simple results
            "summary": summary
        }
        
        return result_dict, original_df
    
    finally:
        engine.quit()


@task(name="simplify_termination_reasons")
def simplify_termination(term):
    if pd.isna(term) or term == '':
        return 'None'
    # WON
    if 'resignation' in term.lower():
        return 'resignation'
    elif 'checkmate' in term.lower():
        return 'checkmate'
    elif 'game abandoned' in term.lower():
        return 'game abandoned'
    elif 'time' in term.lower() and not ('insufficient' in term.lower() or 'stalemate' in term.lower()):
        return 'time'
    # DRAW
    elif 'repetition' in term.lower():
        return 'repetition'
    elif 'agreement' in term.lower():
        return 'agreement'
    elif 'insufficient material' in term.lower():
        return 'insufficient material'
    elif 'stalemate' in term.lower():
        return 'stalemate'
    elif '50-move rule' in term.lower():
        return '50-move rule'
    elif 'timeout vs insufficient material' in term.lower():
        return 'timeout vs insufficient material'
    else:
        return term


@task(name="save_results_to_csv")
def save_results(df, output_path):
    df.to_csv(output_path, index=False)
    return output_path


@task(name="prepare_data")
def prepare_data(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        delimiter = '\t' if '\t' in first_line else ','
    original_df = pd.read_csv(csv_file_path, delimiter=delimiter)
    return original_df

@task(name="process_termination_reasons")
def process_termination_reasons(df):
    """处理终止原因的独立任务"""
    df['Termination'] = df['Termination'].apply(simplify_termination)
    return df

@task(name="save_analysis_results")
def save_analysis_results(df, output_path):
    """保存分析结果的独立任务"""
    saved_path = save_results(df, output_path)
    return saved_path

@flow(name="chess_game_analysis_pipeline")
def analyze_chess_games_flow(
    csv_file_path,
    engine_path,
    threads=2,
    depth=16,
    is_verbose=True,
    output_path='analyzed_results.csv'
):
    """
    Main workflow for analyzing chess games
    """
    logger = get_run_logger()
    
    logger.info(f"Starting analysis of {csv_file_path}")
    logger.info(f"Using engine at {engine_path} with {threads} threads and depth {depth}")
    
    original_df = prepare_data(csv_file_path)
    logger.info(f"Loaded {len(original_df)} games")

    result_dict, df = analyze_csv(
        csv_file_path, 
        engine_path, 
        threads, 
        depth, 
        is_verbose
    )

    #处理终止原因
    df = process_termination_reasons(df)
    logger.info("Processed termination reasons")

    saved_path = save_analysis_results(df, output_path)
    logger.info(f"Analysis complete. Results saved to {saved_path}")

    return saved_path



#workflow2
@task(name="calculate_player_games")
def calculate_player_games(df):
    """统计每位棋手作为白方和黑方的对局数量"""
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
    """计算玩家胜场"""
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
    """计算玩家平局数量"""
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
    """计算玩家失败场次"""
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
    """计算玩家ELO分数"""
    white_elos = df[['White', 'WhiteElo']]
    black_elos = df[['Black', 'BlackElo']]
    
    white_elos.columns = ['user', 'elo']
    black_elos.columns = ['user', 'elo']
    
    df_elo = pd.concat([white_elos, black_elos])
    df_elo = df_elo.groupby('user').mean().reset_index()
    
    return df_elo

@task(name="calculate_player_accuracy")
def calculate_player_accuracy(df):
    """计算玩家准确率"""
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
def comprehensive_player_analysis(df):
    """综合分析玩家数据"""
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
    df_final['opening_distance'] = df_final['accuracy_opening'] - df_final['accuracy']
    df_final['middlegame_distance'] = df_final['accuracy_middlegame'] - df_final['accuracy']
    df_final['endgame_distance'] = df_final['accuracy_endgame'] - df_final['accuracy']
    
    df_final['opening_distance_alt'] = df_final['accuracy_opening'] - df_final['accuracy_opening'].mean()
    df_final['middlegame_distance_alt'] = df_final['accuracy_middlegame'] - df_final['accuracy_middlegame'].mean()
    df_final['endgame_distance_alt'] = df_final['accuracy_endgame'] - df_final['accuracy_endgame'].mean()
    
    df_final.to_csv('df_final.csv', index=False)
    
    return df_final


#main
def deploy_analysis():
    if sys.platform.startswith('win'):
        input_csv_file_path = 'data\\1410-15.csv'
        engine_path = 'C:\\Users\\Administrator\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe'
    else: 
        engine_path = '/opt/homebrew/Cellar/stockfish/17/bin/stockfish'
        input_csv_file_path = 'data/1410-15.csv'

    output_path = 'data/result.csv'
    
    # 运行棋局分析流程
    result = analyze_chess_games_flow(
        csv_file_path=input_csv_file_path,
        engine_path=engine_path,
        threads=2,
        depth=16,
        is_verbose=True,
        output_path=output_path
    )
    return result

def player_analysis():
    #need to change
    df = pd.read_csv("C:\\Users\\Administrator\\Desktop\\simple eda\\simple eda\\EDA\\prefectlearning\\data\\output\\1410-15result.csv")
    result = comprehensive_player_analysis(df)
    print("玩家分析完成，结果已保存到player_final.csv")
    return result

if __name__ == "__main__":
    result = deploy_analysis()
    print(f"棋局分析完成，结果保存至: {result}")
    player_analysis()


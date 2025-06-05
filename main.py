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
from prefect.cache_policies import NO_CACHE
from prefect import flow, task, get_run_logger, serve
import numpy as np
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


#workflow1
def get_eval_str(score, board):
    # score is mate
    if isinstance(score, chess.engine.Mate):
        mate_value = score.mate()
        if (mate_value > 0 and board.turn == chess.WHITE) or (mate_value < 0 and board.turn == chess.BLACK):
            mating_side = "White"
        else:
            mating_side = "Black"
        return f"Mate in {abs(mate_value)} for {mating_side}"
    else:
        # score is centipawn
        cp_score = score.score()
        return f"{cp_score/100.0:.2f}"


def move_accuracy_percent(before, after):
    if after >= before:
        return 100.0  # didnt get worse,think it's a perfect move
    else:
        win_diff = before - after
        raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) + -3.166924740191411
        return max(min(raw + 1, 100), 0)


def winning_chances_percent(cp):
    multiplier = -0.00368208
    chances = 2 / (1 + math.exp(multiplier * cp)) - 1
    return 50 + 50 * max(min(chances, 1), -1)


def harmonic_mean(values):
    n = len(values)
    if n == 0:
        return 0
    reciprocal_sum = sum(1 / x for x in values if x)
    return n / reciprocal_sum if reciprocal_sum else 0


def std_dev(seq):
    if len(seq) == 0:
        return 0.5
    mean = sum(seq) / len(seq)
    variance = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(variance)


def volatility_weighted_mean(accuracies, win_chances, is_white):
    weights = []
    for i in range(len(accuracies)):
        base_index = i * 2 + 1 if is_white else i * 2 + 2
        start_idx = max(base_index - 2, 0)
        end_idx = min(base_index + 2, len(win_chances) - 1)

        sub_seq = win_chances[start_idx:end_idx+1]
        weight = max(min(std_dev(sub_seq), 12), 0.5)
        weights.append(weight)

    weighted_sum = sum(a*w for a, w in zip(accuracies, weights))
    total_weight = sum(weights)
    weighted_mean = weighted_sum / total_weight if total_weight else 0

    return weighted_mean


class SimpleStockfishEngine:
    def __init__(self, engine_path, threads=1):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine path does not exist: {engine_path}")
        
        self.process = subprocess.Popen(
            str(self.engine_path),
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1
        )
        
        self._send_command("uci")
        self._wait_for("uciok")
        self._send_command(f"setoption name Threads value {threads}")
        self._send_command("isready")
        self._wait_for("readyok")
    
    def _send_command(self, command):
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()
    
    def _wait_for(self, text, timeout=5.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if text in line:
                return line
        raise TimeoutError(f"Timeout waiting for '{text}'")
    
    def _read_until_empty(self):
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                break
            lines.append(line)
        return lines
    
    def analyse(self, board, depth=16):
        fen = board.fen()
        self._send_command(f"position fen {fen}")
        self._send_command(f"go depth {depth}")
        
        score = None
        best_move = None
        
        while True:
            line = self.process.stdout.readline().strip()
            if "bestmove" in line:
                best_move = line.split()[1]
                break
            
            if "score" in line:
                parts = line.split()
                score_idx = parts.index("score")
                score_type = parts[score_idx + 1]
                score_value = int(parts[score_idx + 2])
                
                if score_type == "cp":
                    score = chess.engine.Cp(score_value)
                elif score_type == "mate":
                    score = chess.engine.Mate(score_value)
        
        if score is None:
            score = chess.engine.Cp(0)  
            
        return {"score": score, "pv": [best_move] if best_move else []}
    
    def quit(self):
        self._send_command("quit")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


def get_score_value(score, board, mate_score=1000):
    # convert the stockfish score to numeric value,
    # and for mate, it's +/-10000
    if isinstance(score, chess.engine.Mate):
        mate = score.mate()
        value = mate_score if mate > 0 else -mate_score
        if mate > 0:
            value = mate_score - mate
        else:
            value = -mate_score - mate
    else:  # Cp
        value = score.score()
        
    if board.turn == chess.BLACK:
        value = -value
        
    return value

def process_csv(csv_file, engine, depth, is_verbose):
    all_games = []
    skipped_indices = []
    game_details = []  # Store detailed analysis for each game
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        # Read first line to determine delimiter
        first_line = file.readline().strip()
        file.seek(0)
        
        delimiter = '\t' if '\t' in first_line else ','
        csv_reader = csv.DictReader(file, delimiter=delimiter)
        rows = list(csv_reader)

        pbar = tqdm(total=len(rows), desc="Analyzing games")

        
        # Create progress bar with tqdm
        for row_idx, row in enumerate(rows):
            # Get basic information
            white_player = row.get('White', 'Unknown')
            black_player = row.get('Black', 'Unknown')
            white_elo = row.get('WhiteElo', 'N/A')
            black_elo = row.get('BlackElo', 'N/A')
            
            if is_verbose:
                print(f'\n—— Analyzing Game {row_idx+1} ——')
                print(f'White: {white_player} ({white_elo}), Black: {black_player} ({black_elo})')
            
            # Split moves
            moves_str = row.get('Moves', '')
            if not moves_str:
                all_games.append((row_idx, white_player, black_player, None, None, None, None, None))
                skipped_indices.append(row_idx)
                continue
            
            # Check move format
            moves_list = moves_str.split(',')
            has_invalid_moves = any(move_str.strip() and (len(move_str.strip()) < 4 or len(move_str.strip()) > 5) 
                                    for move_str in moves_list)
            
            if has_invalid_moves:
                all_games.append((row_idx, white_player, black_player, None, None, None, None, None))
                skipped_indices.append(row_idx)
                continue
                
            # Initialize parameters
            game_acc_white, game_acc_black = [], []
            game_cp_white, game_cp_black = 0, 0
            game_win_chances = []
            
            # Store detailed information for each move
            move_details = []
            
            board = chess.Board()
            move_number = 1
            analysis_success = True
            
            try:
                for move_idx, move_str in enumerate(moves_list):
                    try:
                        move_str = move_str.strip()
                        if not move_str:
                            continue
                        
                        # Try to parse the move
                        try:
                            move = chess.Move.from_uci(move_str)
                            if move not in board.legal_moves:
                                analysis_success = False
                                break
                        except chess.InvalidMoveError:
                            analysis_success = False
                            break
                            
                        san_move = board.san(move)
                        
                        # Evaluate position before and after the move
                        result_before = engine.analyse(board, depth)
                        score_before = get_score_value(result_before["score"], board)
                        board.push(move)
                        result_after = engine.analyse(board, depth)
                        score_after = get_score_value(result_after["score"], board)
                        
                        # Convert to win percentage
                        win_before_white = winning_chances_percent(score_before)
                        win_after_white = winning_chances_percent(score_after)
                        game_win_chances.append(win_after_white)
                        
                        if board.turn == chess.WHITE:  # Black just moved
                            win_before = 100 - win_before_white
                            win_after = 100 - win_after_white
                        else:  # White just moved
                            win_before = win_before_white
                            win_after = win_after_white
                            
                        accuracy = move_accuracy_percent(win_before, win_after)
                        
                        # Calculate cp loss
                        if board.turn == chess.BLACK:  # White just moved
                            cp_loss = 0 if score_after > score_before else score_before - score_after
                            game_cp_white += cp_loss
                            game_acc_white.append(accuracy)
                            player = "White"
                        else:  # Black just moved
                            cp_loss = 0 if score_after < score_before else score_after - score_before
                            game_cp_black += cp_loss
                            game_acc_black.append(accuracy)
                            player = "Black"
                            
                        # Store move analysis details
                        step_info = {
                            "move_number": (move_idx // 2) + 1,
                            "san_move": san_move,
                            "player": player,
                            "evaluation": get_eval_str(result_after["score"], board),
                            "cp_loss": cp_loss,
                            "accuracy": accuracy,
                            "win_percentage": win_after_white
                        }
                        move_details.append(step_info)
                        
                        if board.turn == chess.WHITE:
                            move_number += 1
                            
                    except Exception as e:
                        if is_verbose:
                            print(f"Game {row_idx+1}: Error processing move {move_str}: {e}")
                        analysis_success = False
                        break
                
                # Calculate metrics for each game
                if analysis_success and (game_acc_white or game_acc_black):
                    # Basic metrics calculation
                    avg_cp_white = game_cp_white / len(game_acc_white) if game_acc_white else None
                    avg_cp_black = game_cp_black / len(game_acc_black) if game_acc_black else None
                    
                    harmonic_white = harmonic_mean(game_acc_white) if game_acc_white else None
                    weighted_white = volatility_weighted_mean(game_acc_white, game_win_chances, True) if game_acc_white else None
                    final_acc_white = (harmonic_white + weighted_white) / 2 if harmonic_white is not None and weighted_white is not None else None
                    
                    harmonic_black = harmonic_mean(game_acc_black) if game_acc_black else None
                    weighted_black = volatility_weighted_mean(game_acc_black, game_win_chances, False) if game_acc_black else None
                    final_acc_black = (harmonic_black + weighted_black) / 2 if harmonic_black is not None and weighted_black is not None else None
                    
                    # Stage analysis - divide moves into 3 phases
                    white_moves = [(i, acc) for i, acc in enumerate(game_acc_white)]
                    black_moves = [(i, acc) for i, acc in enumerate(game_acc_black)]
                    
                    # Stage analysis results
                    stage_analysis = {
                        "beginning": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}},
                        "middle": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}},
                        "endgame": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}}
                    }
                    
                    # Process white moves by stage
                    if white_moves:
                        n_moves = len(white_moves)
                        begin_idx = n_moves // 3
                        mid_idx = 2 * n_moves // 3
                        
                        begin_moves = [move[1] for move in white_moves[:begin_idx]]
                        mid_moves = [move[1] for move in white_moves[begin_idx:mid_idx]]
                        end_moves = [move[1] for move in white_moves[mid_idx:]]
                        
                        if begin_moves:
                            stage_analysis["beginning"]["white"]["accuracy"] = sum(begin_moves) / len(begin_moves)
                            stage_analysis["beginning"]["white"]["std"] = np.std(begin_moves) if len(begin_moves) > 1 else 0
                            
                        if mid_moves:
                            stage_analysis["middle"]["white"]["accuracy"] = sum(mid_moves) / len(mid_moves)
                            stage_analysis["middle"]["white"]["std"] = np.std(mid_moves) if len(mid_moves) > 1 else 0
                            
                        if end_moves:
                            stage_analysis["endgame"]["white"]["accuracy"] = sum(end_moves) / len(end_moves)
                            stage_analysis["endgame"]["white"]["std"] = np.std(end_moves) if len(end_moves) > 1 else 0
                    
                    # Process black moves by stage
                    if black_moves:
                        n_moves = len(black_moves)
                        begin_idx = n_moves // 3
                        mid_idx = 2 * n_moves // 3
                        
                        begin_moves = [move[1] for move in black_moves[:begin_idx]]
                        mid_moves = [move[1] for move in black_moves[begin_idx:mid_idx]]
                        end_moves = [move[1] for move in black_moves[mid_idx:]]
                        
                        if begin_moves:
                            stage_analysis["beginning"]["black"]["accuracy"] = sum(begin_moves) / len(begin_moves)
                            stage_analysis["beginning"]["black"]["std"] = np.std(begin_moves) if len(begin_moves) > 1 else 0
                            
                        if mid_moves:
                            stage_analysis["middle"]["black"]["accuracy"] = sum(mid_moves) / len(mid_moves)
                            stage_analysis["middle"]["black"]["std"] = np.std(mid_moves) if len(mid_moves) > 1 else 0
                            
                        if end_moves:
                            stage_analysis["endgame"]["black"]["accuracy"] = sum(end_moves) / len(end_moves)
                            stage_analysis["endgame"]["black"]["std"] = np.std(end_moves) if len(end_moves) > 1 else 0
                    
                    all_games.append((row_idx, white_player, black_player, avg_cp_white, final_acc_white, avg_cp_black, final_acc_black, stage_analysis))
                    game_details.append({
                        "row_idx": row_idx,
                        "white_player": white_player,
                        "black_player": black_player,
                        "moves": move_details,
                        "summary": {
                            "white": {"cp_loss": avg_cp_white, "accuracy": final_acc_white},
                            "black": {"cp_loss": avg_cp_black, "accuracy": final_acc_black}
                        },
                        "stage_analysis": stage_analysis
                    })
                else:
                    all_games.append((row_idx, white_player, black_player, None, None, None, None, None))
                    skipped_indices.append(row_idx)
            
            except Exception as e:
                if is_verbose:
                    print(f"Game {row_idx+1}: Error analyzing game: {e}")
                    traceback.print_exc()
                all_games.append((row_idx, white_player, black_player, None, None, None, None, None))
                skipped_indices.append(row_idx)
            
            pbar.update(1)
        pbar.close()
    
    if skipped_indices and is_verbose:
        print(f"\nSkipped/Failed games: {len(skipped_indices)}")
    
    return all_games, game_details


@task(name="analyze_chess_games")
def analyze_csv(csv_file_path, engine_path, threads, depth, is_verbose=False):
    """
    Analyze chess games in CSV file, calculate accuracy and CP loss for each game,
    and add results to the original DataFrame
    """
    # Read CSV to DataFrame
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        delimiter = '\t' if '\t' in first_line else ','
    original_df = pd.read_csv(csv_file_path, delimiter=delimiter)
    
    engine = SimpleStockfishEngine(engine_path, threads)
    
    try:
        all_games, game_details = process_csv(csv_file_path, engine, depth, is_verbose)
        
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
    #need to change
    csv_file_path = "C:\\Users\\Administrator\\Desktop\\2014.10-15.2\\results\\aftermerge\\1410-15.csv"
    #need to change
    engine_path = 'C:\\Users\\Administrator\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe'
    output_path = 'result.csv'
    
    # 运行棋局分析流程
    result = analyze_chess_games_flow(
        csv_file_path=csv_file_path,
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


#代码完全复制games_table8.py  只是想应用到自己的测试数据集上看结果对不对
#environment setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
import math
import numpy as np
import asyncio
from tqdm import tqdm
import tqdm
from pathlib import Path
from stockfish import Stockfish  
from mpi4py import MPI
#强制更换 event loop（解决部分 Windows 异步问题）
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from prefect.tasks import task_input_hash
from prefect.cache_policies import NO_CACHE
from prefect import flow, task, get_run_logger, serve
from typing import Dict, List, Tuple, Any, Optional

def get_eval_str(score, board):
    """_summary_
    Convert the chess evaluation into a readable string for people.

    Args:
        score (chess.engine.Mate & chess.engine.PovScore): the Evaluation
              if mate evaluation: contains the move numbers to checkmate
              if score evaluation: centipawn advantage/disadvantage
        board (chess.Board): the current board state & 
                             board.turn indicates who will move next True for white,False for black

    Returns:
        str(the formatted evaluation result):
             if Mate: a integer string
             if Score: a float string with 2 demicals
             positive for white has advantange
    """    
    #score is mate
    if isinstance(score, chess.engine.Mate):
        mate_value = score.mate()
        return str(mate_value if board.turn==chess.WHITE else -mate_value) #make sure the result always for white perspective
    else:
        #score is centipawn
        cp_score = score.score()
        return f"{cp_score/100.0:.2f}" if board.turn==chess.WHITE else f"{-cp_score/100.0:.2f}" 
    
def move_accuracy_percent(before, after):
    """_summary_
    Calcute move accuracy percentage basedon the evaluation-change

    Args:
        before (_type_): pre-move evaluation change
        after (_type_): post-move evaluation change

    Returns:
        float: accuracy percentage (0.0-100.0)
    """    
    if after >= before:
        return 100.0 #didnt get worse,think it's a perfect move
    else:
        win_diff = before - after
        raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) + -3.166924740191411
        return max(min(raw + 1, 100), 0)

def winning_chances_percent(cp):
    """_summary_
    convert centipawns into win probability percentage

    Args:
        cp (int): engine raw score in centipawns

    Returns:
        float: win probability percentage(0.0--100.0)
    """    
    multiplier = -0.00368208
    chances = 2 / (1 + math.exp(multiplier * cp)) - 1
    return 50 + 50 * max(min(chances, 1), -1)

def harmonic_mean(values):
    """_summary_
    calculate harmonic mean of a sequence

    Args:
        values (_type_): iterable of numerical numbers

    Returns:
        float: harmonic mean
    """    
    n = len(values)
    if n == 0:
        return 0
    reciprocal_sum = sum(1 / x for x in values if x)
    return n / reciprocal_sum if reciprocal_sum else 0

def std_dev(seq):
    """_summary_
    calculate standard deviation with special empty sequence handling(population standard deviation formula)
    0.5 for empty sequences

    Args:
        seq (int/float): numerical sequence

    Returns:
        float: standard deviation
    """    
    if len(seq) == 0:
        return 0.5 
    mean = sum(seq) / len(seq)
    variance = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(variance)

def volatility_weighted_mean(accuracies, win_chances, is_white):
    """_summary_

    Args:
        accuracies (list): list of accuracy percentages
        win_chances (list): list of win probability perentages
        is_white (bool): indicate white's move

    Returns:
        float: weighted mean accuracy
    """    
    weights = [] #list to put each move's weight
    for i in range(len(accuracies)):
        base_index = i * 2 + 1 if is_white else i * 2 + 2
        start_idx = max(base_index - 2, 0)
        end_idx = min(base_index + 2, len(win_chances) - 1)

        sub_seq = win_chances[start_idx:end_idx+1]
        weight = max(min(std_dev(sub_seq), 12), 0.5)
        weights.append(weight)
    
    #Weight calculation: 给胜率波动大的步子(棋局转折点）更高的权重
    #    1. Determine base index: odd for white, even for black
    #    2. Create 5-point window centered at base index
    #    3. Window std_dev → weight = clamp(std_dev, 0.5, 12)

    weighted_sum = sum(a*w for a,w in zip(accuracies,weights))
    total_weight = sum(weights)
    weighted_mean = weighted_sum / total_weight if total_weight else 0

    return weighted_mean

class SimpleStockfishEngine:
    """_summary_ 
    a python interface for stockfish engine,provide basic board analysis and evaluation
    engine initialization and protocol handshake
    deep position analysis
    position evaluation
    process management
    """     
    def __init__(self, engine_path, threads=1):
        """_summary_
        initialize and launch Stockfish engine process
        perform uci handshake, set computation threads

        Args:
            engine_path: path to stockfish
            threads (int, optional): number of cpu threads to use Defaults to 1.

        Raises:
            FileNotFoundError: engine path is invalid
            timeouterror: protocol handshake times out
        """        
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"引擎路径不存在: {engine_path}")
        
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
        raise TimeoutError(f"等待'{text}'超时")
    
    def _read_until_empty(self):
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                break
            lines.append(line)
        return lines
    
    def analyse(self, board, depth=16):
        """_summary_
        analyze current board position and return dict with position score and best moves

        Args:
            board (_type_): chess.Board represents current position
            depth (int, optional): moves to look ahead. Defaults to 16.

        Returns:
            'score': postion score(CP/Mate)
            'pv': list of principal variation moves 
        """        
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
    
    def set_fen_position(self,fen):
        self._send_command(f"position fen {fen}")

    def get_evaluation(self, depth=10):
        """_summary_
        quick evaluation of current position

        Args:
            depth (int, optional): calculation depth. Defaults to 10.

        Returns:
            {"type": Score type ("cp" or "mate"), "value": Numeric value}
        """        
        self._send_command(f"go depth {depth}")

        score_type = None
        score_value = None

        while True:
            line = self.process.stdout.readline().strip()
            if "bestmove" in line:
                break
            if "score" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    score_type = parts[score_idx + 1]
                    score_value = int(parts[score_idx + 2])
                except (ValueError, IndexError):
                    continue  # 跳过无效行

        if score_type is None or score_value is None:
            return {"type": "cp", "value": 0}

        return {"type": score_type, "value": score_value}
    
    def quit(self):
        """_summary_
        terminate engine process; force kill when timeout 
        """        
        self._send_command("quit")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

def get_score_value(score, board, mate_score=1000):
    #convert the stockfish score to numeric value,
    #adjust sign based on current player's turn
    """_summary_

    Args:
        score (cp/mate): get from stockfish
        board (chess.Board): current state
        mate_score (int, optional): Defaults to 1000.

    Returns:
        int: a numeric value
    """    
    if isinstance(score, chess.engine.Mate):
        mate = score.mate()
        value = mate_score if mate > 0 else -mate_score
        if mate > 0: #white will checkmate black
            value = mate_score - mate
        else:  #black will checkmate white
            value = -mate_score - mate
    else:  # Cp
        value = score.score()
        
    if board.turn == chess.BLACK:
        value = -value
        
    return value

@task(name='Process CSV Chunk',retries=0, log_prints=True)
def process_chunk(chunk_df, start_idx, engine_path, threads, depth, is_verbose, temp_csv_path):
    """_summary_     a function for processing dataframe chunks
    save data chunk to tempory csv   clean up temporary files and resources
  
    Args:
        chunk_df (pd.Dataframe): data chunk to process
        start_idx (int): global start index
        engine_path (str): path to stockfish
        threads (int): number of threads
        depth (int): analysis depth
        is_verbose (bool): verbose flag
        temp_csv_path (str): temporary csv file path

    Returns:
        tuple: analysis rersult:(all_games,game_details,game_evaluations)
    """
    # get current mpi process rank & save data chunk into temporary csv file
    rank = MPI.COMM_WORLD.Get_rank()
    chunk_df.to_csv(temp_csv_path, index=False)
    
    engine = SimpleStockfishEngine(engine_path, threads)
    try:
        # invoke csv functio with offset
        all_games, game_details, game_evaluations = process_csv_with_offset(temp_csv_path, engine, depth, is_verbose, start_idx)
        print(f"[Rank {rank}] 本地处理完毕，games: {len(all_games)}, details: {len(game_details)}")
        return all_games, game_details, game_evaluations
    except Exception as e:
        print(f"[Rank {rank}] ❌ 分析失败: {e}")
        return [], [], []
    finally:
        engine.quit()
        #release resource4s clean up
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

def process_csv_with_offset(csv_file, engine, depth, is_verbose, start_idx=0):
    """_summary_
    function with global index offset 
    read csv ; process games by row ;compute all the index

    Args:
        csv_file (str): file path
        engine: stockfish engine instance
        depth (int) 
        is_verbose (bool): verbose flag
        start_idx (int, optional): glovbal row index offset. Defaults to 0.

    Returns:
        tuple: analysis rersult:(all_games,game_details,game_evaluations)
    """    
    all_games = []
    skipped_indices = []
    game_details = []
    game_evaluations = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        first_line = file.readline().strip()
        file.seek(0)
        
        delimiter = '\t' if '\t' in first_line else ','
        csv_reader = csv.DictReader(file, delimiter=delimiter)
        rows = list(csv_reader)

        # 创建主进度条
        pbar = tqdm.tqdm(total=len(rows), 
                    desc="总体进度", 
                    disable=not is_verbose)
        
        #iterate each row
        for local_idx, row in enumerate(rows):
            #calculate global index
            global_idx = start_idx + local_idx
            
            #basic info
            white_player = row.get('White', 'Unknown')
            black_player = row.get('Black', 'Unknown')
            white_elo = row.get('WhiteElo', 'N/A')
            black_elo = row.get('BlackElo', 'N/A')
            
            if is_verbose:
                pbar.set_description(f"Game {global_idx+1}: {white_player} vs {black_player}")
            
            #get move sequence str
            moves_str = row.get('Moves', '')
            #move sequence is empty
            if not moves_str:
                all_games.append((global_idx, white_player, black_player, None, None, None, None, None, None))
                skipped_indices.append(global_idx)
                game_evaluations.append(None)
                pbar.update(1)
                continue
            
            #check if legal
            moves_list = moves_str.split(',')
            has_invalid_moves = any(move_str.strip() and (len(move_str.strip()) < 4 or len(move_str.strip()) > 5) 
                                     for move_str in moves_list)
            #skip illegal
            if has_invalid_moves:
                all_games.append((global_idx, white_player, black_player, None, None, None, None, None, None))
                skipped_indices.append(global_idx)
                game_evaluations.append(None)
                pbar.update(1)
                continue
                
            #initial
            game_acc_white, game_acc_black = [], []
            game_cp_white, game_cp_black = 0, 0
            game_win_chances = []
            evaluation_list = []  
            
            #detailed information
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
                        
                        #parse move
                        try:
                            move = chess.Move.from_uci(move_str)
                            if move not in board.legal_moves:
                                analysis_success = False
                                break
                        except chess.InvalidMoveError:
                            analysis_success = False
                            break
                            
                        san_move = board.san(move)
                        
                        #pre-move analysis
                        result_before = engine.analyse(board, depth)
                        score_before = get_score_value(result_before["score"], board)
                        #
                        current_eval = get_eval_str(result_before["score"], board)
                        evaluation_list.append(current_eval)

                        board.push(move)
                        result_after = engine.analyse(board, depth)
                        score_after = get_score_value(result_after["score"], board)
                        
                        #calculate win probability
                        win_before_white = winning_chances_percent(score_before)
                        win_after_white = winning_chances_percent(score_after)
                        game_win_chances.append(win_after_white)
                        
                        if board.turn == chess.WHITE:  # 黑方刚走完
                            win_before = 100 - win_before_white
                            win_after = 100 - win_after_white
                        else:  # 白方刚走完
                            win_before = win_before_white
                            win_after = win_after_white
                            
                        accuracy = move_accuracy_percent(win_before, win_after)
                        
                        #calculate CPL Acc
                        if board.turn == chess.BLACK:  # 白方刚走完
                            cp_loss = 0 if score_after > score_before else score_before - score_after
                            game_cp_white += cp_loss
                            game_acc_white.append(accuracy)
                            player = "White"
                        else:  # 黑方刚走完
                            cp_loss = 0 if score_after < score_before else score_after - score_before
                            game_cp_black += cp_loss
                            game_acc_black.append(accuracy)
                            player = "Black"
                            
                        #store move details
                        step_info = {
                            "move_number": (move_idx // 2) + 1, #full move number
                            "san_move": san_move,
                            "player": player,
                            "evaluation": get_eval_str(result_after["score"], board),
                            "cp_loss": cp_loss,
                            "accuracy": accuracy,
                            "win_percentage": win_after_white
                        }
                        move_details.append(step_info)
                        #update move counter
                        if board.turn == chess.WHITE:
                            move_number += 1
                            
                    except Exception as e:
                        if is_verbose:
                            pbar.write(f"Game{global_idx+1}: Error processing move {move_str}: {str(e)[:50]}")
                        analysis_success = False
                        break
                
                #whole game statistics calculate
                if analysis_success and (game_acc_white or game_acc_black):
                    #basic
                    #cpl
                    avg_cp_white = game_cp_white / len(game_acc_white) if game_acc_white else None
                    avg_cp_black = game_cp_black / len(game_acc_black) if game_acc_black else None
                    #acc (harmonic + wolatility weighted)
                    harmonic_white = harmonic_mean(game_acc_white) if game_acc_white else None
                    weighted_white = volatility_weighted_mean(game_acc_white, game_win_chances, True) if game_acc_white else None
                    final_acc_white = (harmonic_white + weighted_white) / 2 if harmonic_white is not None and weighted_white is not None else None
                    harmonic_black = harmonic_mean(game_acc_black) if game_acc_black else None
                    weighted_black = volatility_weighted_mean(game_acc_black, game_win_chances, False) if game_acc_black else None
                    final_acc_black = (harmonic_black + weighted_black) / 2 if harmonic_black is not None and weighted_black is not None else None
                    
                    #phase analysis 
                    white_moves = [(i, acc) for i, acc in enumerate(game_acc_white)]
                    black_moves = [(i, acc) for i, acc in enumerate(game_acc_black)]
                    
                    #initialize stage analysis
                    stage_analysis = {
                        "beginning": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}},
                        "middle": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}},
                        "endgame": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}}
                    }
                    
                    #white phase analysis
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
                    
                    #black phase analysis
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

                    #add complete game records
                    #generate dataframe from this
                    all_games.append((global_idx, white_player, black_player, avg_cp_white, final_acc_white, 
                                      avg_cp_black, final_acc_black, stage_analysis,
                                      ','.join(evaluation_list) ))
                    #evaluations
                    game_evaluations.append(','.join(evaluation_list))  #new
                    #more detailed(include move details) but bigger
                    game_details.append({
                        "row_idx": global_idx,
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
                    all_games.append((global_idx, white_player, black_player, None, None, None, None, None, None))
                    skipped_indices.append(global_idx)
            
            except Exception as e:
                if is_verbose:
                    pbar.write(f"Game {global_idx+1}: Critical error: {str(e)[:100]}")
                    traceback.print_exc()
                all_games.append((global_idx, white_player, black_player, None, None, None, None, None, None))
                skipped_indices.append(global_idx)
            
            # 更新总进度条
            pbar.update(1)
            
        pbar.close()
    
    if skipped_indices and is_verbose:
        print(f"\n跳过/分析失败的棋局数: {len(skipped_indices)}")
    
    return all_games, game_details, game_evaluations

@task(name="Distribute Data")
def distribute_data(csv_file_path: str) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, int]]]:
    """read the csv file and distribute the data to different ranks
    
    Args:
        csv_file_path(str): path
        
    Returns:
        Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, int]]]:
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"[Rank {rank}] 读取CSV文件: {csv_file_path}")
        original_df = pd.read_csv(csv_file_path)
        print(f"[Rank {rank}] 读取了 {len(original_df)} 行数据")
        
        #calculate how to split
        total_rows = len(original_df)
        rows_per_process = total_rows // size
        remainder = total_rows % size
        
        # rank0 process first data chunk
        start_idx = 0
        end_idx = rows_per_process + (1 if remainder > 0 else 0)
        local_df = original_df.iloc[start_idx:end_idx].copy()
        
        #分发数据块列表
        data_chunks = [(local_df, start_idx)]
        
        #distribute data chunk to different ranks
        for i in range(1, size):
            start = i * rows_per_process + min(i, remainder)
            end = start + rows_per_process + (1 if i < remainder else 0)
            if start < total_rows and start < end:
                work_df = original_df.iloc[start:end].copy()
                data_chunks.append((work_df, start))
                comm.send((work_df, start), dest=i, tag=100)
            else:
                data_chunks.append((pd.DataFrame(), -1))
                comm.send((pd.DataFrame(), -1), dest=i, tag=100)
        
        return original_df, data_chunks
    else:
        return None, None

@task(name="Collect Results")
def collect_results(original_df: pd.DataFrame, all_games_gathered: List, game_details_gathered: List, 
                   game_evaluations_gathered: List) -> Tuple[Dict, pd.DataFrame]:
    """gather all ranks' analysis results
    
    Args:
        original_df
        all_games_gathered
        game_details_gathered
        game_evaluations_gathered
        
    Returns:
        Tuple[Dict, pd.DataFrame]
    """
    rank = MPI.COMM_WORLD.Get_rank()
    
    print(f"[Rank {rank}] 收集到 {len(all_games_gathered)} 个进程的结果")
    for i, games in enumerate(all_games_gathered):
        print(f"[Rank {rank}] 进程 {i} 返回了 {len(games)} 个游戏结果")
        
    combined_games = []
    for sublist in all_games_gathered:
        combined_games.extend(sublist)
    
    combined_details = []
    for sublist in game_details_gathered:
        combined_details.extend(sublist)
        
    combined_evaluations = []
    for sublist in game_evaluations_gathered:
        combined_evaluations.extend(sublist)
    
    print(f"[Rank {rank}] 合并后的游戏总数: {len(combined_games)}")
    
    #create an empty list that matches the length of the original DataFrame
    white_cp_loss_list = [None] * len(original_df)
    white_accuracy_list = [None] * len(original_df)
    black_cp_loss_list = [None] * len(original_df)
    black_accuracy_list = [None] * len(original_df)
    evaluation_list = [None] * len(original_df)
    
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
    
    valid_games = 0
    total_avg_cp_white = total_avg_cp_black = 0.0
    total_acc_white = total_acc_black = 0.0
    
    #put results into lists
    for game in combined_games:
        row_idx, white, black, avg_cp_white, acc_white, avg_cp_black, acc_black, stage_analysis, evaluation = game
        if row_idx < 0 or row_idx >= len(original_df):
            print(f"[Rank {rank}] 警告: 无效的行索引 {row_idx}，跳过")
            continue
        #填充结果
        white_cp_loss_list[row_idx] = avg_cp_white
        white_accuracy_list[row_idx] = acc_white
        black_cp_loss_list[row_idx] = avg_cp_black
        black_accuracy_list[row_idx] = acc_black
        evaluation_list[row_idx] = evaluation
        
        #填充阶段分析结果
        if stage_analysis is not None:
            white_beginning_acc[row_idx] = stage_analysis["beginning"]["white"]["accuracy"]
            white_beginning_std[row_idx] = stage_analysis["beginning"]["white"]["std"]
            white_middle_acc[row_idx] = stage_analysis["middle"]["white"]["accuracy"]
            white_middle_std[row_idx] = stage_analysis["middle"]["white"]["std"]
            white_endgame_acc[row_idx] = stage_analysis["endgame"]["white"]["accuracy"]
            white_endgame_std[row_idx] = stage_analysis["endgame"]["white"]["std"]
            
            black_beginning_acc[row_idx] = stage_analysis["beginning"]["black"]["accuracy"]
            black_beginning_std[row_idx] = stage_analysis["beginning"]["black"]["std"]
            black_middle_acc[row_idx] = stage_analysis["middle"]["black"]["accuracy"]
            black_middle_std[row_idx] = stage_analysis["middle"]["black"]["std"]
            black_endgame_acc[row_idx] = stage_analysis["endgame"]["black"]["accuracy"]
            black_endgame_std[row_idx] = stage_analysis["endgame"]["black"]["std"]
        
        #只统计有效对局
        if avg_cp_white is not None and acc_white is not None and avg_cp_black is not None and acc_black is not None:
            valid_games += 1
            total_avg_cp_white += avg_cp_white
            total_avg_cp_black += avg_cp_black
            total_acc_white += acc_white
            total_acc_black += acc_black
    
    #add new ccolumns
    original_df['White CP Loss'] = white_cp_loss_list
    original_df['White Accuracy'] = white_accuracy_list
    original_df['Black CP Loss'] = black_cp_loss_list
    original_df['Black Accuracy'] = black_accuracy_list
    original_df['Evaluation'] = evaluation_list
    
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
    
    # 输出统计信息
    print(f"\n总棋局数: {len(original_df)}")
    print(f"成功分析棋局数: {valid_games}")
    print(f"跳过/分析失败棋局数: {len(original_df) - valid_games}")
    
    summary = {}
    if valid_games > 0:
        print("\n===== 总体统计 =====")
        print(f'有效分析对局数: {valid_games}')
        print(f'白方平均: {total_avg_cp_white/valid_games:.1f}cp损失, {total_acc_white/valid_games:.1f}%准确率')
        print(f'黑方平均: {total_avg_cp_black/valid_games:.1f}cp损失, {total_acc_black/valid_games:.1f}%准确率')
        
        summary = {
            "total_games": valid_games,
            "avg_white_cp_loss": total_avg_cp_white/valid_games,
            "avg_white_accuracy": total_acc_white/valid_games,
            "avg_black_cp_loss": total_avg_cp_black/valid_games,
            "avg_black_accuracy": total_acc_black/valid_games
        }
    
    result_dict = {
        "game_results": combined_details,
        "summary": summary
    }
    
    return result_dict, original_df

@task(name="Save Results",log_prints=True)
def save_results(mediate_df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """save results into csv file
    
    Args:
        mediate_df
        output_dir
        filename
        
    Returns:
        str
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    mediate_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")
    return output_path

@flow(name="Chess Analysis Pipeline",version='1.0')
def analyze_chess_games(
    csv_file_path: str, 
    engine_path: str, 
    threads: int = 1, 
    depth: int = 16, 
    is_verbose: bool = False,
    output_dir: str = "./output",
    output_filename: str = "analysis_results.csv"
) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[str]]:
    """
    Main process for analyzing chess games in parallel using MPI.
    
    Args:
        csv_file_path: Path to the input CSV file.
    engine_path: Path to the Stockfish engine.
    threads: Number of threads used by each process.
    depth: Analysis depth.
    is_verbose: Whether to display detailed output.
    output_dir: Output directory; if None, the default path is used.
    output_filename: Name of the output file.
        
    Returns:
        Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[str]]: 
             A result dictionary, a DataFrame containing analysis results, and the output file path 
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f'[Rank{rank}]进程启动，总进程数{size}')
    
    # 设置默认输出目录
    if output_dir is None and rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../data/output')
    
    # 分发数据
    original_df, data_chunks = distribute_data(csv_file_path)
    
    # 处理本地数据块
    all_games, game_details, game_evaluations = [], [], []
    
    if rank == 0 and data_chunks:
        local_df, start_idx = data_chunks[0]
        if not local_df.empty:
            temp_csv_path = f"temp_rank_{rank}.csv"
            all_games, game_details, game_evaluations = process_chunk(
                local_df, start_idx, engine_path, threads, depth, is_verbose, temp_csv_path
            )
    elif rank > 0:
        # 非主进程接收数据
        local_df, start_idx = comm.recv(source=0, tag=100)
        if not local_df.empty:
            temp_csv_path = f"temp_rank_{rank}.csv"
            all_games, game_details, game_evaluations = process_chunk(
                local_df, start_idx, engine_path, threads, depth, is_verbose and rank == 0, temp_csv_path
            )
    
    # 收集结果
    all_games_gathered = comm.gather(all_games, root=0)
    game_details_gathered = comm.gather(game_details, root=0)
    game_evaluations_gathered = comm.gather(game_evaluations, root=0)
    
    # 主进程处理和保存结果
    if rank == 0 and all_games_gathered:
        result_dict, mediate_df = collect_results(
            original_df, all_games_gathered, game_details_gathered, game_evaluations_gathered
        )
        
        # 保存结果
        output_path = save_results(mediate_df, output_dir, output_filename)
        
        return result_dict, mediate_df, output_path
    else:
        return None, None, None

if __name__ == "__main__":
    """
    Coordinates the main entry point for distributed analysis of chess games from a CSV file,
    utilizing MPI for parallel processing. The analysis includes engine evaluations,
    accuracy calculations, and phase-based performance metrics.
    """    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "../data/input/1410-15.csv")
    engine_path = 'C:/Users/Administrator/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'
    threads = 1  
    depth = 16  
    is_verbose = True
    output_dir=os.path.join(script_dir,'../data/output')
    output_filename = "mpi1686.csv"
    
    #运行流程
    result_dict, mediate_df, output_path = analyze_chess_games(
        csv_file_path=csv_file_path,
        engine_path=engine_path,
        threads=threads,
        depth=depth,
        is_verbose=is_verbose,
        output_dir=output_dir,
        output_filename=output_filename
    )

#mpiexec -n 12 python "C:\Users\Administrator\Desktop\simple eda\simple eda\EDA\prefectlearning\models\games_table8.py"
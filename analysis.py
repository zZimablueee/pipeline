import chess.pgn
import chess.engine
import math
from pathlib import Path
import csv
import io
import traceback
import subprocess
import time
import numpy as np
from tqdm import tqdm


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


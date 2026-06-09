"""
평가 모듈.

evaluate_vs_random : AlphaZero 체크포인트 vs 랜덤봇. 흑/백 교대 측정.
arena_battle        : 두 세대 체크포인트를 N판 대전해 승률 비교.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.gomoku import GomokuEnv
from agents.alphazero.network import AlphaZeroNet
from agents.alphazero.mcts import MCTS
from agents.random_agent import RandomAgent


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────

def _load_net(ckpt_path: str, device: str = "cpu") -> tuple[AlphaZeroNet, int, int]:
    """체크포인트 → (네트워크, board_size, n_in_row)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    board_size = ckpt.get("board_size", 6)
    n_in_row = ckpt.get("n_in_row", 4)

    # 네트워크 구조는 저장된 state_dict의 shape에서 추론
    sd = ckpt["net_state"]
    # input_conv 첫 conv의 weight: (n_filters, 3, 3, 3)
    n_filters = sd["input_conv.0.weight"].shape[0]
    n_res_blocks = sum(1 for k in sd if k.startswith("res_blocks.") and k.endswith(".conv1.weight"))

    net = AlphaZeroNet(
        board_size=board_size,
        n_res_blocks=n_res_blocks,
        n_filters=n_filters,
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net, board_size, n_in_row


def _play_one(
    env: GomokuEnv,
    black_fn,   # callable(board, player) -> action (int)
    white_fn,
) -> int:
    """한 게임 진행. 반환: 1=BLACK 승 / -1=WHITE 승 / 0=무승부."""
    board = env.reset()
    player = env.current_player
    done = False

    while not done:
        if player == 1:  # BLACK
            action = black_fn(board, player)
        else:
            action = white_fn(board, player)
        board, reward, done, info = env.step(action)
        if not done:
            player = env.current_player

    if reward == 1:
        return env.current_player   # 마지막 착수자 = 승자
    return 0


def _mcts_fn(net: AlphaZeroNet, env: GomokuEnv, n_sim: int, device: str):
    """MCTS greedy 착수 함수를 반환하는 클로저."""
    def fn(board, player) -> int:
        mcts = MCTS(net, env, n_simulations=n_sim, device=device)
        pi = mcts.get_action_probs(board, player, temperature=1e-4)
        return int(np.argmax(pi))
    return fn


def _random_fn(env: GomokuEnv):
    agent = RandomAgent()
    def fn(board, player) -> int:
        return agent.select_action(board, env.legal_actions(board))
    return fn


# ──────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────

def evaluate_vs_random(
    ckpt_path: str,
    n_games: int = 100,
    n_simulations: int = 200,
    device: str = "cpu",
) -> dict:
    """
    AlphaZero 체크포인트 vs 랜덤봇.
    흑/백 각 n_games//2 판씩 교대 측정.

    반환:
      {
        "win_rate": float,          # 전체 승률
        "as_black": {"win", "loss", "draw"},
        "as_white": {"win", "loss", "draw"},
      }
    """
    net, board_size, n_in_row = _load_net(ckpt_path, device)
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    half = n_games // 2

    az = _mcts_fn(net, env, n_simulations, device)
    rnd = _random_fn(env)

    stats = {"as_black": {"win": 0, "loss": 0, "draw": 0},
             "as_white": {"win": 0, "loss": 0, "draw": 0}}

    # AlphaZero = BLACK
    for _ in range(half):
        result = _play_one(env, black_fn=az, white_fn=rnd)
        if result == 1:
            stats["as_black"]["win"] += 1
        elif result == -1:
            stats["as_black"]["loss"] += 1
        else:
            stats["as_black"]["draw"] += 1

    # AlphaZero = WHITE
    for _ in range(half):
        result = _play_one(env, black_fn=rnd, white_fn=az)
        if result == -1:
            stats["as_white"]["win"] += 1
        elif result == 1:
            stats["as_white"]["loss"] += 1
        else:
            stats["as_white"]["draw"] += 1

    total_wins = stats["as_black"]["win"] + stats["as_white"]["win"]
    win_rate = total_wins / n_games

    stats["win_rate"] = win_rate
    print(f"\n  vs 랜덤봇 평가 결과 ({n_games}판, 흑{half}/백{half})")
    print(f"  흑 기준: 승{stats['as_black']['win']} 패{stats['as_black']['loss']} 무{stats['as_black']['draw']}")
    print(f"  백 기준: 승{stats['as_white']['win']} 패{stats['as_white']['loss']} 무{stats['as_white']['draw']}")
    print(f"  전체 승률: {win_rate:.3f}")
    return stats


def arena_battle(
    ckpt_new: str,
    ckpt_old: str,
    n_games: int = 100,
    n_simulations: int = 200,
    device: str = "cpu",
) -> dict:
    """
    두 체크포인트 대전. 흑/백 각 n_games//2 판씩 교대.
    반환:
      {
        "new_win_rate": float,
        "old_win_rate": float,
        "draw_rate": float,
        "new_wins": int, "old_wins": int, "draws": int,
      }
    """
    net_new, board_size, n_in_row = _load_net(ckpt_new, device)
    net_old, _, _ = _load_net(ckpt_old, device)
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    half = n_games // 2

    fn_new = _mcts_fn(net_new, env, n_simulations, device)
    fn_old = _mcts_fn(net_old, env, n_simulations, device)

    new_wins = old_wins = draws = 0

    # NEW = BLACK
    for _ in range(half):
        result = _play_one(env, black_fn=fn_new, white_fn=fn_old)
        if result == 1:
            new_wins += 1
        elif result == -1:
            old_wins += 1
        else:
            draws += 1

    # NEW = WHITE
    for _ in range(half):
        result = _play_one(env, black_fn=fn_old, white_fn=fn_new)
        if result == -1:
            new_wins += 1
        elif result == 1:
            old_wins += 1
        else:
            draws += 1

    result = {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "new_win_rate": new_wins / n_games,
        "old_win_rate": old_wins / n_games,
        "draw_rate": draws / n_games,
    }
    print(f"\n  아레나 결과 (신규 vs 구세대, {n_games}판)")
    print(f"  신규 체크포인트: {os.path.basename(ckpt_new)}")
    print(f"  구세대 체크포인트: {os.path.basename(ckpt_old)}")
    print(f"  신규: {new_wins}승  구세대: {old_wins}승  무: {draws}")
    print(f"  신규 승률: {result['new_win_rate']:.3f}")
    return result

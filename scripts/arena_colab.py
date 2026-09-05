"""
scripts/arena_colab.py
세대 아레나 스크립트 — Google Colab 실행용.

Colab 셀:
  !cd /content/rl_omok && python scripts/arena_colab.py

옵션 (환경변수):
  N_SIM=200            탐색 횟수 (기본값)
  GAMES_PER_COLOR=25   흑/백 각 게임 수 (기본값)
"""

import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from env.gomoku import GomokuEnv
from agents.alphazero.network import AlphaZeroNet
from agents.alphazero.mcts import MCTS

# ── 설정 ──────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
_DRIVE          = "/content/drive/MyDrive/rl_omok"
CKPT_DIR        = os.path.join(_DRIVE, "checkpoints") if os.path.isdir("/content/drive") \
                  else os.path.join(os.path.dirname(__file__), "..", "checkpoints")
N_SIM           = int(os.environ.get("N_SIM", 200))
GAMES_PER_COLOR = int(os.environ.get("GAMES_PER_COLOR", 25))

# ── 평가 대상 세대 ─────────────────────────────────────────────────────────
ITERS = [10, 20, 30]   # 없는 파일은 자동으로 건너뜀


# ─────────────────────────────────────────────────────────────────────────

def _load_net(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["net_state"]
    board_size   = ckpt.get("board_size", 15)
    n_in_row     = ckpt.get("n_in_row", 5)
    n_filters    = sd["input_conv.0.weight"].shape[0]
    n_res_blocks = sum(1 for k in sd
                       if k.startswith("res_blocks.") and k.endswith(".conv1.weight"))
    net = AlphaZeroNet(
        board_size=board_size,
        n_res_blocks=n_res_blocks,
        n_filters=n_filters,
    ).to(DEVICE)
    net.load_state_dict(sd)
    net.eval()
    return net, board_size, n_in_row


def _mcts_fn(net, env):
    def fn(board, player) -> int:
        mcts = MCTS(net, env, n_simulations=N_SIM, device=DEVICE)
        pi = mcts.get_action_probs(board, player, temperature=1.0)
        return int(np.random.choice(len(pi), p=pi))
    return fn


def _play_one(env, black_fn, white_fn) -> int:
    """반환: 1=흑 승 / -1=백 승 / 0=무"""
    board = env.reset()
    player = env.current_player
    done = False
    while not done:
        action = black_fn(board, player) if player == 1 else white_fn(board, player)
        board, reward, done, info = env.step(action)
        if not done:
            player = env.current_player
    if reward == 1:
        return env.current_player
    if reward == -1:
        return -env.current_player  # 금수 착수자의 상대방 승
    return 0


def arena(label_a: str, ckpt_a: str, label_b: str, ckpt_b: str) -> dict:
    """
    A(신세대) vs B(구세대). 흑/백 각 GAMES_PER_COLOR판.
    A 기준 흑 승/패/무  +  백 승/패/무  반환.
    """
    net_a, board_size, n_in_row = _load_net(ckpt_a)
    net_b, _,          _        = _load_net(ckpt_b)
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)

    fn_a = _mcts_fn(net_a, env)
    fn_b = _mcts_fn(net_b, env)
    n    = GAMES_PER_COLOR
    total_games = n * 2

    # ── A = 흑 ──────────────────────────────────────────────────────────
    bw = bl = bd = 0
    print(f"\n  [{label_a} 흑] vs [{label_b} 백]  {n}판 진행 중...", flush=True)
    t0 = time.time()
    for i in range(n):
        r = _play_one(env, fn_a, fn_b)
        if   r ==  1: bw += 1
        elif r == -1: bl += 1
        else:         bd += 1
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (n - i - 1)
        print(f"  {i+1:2d}/{n}  흑 {bw}승/{bl}패  남은시간~{remaining/60:.1f}m", end="\r", flush=True)
    print()

    # ── A = 백 ──────────────────────────────────────────────────────────
    ww = wl = wd = 0
    print(f"  [{label_a} 백] vs [{label_b} 흑]  {n}판 진행 중...", flush=True)
    t0 = time.time()
    for i in range(n):
        r = _play_one(env, fn_b, fn_a)
        if   r == -1: ww += 1
        elif r ==  1: wl += 1
        else:         wd += 1
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (n - i - 1)
        print(f"  {i+1:2d}/{n}  백 {ww}승/{wl}패  남은시간~{remaining/60:.1f}m", end="\r", flush=True)
    print()

    total_win  = bw + ww
    total_loss = bl + wl
    total_draw = bd + wd

    print(f"\n  {'─'*52}")
    print(f"  {label_a}  vs  {label_b}  (총 {total_games}판, n_sim={N_SIM})")
    print(f"  흑(선공) :  {bw:2d}승 {bl:2d}패 {bd:2d}무  ({bw/n:.0%})")
    print(f"  백(후공) :  {ww:2d}승 {wl:2d}패 {wd:2d}무  ({ww/n:.0%})")
    print(f"  종합     :  {total_win:2d}승 {total_loss:2d}패 {total_draw:2d}무  "
          f"({total_win}/{total_games} = {total_win/total_games:.1%})")
    print(f"  {'─'*52}")

    return {
        "label_a": label_a, "label_b": label_b,
        "black_win": bw, "black_loss": bl, "black_draw": bd,
        "white_win": ww, "white_loss": wl, "white_draw": wd,
        "total_win": total_win, "total_loss": total_loss,
        "total_draw": total_draw, "total": total_games,
        "win_rate": total_win / total_games,
    }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 54)
    print("최종 요약")
    print("=" * 54)
    for r in results:
        print(
            f"  {r['label_a']} vs {r['label_b']}  :  "
            f"흑 {r['black_win']}승 / 백 {r['white_win']}승  "
            f"→  종합 {r['total_win']}/{r['total']} "
            f"({r['win_rate']:.1%})"
        )
    print("=" * 54)

    # 선공 이점 진단
    print("\n[선공 이점 분석]")
    all_black = sum(r["black_win"] for r in results)
    all_white = sum(r["white_win"] for r in results)
    all_n     = sum(r["total"] // 2 for r in results)
    print(f"  전체 흑 승률 : {all_black}/{all_n} = {all_black/all_n:.1%}")
    print(f"  전체 백 승률 : {all_white}/{all_n} = {all_white/all_n:.1%}")
    if abs(all_black - all_white) / all_n < 0.10:
        print("  → 색깔 이점이 10% 이내 — 성장이 승률을 주도하고 있음")
    else:
        print("  → 선공/후공 이점이 크게 작용 중 — 판 수를 늘리거나 원인 분석 필요")


if __name__ == "__main__":
    print("=" * 54)
    print(f"세대 아레나  —  n_sim={N_SIM}, 흑/백 각 {GAMES_PER_COLOR}판")
    print(f"device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"ckpt   : {CKPT_DIR}")
    print("=" * 54)

    # 체크포인트 수집
    ckpts: dict[str, str] = {}
    for it in ITERS:
        fname = f"az_15x15_iter{it:04d}.pt"
        path  = os.path.join(CKPT_DIR, fname)
        if os.path.exists(path):
            label = f"iter{it:04d}"
            ckpts[label] = path
            size_mb = os.path.getsize(path) / 1e6
            print(f"  [OK] {fname}  ({size_mb:.1f} MB)")
        else:
            print(f"  [!!] 없음: {path}")

    available = sorted(ckpts.keys())
    if len(available) < 2:
        print("\n체크포인트 2개 이상 필요. CKPT_DIR을 확인하세요:")
        print(f"  ls {CKPT_DIR}")
        sys.exit(1)

    # 대전 순서: 최신이 항상 label_a(신세대)
    pairs = [
        (available[-1], available[-2]),   # 30 vs 20 (또는 30 vs 10)
    ]
    if len(available) >= 3:
        pairs += [
            (available[-1], available[0]),  # 30 vs 10
            (available[-2], available[0]),  # 20 vs 10
        ]

    results = []
    t_total = time.time()
    for a, b in pairs:
        r = arena(a, ckpts[a], b, ckpts[b])
        results.append(r)

    print_summary(results)
    print(f"\n총 소요: {(time.time()-t_total)/60:.1f}분")

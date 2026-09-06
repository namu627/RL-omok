"""
scripts/selfplay_viz.py
학습된 AlphaZero 모델의 self-play를 시각화한다 (세대 비교 + PNG).

무엇을 하나
-----------
- checkpoints_banan1 의 한 세대 모델을 불러와, 그 모델이 흑·백 양쪽을 두는
  self-play 한 판을 진행한다 (렌주 15×15, n_in_row=5).
- 15×15 오목판을 그려 흑/백 돌 + 수순 번호(1,2,3...)를 표시하고,
  결과(승자 / 총 수)와 승리 5목 라인을 강조해 PNG로 저장한다.
- 여러 세대를 한 장에 나란히 놓아 "약할 때 vs 강할 때" 를 비교한다.

수(手) 선택 정책 (평가와 동일한 결정론 지향 + 약간의 무작위성)
--------------------------------------------------------------
- MCTS 방문수 분포에 낮은 temperature 를 적용해 "최선의 수"에 가깝게 두되,
  완전 greedy 는 아니어서 판마다 조금씩 달라진다.
    · 초반(OPENING_MOVES 수)  : temperature = TEMP_OPENING  (탐험 여지)
    · 그 이후                  : temperature = TEMP_ENDGAME  (거의 최선수)
  분포에서 np.random.choice 로 샘플. --seed 로 재현 가능.

GPU 사용
--------
- 기본 n_sim 이 작고(=120) 판 수도 세대당 1판이라 부하가 매우 작다.
- torch 스레드 수를 제한하고, --device 로 cpu 강제 가능.

실행 예
-------
  # iter5 vs iter30 비교 (기본)
  python -X utf8 scripts/selfplay_viz.py

  # 세대·시드·탐색량 지정
  python -X utf8 scripts/selfplay_viz.py --iters 5 15 30 --seed 7 --n-sim 160

  # GPU 안 쓰기
  python -X utf8 scripts/selfplay_viz.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from env.gomoku import GomokuEnv, BLACK, WHITE
from agents.alphazero.network import AlphaZeroNet
from agents.alphazero.mcts import MCTS

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ROOT, "checkpoints_banan1")
OUT_DIR  = os.path.join(ROOT, "selfplay_viz")

# ── 렌주 학습 설정 (train_renju_colab.py 와 동일) ────────────────────────
BOARD_SIZE = 15
N_IN_ROW   = 5
RENJU      = True
C_PUCT     = 1.5

# ── 수 선택 정책 ────────────────────────────────────────────────────────
OPENING_MOVES = 8
TEMP_OPENING  = 0.60   # 초반: 어느 정도 다양성
TEMP_ENDGAME  = 0.20   # 이후: 방문수 분포가 급격히 뾰족 → 거의 최선수


# ─────────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────────

def load_net(ckpt_path: str, device: str) -> AlphaZeroNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["net_state"]
    n_filters    = ckpt.get("n_filters")    or sd["input_conv.0.weight"].shape[0]
    n_res_blocks = ckpt.get("n_res_blocks") or sum(
        1 for k in sd if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
    )
    board_size   = ckpt.get("board_size", BOARD_SIZE)
    net = AlphaZeroNet(
        board_size=board_size, n_res_blocks=n_res_blocks, n_filters=n_filters
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


def ckpt_for_iter(it: int) -> str:
    path = os.path.join(CKPT_DIR, f"az_15x15_iter{it:04d}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


# ─────────────────────────────────────────────────────────────────────────
# self-play 한 판
# ─────────────────────────────────────────────────────────────────────────

def play_selfplay(net: AlphaZeroNet, env: GomokuEnv, n_sim: int, device: str,
                  rng: np.random.Generator) -> dict:
    """
    net 이 흑·백 양쪽을 두는 self-play 한 판.
    반환: {moves, winner, n_moves, reason, root_values}
      moves      : [(row, col, player), ...] 수순대로
      winner     : 1(흑)/-1(백)/0(무)
      root_values: 각 수에서 착수 직전 루트 가치(현재 플레이어 시점) — 자신감 추이
    """
    board = env.reset()
    player = env.current_player
    moves: list[tuple[int, int, int]] = []
    root_values: list[float] = []
    done = False
    reason = ""

    while not done:
        # MCTS._legal_actions_for()가 env.current_player를 탐색 중 덮어쓰고
        # 복원하지 않는다(공유 MCTS의 잠재 버그). 착수 색이 틀어지지 않도록
        # get_action_probs 호출 전·후로 실제 착수자로 동기화한다.
        env.current_player = player
        mcts = MCTS(net, env, n_simulations=n_sim, c_puct=C_PUCT, device=device)
        ply = len(moves)
        temp = TEMP_OPENING if ply < OPENING_MOVES else TEMP_ENDGAME

        pi = mcts.get_action_probs(board, player, temperature=temp)

        # 루트 가치(참고용): 네트워크 직접 평가
        _, v = net.predict(board, player, device=device)
        root_values.append(float(v))

        legal = np.flatnonzero(pi > 0)
        action = int(rng.choice(legal, p=pi[legal] / pi[legal].sum()))

        r, c = divmod(action, env.board_size)
        moves.append((r, c, player))

        env.current_player = player          # 탐색으로 오염된 값 복구 후 착수
        board, reward, done, info = env.step(action)
        if done:
            reason = info.get("reason", "")
        if not done:
            player = env.current_player

    return {
        "moves": moves,
        "winner": env.winner,
        "n_moves": env.move_count,
        "reason": reason,
        "root_values": root_values,
    }


# ─────────────────────────────────────────────────────────────────────────
# 승리 5목 라인 탐색 (강조용)
# ─────────────────────────────────────────────────────────────────────────

def find_winning_line(moves, board_size, n_in_row):
    grid = np.zeros((board_size, board_size), dtype=int)
    for r, c, p in moves:
        grid[r, c] = p
    lr, lc, lp = moves[-1]
    if grid[lr, lc] != lp:
        return []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        line = [(lr, lc)]
        for sign in (1, -1):
            r, c = lr + sign * dr, lc + sign * dc
            while 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == lp:
                line.append((r, c))
                r += sign * dr
                c += sign * dc
        if len(line) >= n_in_row:
            return line
    return []


# ─────────────────────────────────────────────────────────────────────────
# 판 그리기
# ─────────────────────────────────────────────────────────────────────────

def draw_board(ax, result: dict, title: str, board_size: int = BOARD_SIZE):
    moves = result["moves"]
    winner = result["winner"]
    win_line = set(find_winning_line(moves, board_size, N_IN_ROW))

    # 바탕
    ax.set_facecolor("#e8b878")
    ax.set_xlim(-1.0, board_size)
    ax.set_ylim(-1.0, board_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()                      # row 0 을 위로
    for i in range(board_size):
        ax.plot([i, i], [0, board_size - 1], color="#5c3a1e", lw=0.7, zorder=1)
        ax.plot([0, board_size - 1], [i, i], color="#5c3a1e", lw=0.7, zorder=1)

    # 화점
    for sr in (3, 7, 11):
        for sc in (3, 7, 11):
            ax.plot(sc, sr, "o", ms=4, color="#5c3a1e", zorder=2)

    last_idx = len(moves) - 1
    for i, (r, c, p) in enumerate(moves):
        is_black = (p == BLACK)
        face = "#1b1b1b" if is_black else "#fbfbfb"
        edge = "#000000"
        on_win = (r, c) in win_line
        ax.add_patch(Circle(
            (c, r), 0.44, facecolor=face, edgecolor=edge,
            lw=1.0, zorder=3,
        ))
        if on_win:
            ax.add_patch(Circle(
                (c, r), 0.46, facecolor="none", edgecolor="#d62728",
                lw=2.2, zorder=5,
            ))
        if i == last_idx and not on_win:
            ax.add_patch(Circle(
                (c, r), 0.46, facecolor="none", edgecolor="#1f77b4",
                lw=2.0, zorder=5,
            ))
        ax.text(
            c, r, str(i + 1),
            ha="center", va="center",
            fontsize=6.5 if len(moves) > 60 else 7.5,
            color="#f5f5f5" if is_black else "#111111",
            zorder=4,
        )

    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.set_xticklabels(range(board_size), fontsize=6)
    ax.set_yticklabels(range(board_size), fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color("#5c3a1e")

    if winner == BLACK:
        res = f"흑(Black) 승 · {result['n_moves']}수"
    elif winner == WHITE:
        res = f"백(White) 승 · {result['n_moves']}수"
    else:
        res = f"무승부 · {result['n_moves']}수"
    if result["reason"]:
        res += f"  ({result['reason']})"
    ax.set_title(f"{title}\n{res}", fontsize=11, fontweight="bold", pad=8)

    # ── 참고 지표 (약함/강함 성향 비교용) ────────────────────────────────
    grid = np.zeros((board_size, board_size), dtype=int)
    for r, c, p in moves:
        grid[r, c] = p

    def _adjacent_to_opp(r, c, me):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_size and 0 <= cc < board_size and grid[rr, cc] == -me:
                    return True
        return False

    # 백이 흑 돌에 인접해 둔 비율 = "국지적으로 대응/방어하려는 성향"
    w_moves = [(r, c) for r, c, p in moves if p == WHITE]
    w_local = sum(1 for r, c in w_moves if _adjacent_to_opp(r, c, WHITE))
    local_ratio = w_local / max(len(w_moves), 1)

    rv = result.get("root_values") or [0.0]
    conf = float(np.mean(np.abs(rv)))

    ax.set_xlabel(
        f"백의 국지 대응 비율 {local_ratio:.0%}  ·  모델 평균 확신도 |v| {conf:.2f}",
        fontsize=9,
    )


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AlphaZero self-play 시각화 (세대 비교)")
    ap.add_argument("--iters", type=int, nargs="+", default=[5, 30],
                    help="비교할 세대 목록 (기본: 5 30)")
    ap.add_argument("--n-sim", type=int, default=120, help="MCTS 시뮬레이션 수 (기본 120)")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드 (재현용)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cuda", "cpu"], help="추론 장치")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU 스레드 상한")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = args.device
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"self-play 시각화  |  세대 {args.iters}  |  n_sim={args.n_sim}  "
          f"|  device={device}  |  seed={args.seed}")
    print("=" * 60)

    env = GomokuEnv(board_size=BOARD_SIZE, n_in_row=N_IN_ROW, renju=RENJU)
    results: list[tuple[int, dict]] = []

    for it in args.iters:
        path = ckpt_for_iter(it)
        net = load_net(path, device)
        rng = np.random.default_rng(args.seed + it)   # 세대마다 다른 스트림
        torch.manual_seed(args.seed + it)

        t0 = time.time()
        res = play_selfplay(net, env, args.n_sim, device, rng)
        dt = time.time() - t0

        wtxt = {BLACK: "흑 승", WHITE: "백 승", 0: "무"}[res["winner"]]
        print(f"  iter{it:02d}: {wtxt} · {res['n_moves']}수 · {dt:.1f}s"
              + (f" · {res['reason']}" if res["reason"] else ""))
        results.append((it, res))

        # 개별 PNG
        fig, ax = plt.subplots(figsize=(6.2, 6.8))
        strength = "약함" if it <= 10 else ("중간" if it <= 20 else "강함")
        draw_board(ax, res, f"iter{it}  ({strength})")
        fig.tight_layout()
        p_ind = os.path.join(OUT_DIR, f"selfplay_iter{it:02d}.png")
        fig.savefig(p_ind, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {p_ind}")

    # 비교 PNG (나란히)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.8))
    if n == 1:
        axes = [axes]
    for ax, (it, res) in zip(axes, results):
        strength = "약함" if it <= 10 else ("중간" if it <= 20 else "강함")
        draw_board(ax, res, f"iter{it}  ({strength})")
    fig.suptitle(
        "AlphaZero 렌주 self-play — 세대 비교  "
        f"(n_sim={args.n_sim}, temp {TEMP_OPENING}→{TEMP_ENDGAME}, seed={args.seed})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    p_cmp = os.path.join(OUT_DIR, "selfplay_compare.png")
    fig.savefig(p_cmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n비교 이미지 → {p_cmp}")
    print("완료.")


if __name__ == "__main__":
    main()

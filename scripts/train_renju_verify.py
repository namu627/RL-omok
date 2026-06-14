"""
Phase 4a 검증 스크립트 — AlphaZero (15×15 렌주, 방안 B).

설정:
  n_sim=50 / 5게임/iter / 20iter / eval_interval=10 / eval_games=20
  체크포인트: 10iter마다 저장
  목적: 렌주 self-play 파이프라인 정상 동작 확인
  - 학습 중 forbidden-move 패배가 역전파로 처리되는지
  - 승률이 0에서 올라오는지

실행:
  python -X utf8 scripts/train_renju_verify.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.alphazero.trainer import AlphaZeroTrainer

ROOT     = os.path.join(os.path.dirname(__file__), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────
BOARD_SIZE   = 15
N_IN_ROW     = 5
RENJU        = True
N_RES_BLOCKS = 3
N_FILTERS    = 64

N_SIM        = 50
C_PUCT       = 1.5

N_ITERATIONS         = 20
SELF_PLAY_PER_ITER   = 5
BATCH_SIZE           = 128
TRAIN_STEPS_PER_ITER = 5
EVAL_INTERVAL        = 10
EVAL_GAMES           = 20
TEMPERATURE_CUTOFF   = 20
CKPT_INTERVAL        = 10

LR     = 2e-3
L2_REG = 1e-4

PLOT_PATH = os.path.join(ROOT, "az_renju_verify_winrate.png")
WR_PATH   = os.path.join(CKPT_DIR, "az_renju_verify_winrates.json")


def plot_winrate_curve(win_rates: list[float]) -> None:
    if not win_rates:
        return
    eval_iters = [(i + 1) * EVAL_INTERVAL for i in range(len(win_rates))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_iters, win_rates, "o-", linewidth=2, color="steelblue",
            markersize=7, label="vs 랜덤봇 승률")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% 기준선")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate vs Random Agent")
    ax.set_title(f"AlphaZero Phase 4a 검증 — 렌주 15×15 (n_sim={N_SIM})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for x, y in zip(eval_iters, win_rates):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"승률 곡선 저장: {os.path.abspath(PLOT_PATH)}")
    plt.close(fig)


def main():
    print("=" * 58)
    print("Phase 4a 검증 — AlphaZero 렌주 15×15 (방안 B)")
    print(f"  MCTS {N_SIM}sim  |  self-play {SELF_PLAY_PER_ITER}게임/iter")
    print(f"  {N_ITERATIONS}iter  |  eval {EVAL_INTERVAL}iter마다 {EVAL_GAMES}판")
    print(f"  렌주 금수 처리: 루트=필터, 트리내부=raw+step판정")
    print("=" * 58)

    trainer = AlphaZeroTrainer(
        board_size=BOARD_SIZE,
        n_in_row=N_IN_ROW,
        renju=RENJU,
        n_res_blocks=N_RES_BLOCKS,
        n_filters=N_FILTERS,
        n_simulations=N_SIM,
        c_puct=C_PUCT,
        n_iterations=N_ITERATIONS,
        self_play_games_per_iter=SELF_PLAY_PER_ITER,
        batch_size=BATCH_SIZE,
        train_steps_per_iter=TRAIN_STEPS_PER_ITER,
        eval_interval=EVAL_INTERVAL,
        eval_games=EVAL_GAMES,
        lr=LR,
        l2_reg=L2_REG,
        ckpt_dir=CKPT_DIR,
        device="cpu",
        temperature_cutoff=TEMPERATURE_CUTOFF,
        ckpt_interval=CKPT_INTERVAL,
    )

    t0 = time.time()
    win_rates = trainer.train()
    total = time.time() - t0

    print(f"\n학습 완료 — 총 소요: {total/60:.1f}분")

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(WR_PATH, "w", encoding="utf-8") as f:
        json.dump(win_rates, f)

    plot_winrate_curve(win_rates)

    if win_rates:
        print(f"\n최종 승률 (vs 랜덤봇): {win_rates[-1]:.3f}")
    print("=== Phase 4a 검증 완료 ===")


if __name__ == "__main__":
    main()

"""
Phase 3 본 학습 스크립트 — AlphaZero (6×6/4목, Option A).

설정:
  n_sim=100 / 10게임/iter / 60iter / eval_interval=10 / eval_games=50
  체크포인트: 20iter마다 세대번호 포함 저장
  학습 후: 세대 간 아레나 대결 (iter020 vs iter040 vs iter060/final)
  출력: 승률 곡선 az_phase3_winrate.png

실행:
  python -X utf8 scripts/train_phase3_full.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.alphazero.trainer import AlphaZeroTrainer
from eval.arena import arena_battle

ROOT     = os.path.join(os.path.dirname(__file__), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# ── 하이퍼파라미터 (Option A) ──────────────────────────────────────────
BOARD_SIZE   = 6
N_IN_ROW     = 4
N_RES_BLOCKS = 3
N_FILTERS    = 64

N_SIM        = 100
C_PUCT       = 1.5

N_ITERATIONS         = 60
SELF_PLAY_PER_ITER   = 10
BATCH_SIZE           = 256
TRAIN_STEPS_PER_ITER = 5
EVAL_INTERVAL        = 10
EVAL_GAMES           = 50
TEMPERATURE_CUTOFF   = 10
CKPT_INTERVAL        = 20   # 20iter마다 번호 포함 체크포인트

LR     = 2e-3
L2_REG = 1e-4

PLOT_PATH = os.path.join(ROOT, "az_phase3_winrate.png")
WR_PATH   = os.path.join(CKPT_DIR, "az_6x6_winrates.json")
ARENA_PATH = os.path.join(ROOT, "az_phase3_arena.png")


# ── 세대 간 아레나 대결 + 시각화 ─────────────────────────────────────

def run_arena(ckpt_dir: str) -> None:
    """저장된 iter 체크포인트끼리 순서대로 대결."""
    import glob
    pattern = os.path.join(ckpt_dir, f"az_{BOARD_SIZE}x{BOARD_SIZE}_iter*.pt")
    ckpts = sorted(glob.glob(pattern))
    if len(ckpts) < 2:
        print("  아레나: 체크포인트 2개 이상 필요 — 건너뜀")
        return

    print(f"\n{'='*55}")
    print("세대 간 아레나 대결")
    print(f"{'='*55}")

    labels = []
    new_winrates = []
    pairs = []

    for i in range(len(ckpts) - 1):
        old_ckpt = ckpts[i]
        new_ckpt = ckpts[i + 1]
        old_label = os.path.basename(old_ckpt).replace(".pt", "")
        new_label = os.path.basename(new_ckpt).replace(".pt", "")
        print(f"\n  {new_label} vs {old_label}")
        result = arena_battle(new_ckpt, old_ckpt, n_games=40, n_simulations=N_SIM)
        pairs.append((new_label, result["new_win_rate"]))
        new_winrates.append(result["new_win_rate"])
        labels.append(f"{new_label}\nvs {old_label}")

    # 아레나 결과 시각화
    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 2), 5))
    colors = ["green" if w >= 0.5 else "salmon" for _, w in pairs]
    ax.bar(range(len(pairs)), [w for _, w in pairs], color=colors, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% (동등)")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("신규 세대 승률")
    ax.set_title("AlphaZero Phase 3 — 세대 간 아레나 대결")
    ax.set_ylim(0, 1)
    ax.legend()
    for i, (_, w) in enumerate(pairs):
        ax.text(i, w + 0.02, f"{w:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(ARENA_PATH, dpi=150)
    print(f"\n  아레나 그래프 저장: {os.path.abspath(ARENA_PATH)}")
    plt.close(fig)


# ── 승률 곡선 시각화 ──────────────────────────────────────────────────

def plot_winrate_curve(win_rates: list[float]) -> None:
    if not win_rates:
        return
    eval_iters = [(i + 1) * EVAL_INTERVAL for i in range(len(win_rates))]
    rates = np.array(win_rates)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_iters, rates, "o-", alpha=0.4, linewidth=1.5,
            color="steelblue", markersize=4, label="vs 랜덤봇 승률")

    # 이동평균 (데이터 충분할 때)
    if len(rates) >= 3:
        window = min(3, len(rates))
        kernel = np.ones(window) / window
        smoothed = np.convolve(rates, kernel, mode="valid")
        smooth_x = eval_iters[window - 1:]
        ax.plot(smooth_x, smoothed, linewidth=2.5, color="steelblue",
                label=f"이동평균 (w={window})")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% 기준선")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate vs Random Agent")
    ax.set_title(f"AlphaZero Phase 3 ({BOARD_SIZE}×{BOARD_SIZE}/{N_IN_ROW}목) — vs 랜덤봇 승률")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if win_rates:
        ax.annotate(
            f"최종: {win_rates[-1]:.3f}",
            xy=(eval_iters[-1], win_rates[-1]),
            xytext=(-60, 15), textcoords="offset points",
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"승률 곡선 저장: {os.path.abspath(PLOT_PATH)}")
    plt.close(fig)


# ── 메인 ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Phase 3 본 학습 — AlphaZero (Option A)")
    print(f"  {BOARD_SIZE}×{BOARD_SIZE}/{N_IN_ROW}목  |  MCTS {N_SIM}sim")
    print(f"  {N_ITERATIONS}iter  |  self-play {SELF_PLAY_PER_ITER}게임/iter")
    print(f"  평가 {EVAL_INTERVAL}iter마다 {EVAL_GAMES}판  |  ckpt {CKPT_INTERVAL}iter마다")
    print("=" * 55)

    trainer = AlphaZeroTrainer(
        board_size=BOARD_SIZE,
        n_in_row=N_IN_ROW,
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

    # 승률 기록 저장
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(WR_PATH, "w", encoding="utf-8") as f:
        json.dump(win_rates, f)
    print(f"승률 기록: {WR_PATH}")

    # 승률 곡선
    plot_winrate_curve(win_rates)

    # 세대 간 아레나 대결
    run_arena(CKPT_DIR)

    print(f"\n최종 승률 (vs 랜덤봇): {win_rates[-1]:.3f}" if win_rates else "")
    print("=== Phase 3 완료 ===")


if __name__ == "__main__":
    main()

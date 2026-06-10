"""
9×9/5목 자유룰 AlphaZero 검증 학습.

목적: "보드를 키워도 수렴 신호가 보이는가" 확인 (강한 모델 목표 아님).
설정: sim=50 / 5게임/iter / 60iter / eval 20iter마다 20판
예상: ~32분

실행:
  python -X utf8 scripts/train_9x9_verify.py
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

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────
BOARD_SIZE   = 9
N_IN_ROW     = 5
N_RES_BLOCKS = 4   # 9×9이므로 한 블록 추가
N_FILTERS    = 64

N_SIM        = 50
C_PUCT       = 1.5

N_ITERATIONS         = 60
SELF_PLAY_PER_ITER   = 5
BATCH_SIZE           = 256
TRAIN_STEPS_PER_ITER = 5
EVAL_INTERVAL        = 20   # 20iter마다 (총 3회)
EVAL_GAMES           = 20
TEMPERATURE_CUTOFF   = 15   # 9×9은 게임이 길어 탐색 구간 늘림
CKPT_INTERVAL        = 20

LR     = 2e-3
L2_REG = 1e-4

TAG       = f"az_{BOARD_SIZE}x{BOARD_SIZE}"
PLOT_PATH = os.path.join(ROOT, f"{TAG}_verify_winrate.png")
WR_PATH   = os.path.join(CKPT_DIR, f"{TAG}_verify_winrates.json")
ARENA_PATH = os.path.join(ROOT, f"{TAG}_verify_arena.png")


# ── 승률 곡선 ─────────────────────────────────────────────────────────

def plot_winrate_curve(win_rates: list[float]) -> None:
    if not win_rates:
        return
    eval_iters = [(i + 1) * EVAL_INTERVAL for i in range(len(win_rates))]
    rates = np.array(win_rates)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eval_iters, rates, "o-", linewidth=2, color="steelblue",
            markersize=7, label="Win Rate vs Random")
    for x, y in zip(eval_iters, rates):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate vs Random Agent")
    ax.set_title(f"AlphaZero 9x9/5-in-a-row — Win Rate vs Random (sim={N_SIM})")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Win rate curve saved: {os.path.abspath(PLOT_PATH)}")
    plt.close(fig)


# ── 세대 간 아레나 ────────────────────────────────────────────────────

def run_arena(ckpt_dir: str) -> None:
    import glob
    pattern = os.path.join(ckpt_dir, f"{TAG}_iter*.pt")
    ckpts = sorted(glob.glob(pattern))
    if len(ckpts) < 2:
        print("  Arena: need at least 2 checkpoints — skipping")
        return

    print(f"\n{'='*50}")
    print("Generation Arena Battle")
    print(f"{'='*50}")

    pairs = []
    for i in range(len(ckpts) - 1):
        old_ck, new_ck = ckpts[i], ckpts[i + 1]
        old_label = os.path.basename(old_ck).replace(".pt", "")
        new_label = os.path.basename(new_ck).replace(".pt", "")
        print(f"\n  {new_label} vs {old_label}")
        result = arena_battle(new_ck, old_ck, n_games=20, n_simulations=N_SIM)
        pairs.append((f"{new_label}\nvs {old_label}", result["new_win_rate"]))

    # 아레나 그래프
    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 2.5), 5))
    colors = ["steelblue" if w > 0.5 else "salmon" if w < 0.5 else "gray"
              for _, w in pairs]
    ax.bar(range(len(pairs)), [w for _, w in pairs], color=colors, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="50% (even)")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([l for l, _ in pairs], fontsize=8)
    ax.set_ylabel("New Gen Win Rate")
    ax.set_title(f"AlphaZero 9x9 — Generation Arena (sim={N_SIM})")
    ax.set_ylim(0, 1.1)
    ax.legend()
    for i, (_, w) in enumerate(pairs):
        ax.text(i, w + 0.03, f"{w:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ARENA_PATH, dpi=150)
    print(f"\n  Arena graph saved: {os.path.abspath(ARENA_PATH)}")
    plt.close(fig)


# ── 메인 ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(f"9x9/5-in-a-row AlphaZero Verification")
    print(f"  sim={N_SIM}  |  {SELF_PLAY_PER_ITER} games/iter  |  {N_ITERATIONS} iter")
    print(f"  eval every {EVAL_INTERVAL} iter ({EVAL_GAMES} games)  |  ckpt every {CKPT_INTERVAL} iter")
    print(f"  ResBlocks={N_RES_BLOCKS}  Filters={N_FILTERS}")
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
    print(f"\nDone — total: {total/60:.1f} min")

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(WR_PATH, "w") as f:
        json.dump(win_rates, f)
    print(f"Win rates saved: {WR_PATH}")

    plot_winrate_curve(win_rates)
    run_arena(CKPT_DIR)

    if win_rates:
        print(f"\nFinal win rate vs random: {win_rates[-1]:.3f}")
    print("=== 9x9 Verification Done ===")


if __name__ == "__main__":
    main()

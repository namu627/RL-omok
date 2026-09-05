"""
scripts/train_renju_colab.py
Google Colab GPU 학습 스크립트 — AlphaZero 렌주 15×15.

실행:
  스모크 테스트:  python scripts/train_renju_colab.py smoke
  본 학습:       python scripts/train_renju_colab.py train
  재개:          python scripts/train_renju_colab.py train --resume
"""

import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.alphazero.trainer import AlphaZeroTrainer

# ── Device 자동 선택 ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 경로: Google Drive 마운트 감지 → Drive 우선, 없으면 로컬 ─────────────────
# checkpoints_banan1: 방안1(트리 내부 금수 완전 배제) 전용 실행 — 기존
# checkpoints/, checkpoints_dirichlet/ 와 분리해 덮어쓰기 방지.
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DRIVE   = "/content/drive/MyDrive/rl_omok"
CKPT_DIR = os.path.join(_DRIVE, "checkpoints_banan1") if os.path.isdir("/content/drive") else os.path.join(ROOT, "checkpoints_banan1")
PLOT_DIR = CKPT_DIR

# ── 공통 고정 설정 ────────────────────────────────────────────────────────
BOARD_SIZE           = 15
N_IN_ROW             = 5
RENJU                = True
N_RES_BLOCKS         = 5    # GPU용 확장 (검증 때는 3)
N_FILTERS            = 128  # GPU용 확장 (검증 때는 64)
BATCH_SIZE           = 128  # 로컬 GTX 1650 (4GB) 대응 — 512는 OOM 위험
TRAIN_STEPS_PER_ITER = 10
EVAL_INTERVAL        = 10   # 30iter 규모: 10마다 평가 → iter 10/20/30에서 승률 출력
EVAL_GAMES           = 30
TEMPERATURE_CUTOFF   = 20
CKPT_INTERVAL        = 5    # 5iter마다 체크포인트 저장
LR                   = 2e-3
L2_REG               = 1e-4

# ── [스모크] GPU iter당 시간 측정 → 본 학습 규모 결정용 ──────────────────────
# 스모크 결과를 보고 아래 FULL_* 값을 조정하세요.
SMOKE_N_SIM      = 50
SMOKE_SP_GAMES   = 25   # B안(인터 게임 배치): 본 학습과 동일 sp_games로 속도 측정
SMOKE_N_ITER     = 3

# ── [본 학습] 스모크 후 아래 값 직접 조정 ────────────────────────────────────
FULL_N_SIM       = 200
FULL_SP_GAMES    = 25
FULL_N_ITER      = 30


# ─────────────────────────────────────────────────────────────────────────

def print_device_info():
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM : {gb:.1f} GB")
    print(f"CKPT   : {CKPT_DIR}")


DIRICHLET_ALPHA   = 0.3   # 렌주 15×15 합법 수 규모 기준
DIRICHLET_EPSILON = 0.25  # AlphaZero 논문 표준값


def make_trainer(n_sim, sp_games, n_iter):
    return AlphaZeroTrainer(
        board_size=BOARD_SIZE, n_in_row=N_IN_ROW, renju=RENJU,
        n_res_blocks=N_RES_BLOCKS, n_filters=N_FILTERS,
        n_simulations=n_sim, c_puct=1.5,
        n_iterations=n_iter,
        self_play_games_per_iter=sp_games,
        batch_size=BATCH_SIZE,
        train_steps_per_iter=TRAIN_STEPS_PER_ITER,
        eval_interval=EVAL_INTERVAL, eval_games=EVAL_GAMES,
        lr=LR, l2_reg=L2_REG,
        ckpt_dir=CKPT_DIR, device=DEVICE,
        temperature_cutoff=TEMPERATURE_CUTOFF,
        ckpt_interval=CKPT_INTERVAL,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
    )


def run_smoke():
    print("=" * 58)
    print(f"GPU 스모크 테스트  ({SMOKE_N_ITER}iter, n_sim={SMOKE_N_SIM})")
    print_device_info()
    print("=" * 58)

    trainer = make_trainer(SMOKE_N_SIM, SMOKE_SP_GAMES, SMOKE_N_ITER)
    t0 = time.time()
    trainer.train(resume=False)
    total = time.time() - t0
    avg = total / SMOKE_N_ITER

    print("\n" + "─" * 58)
    print(f"스모크 완료: 총 {total:.0f}s  |  평균 {avg:.1f}s/iter")
    print("\n본 학습 소요 예측 (n_sim 비례, sp_games 비례):")
    for n_iter in [50, 100, 200]:
        for sp in [10, 25]:
            est = avg * n_iter * (sp / SMOKE_SP_GAMES) * (FULL_N_SIM / SMOKE_N_SIM)
            print(f"  {n_iter:3d}iter × {sp:2d}games × n_sim={FULL_N_SIM} : ~{est/3600:.1f}h")
    print("─" * 58)
    print("→ FULL_N_SIM / FULL_SP_GAMES / FULL_N_ITER 값을 조정 후 train 실행")


def run_train(resume: bool = False):
    print("=" * 58)
    print("본 학습 — AlphaZero 렌주 15×15  [Dirichlet 노이즈 활성]")
    print(f"  n_sim={FULL_N_SIM}  sp_games={FULL_SP_GAMES}  n_iter={FULL_N_ITER}")
    print(f"  net: {N_RES_BLOCKS}blocks × {N_FILTERS}ch  |  resume={resume}")
    print(f"  dirichlet: alpha={DIRICHLET_ALPHA}  epsilon={DIRICHLET_EPSILON}")
    print_device_info()
    print("=" * 58)

    os.makedirs(CKPT_DIR, exist_ok=True)
    trainer = make_trainer(FULL_N_SIM, FULL_SP_GAMES, FULL_N_ITER)

    t0 = time.time()
    win_rates = trainer.train(resume=resume)
    total = time.time() - t0

    print(f"\n학습 완료 — 총 {total/3600:.2f}h ({total/60:.0f}min)")

    # JSON 저장
    results = {
        "win_rates": win_rates,
        "total_sec": total,
        "device": DEVICE,
        "n_sim": FULL_N_SIM,
        "sp_games": FULL_SP_GAMES,
        "n_iter": FULL_N_ITER,
        "n_res_blocks": N_RES_BLOCKS,
        "n_filters": N_FILTERS,
    }
    json_path = os.path.join(CKPT_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"결과 JSON: {json_path}")

    # 승률 곡선 플롯
    if win_rates:
        eval_iters = [(i + 1) * EVAL_INTERVAL for i in range(len(win_rates))]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_iters, win_rates, "o-", lw=2, ms=6, color="steelblue",
                label="vs random")
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        for x, y in zip(eval_iters, win_rates):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate vs Random Agent")
        ax.set_title(
            f"AlphaZero Renju 15x15  "
            f"(n_sim={FULL_N_SIM}, {FULL_SP_GAMES}games/iter, {N_RES_BLOCKS}blocks x {N_FILTERS}ch)"
        )
        ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, "az_renju_winrate.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"승률 곡선: {plot_path}")

    print("=== 완료 ===")


# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode   = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    resume = "--resume" in sys.argv

    if mode == "smoke":
        run_smoke()
    elif mode == "train":
        run_train(resume=resume)
    else:
        print("Usage: python train_renju_colab.py [smoke|train] [--resume]")
        sys.exit(1)

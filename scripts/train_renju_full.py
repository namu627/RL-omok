"""
Phase 4a 완전 검증 — AlphaZero 렌주 15×15 (20iter clean run).

이전 az_15x15 체크포인트를 정리하고 처음부터 20iter 완주.
종료 후 4가지 결과를 az_renju_full_results.png로 저장.

실행:
  python -X utf8 scripts/train_renju_full.py
"""

import sys, os, json, time, platform
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from agents.alphazero.trainer import AlphaZeroTrainer
from agents.alphazero.network import AlphaZeroNet
from agents.alphazero.mcts import MCTS, MCTSNode
from env.gomoku import GomokuEnv

# ── 경로 ─────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
PLOT_PATH = os.path.join(ROOT, "az_renju_full_results.png")
JSON_PATH = os.path.join(CKPT_DIR, "az_renju_full_results.json")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────
BOARD_SIZE           = 15
N_IN_ROW             = 5
RENJU                = True
N_RES_BLOCKS         = 3
N_FILTERS            = 64
N_SIM                = 50
C_PUCT               = 1.5
N_ITERATIONS         = 20
SELF_PLAY_PER_ITER   = 5
BATCH_SIZE           = 128
TRAIN_STEPS_PER_ITER = 5
EVAL_INTERVAL        = 10
EVAL_GAMES           = 20
TEMPERATURE_CUTOFF   = 20
CKPT_INTERVAL        = 10   # iter10, iter20 모두 저장
LR                   = 2e-3
L2_REG               = 1e-4
PROBE_GAMES          = 10   # eval마다 raw-root MCTS 금수 프로브
ARENA_GAMES          = 20   # iter10 vs iter20 아레나


# ── 정리 ─────────────────────────────────────────────────────────────────

def clean_old_checkpoints():
    targets = [
        os.path.join(CKPT_DIR, "az_15x15_iter0010.pt"),
        os.path.join(CKPT_DIR, "az_15x15_final.pt"),
        os.path.join(CKPT_DIR, "az_renju_verify_winrates.json"),
        os.path.join(CKPT_DIR, "az_renju_full_results.json"),
        os.path.join(ROOT, "az_renju_verify_winrate.png"),
        os.path.join(ROOT, "az_renju_full_results.png"),
    ]
    removed = [p for p in targets if os.path.exists(p) and not os.remove(p)]
    if removed:
        print(f"정리: {[os.path.basename(p) for p in removed]}")
    else:
        print("정리할 파일 없음 (깨끗한 시작)")


# ── Raw-root MCTS (금수 프로브 전용) ─────────────────────────────────────

class _RawRootMCTS(MCTS):
    """루트에서도 금수 필터 없이 빈 칸 전체를 탐색."""

    def _build_root(self, board: np.ndarray, current_player: int) -> MCTSNode:
        root = MCTSNode(
            parent=None, action=None,
            board=board.copy(), current_player=current_player,
        )
        policy, _ = self.net.predict(board, current_player, device=self.device)
        raw = self._raw_legal_actions(board)
        mask = np.zeros_like(policy)
        mask[raw] = 1.0
        policy = policy * mask
        s = policy.sum()
        policy = policy / s if s > 0 else mask / max(len(raw), 1)
        self._expand(root, policy, raw)
        return root


# ── VerifyTrainer ─────────────────────────────────────────────────────────

class VerifyTrainer(AlphaZeroTrainer):
    """기본 Trainer + forbidden-probe(eval마다) + per-iter 타이밍 기록."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden_probes: list[int] = []
        self.iter_times: list[float] = []

    def train(self) -> list[float]:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        t_start = time.time()

        for iteration in range(1, self.n_iter + 1):
            t_iter = time.time()

            new_data = self._collect_self_play()
            self.buffer.push(new_data)

            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_steps):
                    self._train_step()

            iter_elapsed = time.time() - t_iter
            self.iter_times.append(iter_elapsed)
            total_elapsed = time.time() - t_start

            if iteration % self.eval_interval == 0:
                win_rate = self._evaluate_vs_random()
                self.win_rates.append(win_rate)
                n_forb = self._probe_forbidden(PROBE_GAMES)
                self.forbidden_probes.append(n_forb)
                print(
                    f"  [EVAL] iter {iteration:4d}/{self.n_iter}"
                    f"  buf={len(self.buffer):6d}"
                    f"  win_vs_rnd={win_rate:.3f}"
                    f"  forbidden_probe={n_forb}/{PROBE_GAMES}"
                    f"  iter_t={iter_elapsed:.0f}s"
                    f"  total={total_elapsed/60:.1f}m",
                    flush=True,
                )
            else:
                print(
                    f"  iter {iteration:4d}/{self.n_iter}"
                    f"  buf={len(self.buffer):6d}"
                    f"  iter_t={iter_elapsed:.0f}s"
                    f"  total={total_elapsed/60:.1f}m",
                    flush=True,
                )

            if self.ckpt_interval > 0 and iteration % self.ckpt_interval == 0:
                path = os.path.join(
                    self.ckpt_dir,
                    f"az_{self.board_size}x{self.board_size}_iter{iteration:04d}.pt",
                )
                self.save(path)

        final_path = os.path.join(
            self.ckpt_dir,
            f"az_{self.board_size}x{self.board_size}_final.pt",
        )
        self.save(final_path)
        return self.win_rates

    def _probe_forbidden(self, n_games: int) -> int:
        """
        raw-root MCTS로 n_games 판을 돌려 흑이 금수를 두는 횟수를 센다.
        (루트 필터 제거 → 모델이 스스로 금수를 회피하는지 확인)
        """
        probe_env = GomokuEnv(
            board_size=self.board_size, n_in_row=self.n_in_row, renju=self.renju
        )
        mcts = _RawRootMCTS(
            self.net, probe_env,
            n_simulations=self.n_sim, c_puct=self.c_puct, device=self.device,
        )
        total = 0
        for _ in range(n_games):
            board = probe_env.reset()
            player = probe_env.current_player
            done = False
            while not done:
                pi = mcts.get_action_probs(board, player, temperature=1e-4)
                action = int(np.argmax(pi))
                board, reward, done, info = probe_env.step(action)
                if done and reward == -1.0 and info.get("reason", "").startswith("금수"):
                    total += 1
                if not done:
                    player = probe_env.current_player
        return total


# ── Arena ─────────────────────────────────────────────────────────────────

def load_net(path: str, device: str = "cpu") -> AlphaZeroNet:
    ckpt = torch.load(path, map_location=device)
    net = AlphaZeroNet(
        board_size=BOARD_SIZE,
        n_res_blocks=N_RES_BLOCKS,
        n_filters=N_FILTERS,
    ).to(device)
    net.load_state_dict(ckpt["net_state"])
    net.eval()
    return net


def run_arena(net_early, net_late, n_games: int = 20, device: str = "cpu"):
    """
    iter10(early) vs iter20(late) 아레나. 흑/백 각 half판씩.
    반환: (late_wins, early_wins, draws)
    """
    env = GomokuEnv(board_size=BOARD_SIZE, n_in_row=N_IN_ROW, renju=RENJU)

    def get_action(net, board, player):
        mcts = MCTS(net, env, n_simulations=N_SIM, c_puct=C_PUCT, device=device)
        pi = mcts.get_action_probs(board, player, temperature=1e-4)
        return int(np.argmax(pi))

    late_wins = early_wins = draws = 0
    half = n_games // 2

    for rnd, (late_color, early_color) in enumerate([(1, -1), (-1, 1)]):
        side = "흑" if late_color == 1 else "백"
        print(f"  아레나 라운드{rnd+1} (iter20={side}) ...", end="", flush=True)
        r_late = r_early = r_draw = 0
        for _ in range(half):
            board = env.reset()
            player = env.current_player
            done = False
            while not done:
                if player == late_color:
                    action = get_action(net_late, board, player)
                else:
                    action = get_action(net_early, board, player)
                board, reward, done, info = env.step(action)
                if not done:
                    player = env.current_player
            w = env.winner
            if w == late_color:
                late_wins += 1; r_late += 1
            elif w == early_color:
                early_wins += 1; r_early += 1
            else:
                draws += 1; r_draw += 1
        print(f" iter20 {r_late}승 / iter10 {r_early}승 / 무 {r_draw}")

    return late_wins, early_wins, draws


# ── 플롯 ─────────────────────────────────────────────────────────────────

def plot_results(win_rates, forbidden_probes, iter_times, arena_result, total_time):
    eval_iters = [(i + 1) * EVAL_INTERVAL for i in range(len(win_rates))]
    late_wins, early_wins, draws = arena_result

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"AlphaZero 렌주 15×15 검증 (n_sim={N_SIM}, {SELF_PLAY_PER_ITER}판/iter, {N_ITERATIONS}iter)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # ① 승률 곡선
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eval_iters, win_rates, "o-", lw=2, color="steelblue", ms=10)
    ax1.axhline(0.5, color="gray", ls="--", lw=1, label="50% 기준")
    for x, y in zip(eval_iters, win_rates):
        ax1.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 12),
                     textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Win Rate")
    ax1.set_title("① vs 랜덤봇 승률 곡선"); ax1.set_ylim(0, 1.1)
    ax1.set_xticks(eval_iters); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ② 세대 아레나
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [f"iter20 승\n(후기)", f"iter10 승\n(초기)", "무승부"]
    values = [late_wins, early_wins, draws]
    colors = ["#2ecc71", "#e74c3c", "#bdc3c7"]
    bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 str(v), ha="center", fontsize=14, fontweight="bold")
    ax2.set_ylabel("게임 수")
    ax2.set_title(f"② 세대 아레나 (총 {ARENA_GAMES}판)\niter20 승률: {late_wins/ARENA_GAMES:.0%}")
    ax2.set_ylim(0, ARENA_GAMES + 4); ax2.grid(True, alpha=0.2, axis="y")

    # ③ 금수 회피 프로브
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(eval_iters, forbidden_probes, "s-", lw=2, color="crimson", ms=10)
    ax3.axhline(0, color="green", ls="--", lw=1.5, label="목표: 0회")
    for x, y in zip(eval_iters, forbidden_probes):
        ax3.annotate(str(y), xy=(x, y), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Iteration"); ax3.set_ylabel(f"금수 착수 수 / {PROBE_GAMES}판")
    ax3.set_title(f"③ 금수 회피 프로브\n(루트 필터 제거 시 모델이 스스로 회피하는가)")
    ax3.set_ylim(-0.5, PROBE_GAMES + 1); ax3.set_xticks(eval_iters)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # ④ iter당 소요 시간
    ax4 = fig.add_subplot(gs[1, 1])
    iters = list(range(1, len(iter_times) + 1))
    ax4.bar(iters, iter_times, color="steelblue", alpha=0.6, width=0.8)
    avg_t = float(np.mean(iter_times))
    ax4.axhline(avg_t, color="navy", ls="--", lw=1.5, label=f"평균 {avg_t:.0f}s")
    ax4.set_xlabel("Iteration"); ax4.set_ylabel("초 (sec)")
    ax4.set_title(f"④ iter당 소요 시간 (총 {total_time/60:.1f}분)")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3, axis="y")

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"  결과 저장: {PLOT_PATH}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Phase 4a 완전 검증 — AlphaZero 렌주 15×15")
    print(f"  MCTS {N_SIM}sim  |  self-play {SELF_PLAY_PER_ITER}판/iter  |  {N_ITERATIONS}iter")
    print(f"  eval {EVAL_INTERVAL}iter마다 {EVAL_GAMES}판  |  forbidden probe {PROBE_GAMES}판/eval")
    print("=" * 64)

    os.makedirs(CKPT_DIR, exist_ok=True)
    clean_old_checkpoints()

    # ── 학습 ──────────────────────────────────────────────────────────
    trainer = VerifyTrainer(
        board_size=BOARD_SIZE, n_in_row=N_IN_ROW, renju=RENJU,
        n_res_blocks=N_RES_BLOCKS, n_filters=N_FILTERS,
        n_simulations=N_SIM, c_puct=C_PUCT,
        n_iterations=N_ITERATIONS,
        self_play_games_per_iter=SELF_PLAY_PER_ITER,
        batch_size=BATCH_SIZE,
        train_steps_per_iter=TRAIN_STEPS_PER_ITER,
        eval_interval=EVAL_INTERVAL, eval_games=EVAL_GAMES,
        lr=LR, l2_reg=L2_REG,
        ckpt_dir=CKPT_DIR, device="cpu",
        temperature_cutoff=TEMPERATURE_CUTOFF,
        ckpt_interval=CKPT_INTERVAL,
    )

    t0 = time.time()
    win_rates = trainer.train()
    total_time = time.time() - t0
    print(f"\n학습 완료 — 총 {total_time/60:.1f}분")

    # ── 세대 아레나 ───────────────────────────────────────────────────
    ckpt_iter10 = os.path.join(CKPT_DIR, "az_15x15_iter0010.pt")
    ckpt_iter20 = os.path.join(CKPT_DIR, "az_15x15_final.pt")
    print("\n세대 간 아레나 시작...")
    net_early = load_net(ckpt_iter10)
    net_late  = load_net(ckpt_iter20)
    late_wins, early_wins, draws = run_arena(net_early, net_late, ARENA_GAMES)
    print(f"  최종: iter20 {late_wins}승 / iter10 {early_wins}승 / 무 {draws}  "
          f"(iter20 승률 {late_wins/ARENA_GAMES:.0%})")

    # ── 결과 저장 ─────────────────────────────────────────────────────
    results = {
        "win_rates": win_rates,
        "forbidden_probes": trainer.forbidden_probes,
        "iter_times": trainer.iter_times,
        "arena": {"late_wins": late_wins, "early_wins": early_wins, "draws": draws},
        "total_time_sec": total_time,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_results(
        win_rates=win_rates,
        forbidden_probes=trainer.forbidden_probes,
        iter_times=trainer.iter_times,
        arena_result=(late_wins, early_wins, draws),
        total_time=total_time,
    )

    # ── 최종 요약 ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("최종 요약")
    if len(win_rates) >= 2:
        print(f"  ① 승률 (vs 랜덤봇)  : iter10={win_rates[0]:.3f} → iter20={win_rates[-1]:.3f}")
    elif win_rates:
        print(f"  ① 승률 (vs 랜덤봇)  : iter10={win_rates[0]:.3f}")
    print(f"  ② 아레나 결과       : iter20 {late_wins}승 / iter10 {early_wins}승 / 무 {draws}"
          f"  → iter20 승률 {late_wins/ARENA_GAMES:.0%}")
    forb = trainer.forbidden_probes
    forb_str = " → ".join(f"iter{(i+1)*EVAL_INTERVAL}:{v}회" for i, v in enumerate(forb))
    print(f"  ③ 금수 프로브({PROBE_GAMES}판): {forb_str}")
    avg_t = float(np.mean(trainer.iter_times))
    print(f"  ④ iter 평균 {avg_t:.0f}s / 총 {total_time/60:.1f}분")
    print("=" * 64)
    print("=== Phase 4a 완전 검증 완료 ===")


if __name__ == "__main__":
    main()

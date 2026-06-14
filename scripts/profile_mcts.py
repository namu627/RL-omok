"""
scripts/profile_mcts.py
MCTS 병목 진단 — GPU predict 시간 vs MCTS Python 로직 시간 분리 측정.

Colab:  python scripts/profile_mcts.py
로컬:   python -X utf8 scripts/profile_mcts.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from agents.alphazero.network import AlphaZeroNet, board_to_tensor
from agents.alphazero.mcts import MCTS
from env.gomoku import GomokuEnv

# ── 환경 ──────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BOARD_SIZE = 15
N_SIM      = 50

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed(fn, reps=1):
    """fn을 reps회 실행해 총 경과 시간(초) 반환. GPU sync 포함."""
    gpu_sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    gpu_sync()
    return time.perf_counter() - t


# ═══════════════════════════════════════════════════════════════════════════
# [1]  net.predict() 단일 호출 지연
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[1] net.predict() 단일 호출 지연  (batch=1)")
print("═" * 60)

board_sample = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
board_sample[7, 7] = 1

for label, nb, nf in [("3b×64ch  (로컬 검증)", 3, 64),
                       ("5b×128ch (Colab 설정)", 5, 128)]:
    net = AlphaZeroNet(board_size=BOARD_SIZE, n_res_blocks=nb, n_filters=nf).to(DEVICE)
    net.eval()
    # 워밍업
    for _ in range(20):
        net.predict(board_sample, 1, device=DEVICE)
    gpu_sync()

    N = 300
    elapsed = timed(lambda: net.predict(board_sample, 1, device=DEVICE), reps=N)
    ms = elapsed / N * 1000
    print(f"  {label}: {ms:.2f} ms/call")
    del net


# ═══════════════════════════════════════════════════════════════════════════
# [2]  배치 크기별 처리량 — GPU 활용도 확인
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[2] 배치 크기별 처리량  (5b×128ch)")
print("═" * 60)

net_large = AlphaZeroNet(board_size=BOARD_SIZE, n_res_blocks=5, n_filters=128).to(DEVICE)
net_large.eval()
t1 = board_to_tensor(board_sample, 1).unsqueeze(0).to(DEVICE)

for bsz in [1, 4, 16, 64, 256, 512]:
    batch = t1.expand(bsz, -1, -1, -1).contiguous()
    # 워밍업
    for _ in range(5):
        with torch.no_grad():
            net_large(batch)
    gpu_sync()

    reps = max(1, 200 // bsz)
    elapsed = timed(lambda: (None, net_large(batch))[0], reps=reps)  # noqa: B023
    per_sample_ms = elapsed / (reps * bsz) * 1000
    throughput    = reps * bsz / elapsed
    print(f"  batch={bsz:4d}: {per_sample_ms:.3f} ms/sample  |  {throughput:>8.0f} samples/s")


# ═══════════════════════════════════════════════════════════════════════════
# [3]  MCTS 1회 시간 분해: predict vs CPU 로직
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[3] MCTS get_action_probs() 시간 분해  (n_sim=50)")
print("═" * 60)

env = GomokuEnv(board_size=BOARD_SIZE, n_in_row=5, renju=True)

# net.predict에 타이머 래핑
_pred_total = [0.0]
_pred_count = [0]
_orig_predict = net_large.predict

def _predict_timed(board, current_player, device="cpu"):
    gpu_sync()
    t0 = time.perf_counter()
    result = _orig_predict(board, current_player, device=device)
    gpu_sync()
    _pred_total[0] += time.perf_counter() - t0
    _pred_count[0] += 1
    return result

net_large.predict = _predict_timed  # monkeypatch

mcts = MCTS(net_large, env, n_simulations=N_SIM, c_puct=1.5, device=DEVICE)
start_board = env.reset()

# 워밍업
mcts.get_action_probs(start_board, 1, temperature=1.0)
_pred_total[0] = 0.0
_pred_count[0] = 0

# 측정 (5회 평균)
REPS = 5
gpu_sync()
t0_total = time.perf_counter()
for _ in range(REPS):
    mcts.get_action_probs(start_board, 1, temperature=1.0)
gpu_sync()
t_total   = (time.perf_counter() - t0_total) / REPS
t_predict = _pred_total[0] / REPS
n_predict = _pred_count[0] // REPS
t_cpu_logic = t_total - t_predict

print(f"  MCTS 1회 총 시간    : {t_total*1000:.1f} ms")
print(f"  ├─ net.predict()    : {t_predict*1000:.1f} ms  "
      f"({n_predict}회, 평균 {t_predict/n_predict*1000:.2f} ms/call)  "
      f"← {t_predict/t_total*100:.0f}%")
print(f"  └─ MCTS Python 로직 : {t_cpu_logic*1000:.1f} ms  "
      f"(트리 탐색·expand·legal_actions 등)  "
      f"← {t_cpu_logic/t_total*100:.0f}%")

# PCIe 전송 오버헤드 단독 측정
t1_cpu = board_to_tensor(board_sample, 1).unsqueeze(0)
N_XFER = 500
gpu_sync()
t_xfer = time.perf_counter()
for _ in range(N_XFER):
    tmp = t1_cpu.to(DEVICE)
    _ = tmp.cpu()
gpu_sync()
xfer_ms = (time.perf_counter() - t_xfer) / N_XFER * 1000

print(f"\n  (참고) CPU↔GPU 전송만: {xfer_ms:.3f} ms/회  ({N_XFER}회 평균)")
print(f"  → predict 내 전송 비율: {xfer_ms / (t_predict/n_predict) * 100:.0f}% 추정")


# ═══════════════════════════════════════════════════════════════════════════
# [4]  iter 시간 외삽
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[4] 실측 기반 iter 시간 예측")
print("═" * 60)

avg_moves_per_game = 50
for sp in [5, 25]:
    mcts_per_iter = sp * avg_moves_per_game
    t_iter = t_total * mcts_per_iter
    print(f"  sp_games={sp:2d}: {mcts_per_iter}회 MCTS × {t_total*1000:.1f}ms = {t_iter:.0f}s/iter")

print("\n[결론]")
print(f"  predict가 MCTS의 {t_predict/t_total*100:.0f}%를 차지하고,")
print(f"  predict 1회 {t_predict/n_predict*1000:.2f}ms 중 전송이 약 {xfer_ms:.2f}ms ({xfer_ms/(t_predict/n_predict)*100:.0f}%).")
if t_predict / t_total > 0.7:
    print("  → 병목: net.predict() (GPU batch=1 전송+커널 오버헤드)")
    print("  → 해결 방향: (A) CPU 추론 + GPU 학습 분리,")
    print("               (B) 병렬 self-play + 배치 추론")
else:
    print("  → 병목: MCTS Python 로직 (트리 탐색 / legal_actions 등)")

"""
AlphaZero 학습 루프.

흐름: self-play 데이터 수집 → replay buffer 저장 → 미니배치 학습 → 반복
      주기적으로 랜덤봇 대전 평가 + 체크포인트 저장.
"""

import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from copy import deepcopy

from env.gomoku import GomokuEnv
from agents.alphazero.network import AlphaZeroNet, board_to_tensor
from agents.alphazero.mcts import MCTS


# ──────────────────────────────────────────────────────────────────────
# 데이터 증강 (8-fold 대칭)
# ──────────────────────────────────────────────────────────────────────

def augment_data(
    state_tensor: torch.Tensor,
    pi: np.ndarray,
    z: float,
    board_size: int,
) -> list[tuple[torch.Tensor, np.ndarray, float]]:
    """
    단일 (state, pi, z) 샘플을 4회전 × 2반전 = 8개로 확장.

    state_tensor : (3, H, W)
    pi           : (H*W,)
    z            : float
    """
    results = []
    st = state_tensor.numpy()   # (3, H, W)

    for flip in [False, True]:
        s = st.copy()
        p = pi.copy()
        if flip:
            s = np.flip(s, axis=2).copy()
            p = np.fliplr(p.reshape(board_size, board_size)).flatten()

        for _ in range(4):
            results.append((
                torch.from_numpy(s.copy()),
                p.copy(),
                z,
            ))
            # 90도 회전
            s = np.rot90(s, k=1, axes=(1, 2)).copy()
            p = np.rot90(p.reshape(board_size, board_size)).flatten()

    return results


# ──────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, maxlen: int = 50_000):
        self._buf: deque[tuple] = deque(maxlen=maxlen)

    def push(self, samples: list[tuple]) -> None:
        self._buf.extend(samples)

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────
# AlphaZero Trainer
# ──────────────────────────────────────────────────────────────────────

class AlphaZeroTrainer:
    """
    Parameters
    ----------
    board_size, n_in_row : 환경 설정
    n_res_blocks, n_filters : 네트워크 구조
    n_simulations : MCTS 시뮬레이션 횟수 (self-play)
    c_puct        : PUCT 계수
    n_iterations  : 전체 반복 (self-play → train) 횟수
    self_play_games_per_iter : 매 반복 당 self-play 게임 수
    batch_size    : 미니배치 크기
    train_steps_per_iter : 매 반복 당 학습 스텝 수
    eval_interval : 몇 반복마다 랜덤봇 평가
    eval_games    : 평가 게임 수
    lr, l2_reg    : 옵티마이저 하이퍼파라미터
    ckpt_dir      : 체크포인트 저장 디렉터리
    device        : 'cpu' 또는 'cuda'
    temperature_cutoff : 이 수 이전 착수는 τ=1, 이후는 τ→0 (greedy)
    ckpt_interval : 몇 반복마다 번호 포함 체크포인트 저장 (0이면 비활성)
    """

    def __init__(
        self,
        board_size: int = 6,
        n_in_row: int = 4,
        n_res_blocks: int = 3,
        n_filters: int = 64,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        n_iterations: int = 200,
        self_play_games_per_iter: int = 25,
        batch_size: int = 512,
        train_steps_per_iter: int = 5,
        eval_interval: int = 10,
        eval_games: int = 100,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
        ckpt_dir: str = "checkpoints",
        device: str = "cpu",
        temperature_cutoff: int = 12,
        ckpt_interval: int = 20,
    ):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.n_sim = n_simulations
        self.c_puct = c_puct
        self.n_iter = n_iterations
        self.sp_games = self_play_games_per_iter
        self.batch_size = batch_size
        self.train_steps = train_steps_per_iter
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.temp_cutoff = temperature_cutoff
        self.ckpt_interval = ckpt_interval

        self.env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
        self.net = AlphaZeroNet(
            board_size=board_size,
            n_res_blocks=n_res_blocks,
            n_filters=n_filters,
        ).to(device)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=l2_reg
        )
        self.buffer = ReplayBuffer()
        self.win_rates: list[float] = []

    # ── 공개 API ──────────────────────────────────────────────────────

    def train(self) -> list[float]:
        """n_iter 반복 학습 후 win_rates 반환."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        t_start = time.time()

        for iteration in range(1, self.n_iter + 1):
            t_iter = time.time()

            # self-play 데이터 수집
            new_data = self._collect_self_play()
            self.buffer.push(new_data)

            # 미니배치 학습
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_steps):
                    self._train_step()

            # 주기 평가
            if iteration % self.eval_interval == 0:
                win_rate = self._evaluate_vs_random()
                self.win_rates.append(win_rate)
                elapsed = time.time() - t_start
                iter_time = time.time() - t_iter
                print(
                    f"  iter {iteration:4d}/{self.n_iter}"
                    f"  buf={len(self.buffer):6d}"
                    f"  win={win_rate:.3f}"
                    f"  iter_t={iter_time:.1f}s"
                    f"  total={elapsed/60:.1f}m"
                )
            else:
                elapsed = time.time() - t_start
                iter_time = time.time() - t_iter
                print(
                    f"  iter {iteration:4d}/{self.n_iter}"
                    f"  buf={len(self.buffer):6d}"
                    f"  iter_t={iter_time:.1f}s"
                    f"  total={elapsed/60:.1f}m",
                    flush=True,
                )

            # 번호 포함 체크포인트
            if self.ckpt_interval > 0 and iteration % self.ckpt_interval == 0:
                path = os.path.join(
                    self.ckpt_dir,
                    f"az_{self.board_size}x{self.board_size}_iter{iteration:04d}.pt",
                )
                self.save(path)

        # 최종 저장
        final_path = os.path.join(
            self.ckpt_dir,
            f"az_{self.board_size}x{self.board_size}_final.pt",
        )
        self.save(final_path)
        return self.win_rates

    def save(self, path: str) -> None:
        torch.save(
            {
                "net_state": self.net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "win_rates": self.win_rates,
                "board_size": self.board_size,
                "n_in_row": self.n_in_row,
                "n_filters": self.net.policy_head[0].in_channels,
                "n_res_blocks": len(self.net.res_blocks),
            },
            path,
        )
        print(f"  → 저장: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.win_rates = ckpt.get("win_rates", [])

    # ── 내부 메서드 ───────────────────────────────────────────────────

    def _collect_self_play(self) -> list[tuple]:
        """sp_games 게임 self-play → 증강 포함 데이터 반환."""
        mcts = MCTS(
            self.net, self.env,
            n_simulations=self.n_sim,
            c_puct=self.c_puct,
            device=self.device,
        )
        all_data: list[tuple] = []
        for _ in range(self.sp_games):
            game_data = self._play_one_game(mcts)
            all_data.extend(game_data)
        return all_data

    def _play_one_game(self, mcts: MCTS) -> list[tuple]:
        """
        한 게임 self-play.
        반환: list of (state_tensor, pi, z) — 증강 포함
        """
        board = self.env.reset()
        player = self.env.current_player
        episode: list[tuple] = []  # (state_tensor, pi, player)
        move_count = 0
        done = False

        while not done:
            temperature = 1.0 if move_count < self.temp_cutoff else 1e-4
            pi = mcts.get_action_probs(board, player, temperature=temperature)

            state_t = board_to_tensor(board, player)
            episode.append((state_t, pi.copy(), player))

            action = int(np.random.choice(len(pi), p=pi))
            board, reward, done, info = self.env.step(action)
            if not done:
                player = self.env.current_player
            move_count += 1

        # reward=1 이면 마지막 착수자 승 (env.current_player = 승자, 종료 시 비전환)
        winner = episode[-1][2] if reward == 1 else 0

        # z 값 부여 후 증강
        result: list[tuple] = []
        for state_t, pi, p in episode:
            if winner == 0:
                z = 0.0
            elif p == winner:
                z = 1.0
            else:
                z = -1.0
            result.extend(augment_data(state_t, pi, z, self.board_size))

        return result

    def _train_step(self) -> None:
        """배치 샘플링 → 정책/가치 손실 역전파."""
        batch = self.buffer.sample(self.batch_size)
        state_ts, pis, zs = zip(*batch)

        states = torch.stack(state_ts).to(self.device)          # (B, 3, H, W)
        targets_pi = torch.tensor(
            np.array(pis), dtype=torch.float32, device=self.device
        )                                                         # (B, N)
        targets_z = torch.tensor(
            np.array(zs), dtype=torch.float32, device=self.device
        ).unsqueeze(1)                                            # (B, 1)

        self.net.train()
        log_policy, value = self.net(states)

        # 정책 손실: -sum(pi * log_pi)  (KL divergence 항)
        policy_loss = -(targets_pi * log_policy).sum(dim=1).mean()
        # 가치 손실: MSE
        value_loss = ((value - targets_z) ** 2).mean()
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _evaluate_vs_random(self) -> float:
        """AlphaZero vs 랜덤봇 eval_games 판. 흑/백 각 절반씩."""
        from agents.random_agent import RandomAgent

        random_agent = RandomAgent()
        wins = 0
        half = self.eval_games // 2

        def az_action(board, player):
            mcts = MCTS(self.net, self.env,
                        n_simulations=self.n_sim,
                        c_puct=self.c_puct,
                        device=self.device)
            pi = mcts.get_action_probs(board, player, temperature=1e-4)
            return int(np.argmax(pi))

        def rnd_action(board, player):
            return random_agent.select_action(board, self.env.legal_actions(board))

        # AlphaZero = BLACK
        for _ in range(half):
            board = self.env.reset()
            player = self.env.current_player
            done = False
            while not done:
                action = az_action(board, player) if player == 1 else rnd_action(board, player)
                board, reward, done, info = self.env.step(action)
                if not done:
                    player = self.env.current_player
            if reward == 1 and self.env.current_player == 1:
                wins += 1

        # AlphaZero = WHITE
        for _ in range(half):
            board = self.env.reset()
            player = self.env.current_player
            done = False
            while not done:
                action = az_action(board, player) if player == -1 else rnd_action(board, player)
                board, reward, done, info = self.env.step(action)
                if not done:
                    player = self.env.current_player
            if reward == 1 and self.env.current_player == -1:
                wins += 1

        return wins / self.eval_games

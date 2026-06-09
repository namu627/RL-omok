"""
Phase 1 — 테이블 Q-러닝 (2인 zero-sum self-play, Monte Carlo 리턴)

핵심 아이디어:
  - 상태 정규화: board * current_player → 내 돌=+1, 상대 돌=-1 로 통일.
    → 흑·백 플레이어가 같은 Q-테이블 공유.
  - Monte Carlo 리턴: 한 판이 끝난 뒤 그 판의 모든 (s,a)에 할인된 최종 보상 소급 적용.
    G_t = γ^(T−1−t) × (±1)   (종료 시점에서 t 스텝 전 수에 대한 할인 리턴)
  - TD(0) minimax 부트스트랩을 쓰지 않는 이유:
    Q 초기값이 0이면 비종료 스텝의 타깃도 0이 되어 사실상 마지막 수 2개만 업데이트됨.
    상태 재방문이 드문 6×6 보드에서 Q-값이 전파되지 않음.
"""

import random
import pickle
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env.gomoku import GomokuEnv, BLACK, WHITE, EMPTY
from agents.random_agent import RandomAgent


class QLearningAgent:
    """
    Q-테이블 기반 에이전트.

    state_key: tuple[int, ...]  (보드 36칸 × 현재 플레이어 부호)
    Q-테이블: dict[(state_key, action)] → float
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95):
        self.q_table: dict[tuple, float] = {}
        self.alpha = alpha   # 학습률
        self.gamma = gamma   # 할인율

    # ------------------------------------------------------------------
    # 상태 표현
    # ------------------------------------------------------------------

    def state_key(self, board: np.ndarray, player: int) -> tuple:
        """
        보드를 현재 플레이어(player) 시점으로 정규화.
        두 플레이어 어느 쪽이든 '내 돌=+1, 상대=−1' 로 보이게 된다.
        """
        return tuple(int(v) for v in (board * player).flatten())

    # ------------------------------------------------------------------
    # 행동 선택 (ε-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state_key: tuple, legal_actions: list[int],
                      epsilon: float) -> int:
        """
        ε 확률로 무작위 탐험, (1−ε) 확률로 Q-최대 행동 선택.
        Q 값이 같은 행동이 여럿이면 무작위 선택(동률 다양성 보존).
        """
        if random.random() < epsilon:
            return random.choice(legal_actions)
        q_vals = [self.q_table.get((state_key, a), 0.0) for a in legal_actions]
        max_q = max(q_vals)
        best = [a for a, q in zip(legal_actions, q_vals) if q == max_q]
        return random.choice(best)

    # ------------------------------------------------------------------
    # Q-테이블 업데이트 (벨만 방정식)
    # ------------------------------------------------------------------

    def update(self, state_key: tuple, action: int, target: float) -> None:
        """
        Q(s,a) ← Q(s,a) + α·(target − Q(s,a))

        target: Monte Carlo 리턴  G_t = γ^(T−1−t) × (±1)
        """
        current_q = self.q_table.get((state_key, action), 0.0)
        self.q_table[(state_key, action)] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table,
                         "alpha": self.alpha,
                         "gamma": self.gamma}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]


# ======================================================================
# Self-play 학습 루프 (Monte Carlo 리턴)
# ======================================================================

def self_play_train(
    env: GomokuEnv,
    agent: QLearningAgent,
    n_episodes: int = 30_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_ratio: float = 0.8,   # 전체의 80% 동안 선형 감소
    eval_interval: int = 300,
    eval_games: int = 100,
) -> list[float]:
    """
    Monte Carlo 리턴 기반 Q-러닝 self-play.

    한 판이 끝나면 그 판의 모든 (state_key, action)에 할인된 최종 보상 소급 적용.
      승자 수순: G_t = γ^(T−1−t) × (+1)
      패자 수순: G_t = γ^(T−1−t) × (−1)
      무승부:   G_t = 0

    2인 zero-sum 처리:
      - 상태는 항상 착수 플레이어 시점으로 정규화(board × player)
      - 두 플레이어가 같은 Q-테이블 공유
    """
    rng_agent = RandomAgent()
    win_rates: list[float] = []
    decay_steps = int(n_episodes * eps_decay_ratio)

    for episode in range(n_episodes):

        # ε 선형 감소
        if episode < decay_steps:
            epsilon = eps_start + (eps_end - eps_start) * (episode / decay_steps)
        else:
            epsilon = eps_end

        state = env.reset()
        # 궤적: [(player, state_key, action), ...]
        trajectory: list[tuple] = []

        while True:
            player = env.current_player
            sk = agent.state_key(state, player)
            legal = env.legal_actions(state)
            action = agent.select_action(sk, legal, epsilon)

            next_state, _, done, info = env.step(action)
            trajectory.append((player, sk, action))
            state = next_state

            if done:
                winner = info["winner"]
                break

        # Monte Carlo 소급 업데이트
        # reversed → 종료에서 가까울수록 t=0, 멀수록 t가 큼
        T = len(trajectory)
        for t, (player, sk, action) in enumerate(reversed(trajectory)):
            if winner == EMPTY:
                G = 0.0
            elif player == winner:
                G = agent.gamma ** t   # 종료에서 t 스텝 전 = γ^t
            else:
                G = -(agent.gamma ** t)
            agent.update(sk, action, G)

        # 승률 평가
        if (episode + 1) % eval_interval == 0:
            wr = evaluate_vs_random(env, agent, rng_agent, eval_games)
            win_rates.append(wr)
            print(
                f"  ep {episode + 1:>6}/{n_episodes}"
                f"  ε={epsilon:.3f}"
                f"  승률={wr:.3f}"
                f"  Q항목={len(agent.q_table):,}"
            )

    return win_rates


# ======================================================================
# 평가
# ======================================================================

def evaluate_vs_random(
    env: GomokuEnv,
    agent: QLearningAgent,
    random_agent: RandomAgent,
    n_games: int = 100,
) -> float:
    """
    에이전트 vs 랜덤봇 n_games판 진행, 승률 반환.
    절반은 흑(선공), 절반은 백(후공)으로 대국해 색깔 편향 제거.
    """
    wins = 0
    half = n_games // 2

    for i in range(n_games):
        agent_color = BLACK if i < half else WHITE
        state = env.reset()

        while True:
            p = env.current_player
            legal = env.legal_actions(state)

            if p == agent_color:
                sk = agent.state_key(state, p)
                action = agent.select_action(sk, legal, epsilon=0.0)
            else:
                action = random_agent.select_action(state, legal)

            state, _, done, info = env.step(action)
            if done:
                if info["winner"] == agent_color:
                    wins += 1
                break

    return wins / n_games

"""
Phase 2 — Policy Gradient: REINFORCE → Actor-Critic (NumPy 구현)

[ 모델 구조 ]
  입력:  board × current_player → 36-dim 벡터 (내 돌=+1, 상대=-1)
         Phase 1 Q-러닝과 동일한 상태 정규화.
  정책:  π(a|s) = softmax( W_π @ s + b_π )  합법 수만 마스킹
  가치:  V(s)   = w_v  @ s + b_v            (Actor-Critic 시에만 사용)

[ REINFORCE ]
  - 한 판 궤적 수집 → 종료 후 모든 수에 할인 리턴 G_t 부여
  - 정책 경사:  θ ← θ + α · G_t · ∇θ log π(a_t|s_t)
  - ∇θ log π(a|s) = (e_a − π_masked) ⊗ s  (softmax 역전파)

[ Actor-Critic (MC baseline) ]
  - REINFORCE + V(s) baseline 추가
  - 어드밴티지:  A_t = G_t − V(s_t)
  - 정책 업데이트: θ ← θ + α_π · A_t · ∇θ log π
  - 가치 업데이트: w_v ← w_v + α_v · (G_t − V(s_t)) · s

[ 2인 zero-sum 처리 ]
  - 승자 수 t번째:  G_t = γ^t × (+1)   (Phase 1 Q-러닝과 동일)
  - 패자 수 t번째:  G_t = γ^t × (−1)
  - 무승부:         G_t = 0
  - t=0 은 마지막 수 (종료에서 가장 가까움), t 증가 → 더 앞 수
"""

import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env.gomoku import GomokuEnv, BLACK, WHITE, EMPTY
from agents.random_agent import RandomAgent


# ======================================================================
# 에이전트
# ======================================================================

class PolicyGradientAgent:
    """
    선형 softmax 정책 (+ 선택적 선형 가치 함수).

    파라미터
    --------
    board_size   : 보드 한 변 길이
    lr_policy    : 정책 학습률  α_π
    lr_value     : 가치 학습률  α_v  (use_baseline=True 시만 사용)
    gamma        : 할인율 γ
    use_baseline : False → REINFORCE / True → Actor-Critic
    """

    def __init__(
        self,
        board_size: int = 6,
        lr_policy: float = 1e-3,
        lr_value: float = 5e-3,
        gamma: float = 0.95,
        use_baseline: bool = False,
    ):
        self.board_size = board_size
        self.n = board_size ** 2
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.gamma = gamma
        self.use_baseline = use_baseline

        # 정책 파라미터: W_π (n×n), b_π (n,)
        # 작은 랜덤 초기화 → 초기 정책이 거의 균등하되 대칭 깨짐
        rng = np.random.default_rng(seed=42)
        self.W_pi: np.ndarray = rng.standard_normal((self.n, self.n)) * 0.01
        self.b_pi: np.ndarray = np.zeros(self.n)

        # 가치 파라미터: w_v (n,), b_v (scalar)
        self.w_v: np.ndarray = np.zeros(self.n)
        self.b_v: float = 0.0

    # ------------------------------------------------------------------
    # 상태 특징 벡터
    # ------------------------------------------------------------------

    def features(self, board: np.ndarray, player: int) -> np.ndarray:
        """board × player → (n,) float32.  Q-러닝과 동일한 정규화."""
        return (board * player).flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # 정책: softmax 확률 벡터
    # ------------------------------------------------------------------

    def _softmax_masked(self, feat: np.ndarray, legal: list[int]) -> np.ndarray:
        """
        합법 수에 대해서만 softmax 계산.
        반환: (n,) 벡터, 불법 수 위치 = 0.
        """
        logits = self.W_pi @ feat + self.b_pi       # (n,)
        idx = np.array(legal, dtype=np.int32)
        z = logits[idx]
        z = z - z.max()                              # 수치 안정성 (overflow 방지)
        exp_z = np.exp(z)
        prob_legal = exp_z / exp_z.sum()

        full = np.zeros(self.n, dtype=np.float64)
        full[idx] = prob_legal
        return full

    def select_action(
        self,
        board: np.ndarray,
        player: int,
        legal: list[int],
        greedy: bool = False,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        행동 선택.

        Parameters
        ----------
        greedy : True → 확률 최대 행동 (평가용), False → 확률 비례 샘플링 (학습용)

        Returns
        -------
        (action, prob_vec, feat)
          prob_vec : (n,) 확률 벡터 — 나중에 gradient 계산에 재사용
          feat     : (n,) 상태 벡터 — 나중에 gradient 계산에 재사용
        """
        feat = self.features(board, player)
        prob = self._softmax_masked(feat, legal)
        if greedy:
            action = int(np.argmax(prob))
        else:
            action = int(np.random.choice(self.n, p=prob))
        return action, prob, feat

    # ------------------------------------------------------------------
    # 가치 함수
    # ------------------------------------------------------------------

    def value(self, feat: np.ndarray) -> float:
        return float(np.dot(self.w_v, feat) + self.b_v)

    # ------------------------------------------------------------------
    # 파라미터 업데이트 (에피소드 끝 후 한 수씩 호출)
    # ------------------------------------------------------------------

    def update(
        self,
        feat: np.ndarray,
        action: int,
        prob: np.ndarray,
        G: float,
    ) -> None:
        """
        1 스텝 업데이트.

        REINFORCE:    advantage = G
        Actor-Critic: advantage = G − V(s),  V(s) 도 함께 업데이트

        정책 경사:  ∇θ log π(a|s) = e_a − π
        → W_π += α_π · advantage · (e_a − π) ⊗ s
        → b_π += α_π · advantage · (e_a − π)
        """
        if self.use_baseline:
            v_s = self.value(feat)          # ① V(s) 를 업데이트 전에 읽음
            advantage = G - v_s
            delta = G - v_s                 # TD 오차 (MC 버전) = advantage
            self.w_v += self.lr_value * delta * feat
            self.b_v += self.lr_value * delta
        else:
            advantage = G

        # 정책 경사: ∇log π = e_a − π
        grad = -prob.copy()                 # (n,)  −π
        grad[action] += 1.0                 #       +e_a

        self.W_pi += self.lr_policy * advantage * np.outer(grad, feat)
        self.b_pi += self.lr_policy * advantage * grad

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "W_pi": self.W_pi, "b_pi": self.b_pi,
                "w_v": self.w_v,   "b_v": self.b_v,
                "board_size": self.board_size,
                "use_baseline": self.use_baseline,
                "lr_policy": self.lr_policy,
                "lr_value": self.lr_value,
                "gamma": self.gamma,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        for k in ("W_pi", "b_pi", "w_v", "b_v",
                  "board_size", "use_baseline", "lr_policy", "lr_value", "gamma"):
            if k in d:
                setattr(self, k, d[k])


# ======================================================================
# Self-play 학습 루프
# ======================================================================

def self_play_train(
    env: GomokuEnv,
    agent: PolicyGradientAgent,
    n_episodes: int = 30_000,
    eval_interval: int = 300,
    eval_games: int = 300,
) -> list[float]:
    """
    Monte Carlo 리턴 기반 Policy Gradient self-play.

    ─ 궤적 수집 ─────────────────────────────────────────────────────
    매 에피소드:  (player, action, prob_copy, feat_copy) 를 스텝마다 기록.

    ─ 보상 처리 (2인 zero-sum) ──────────────────────────────────────
    reversed(trajectory) 로 순회 → t=0: 마지막 수, t=T-1: 첫 번째 수
      승자 수: G_t = γ^t  (+1)
      패자 수: G_t = γ^t  (−1)
      무승부:  G_t = 0
    Phase 1 Q-러닝과 동일한 부호 처리.

    ─ 파라미터 업데이트 ─────────────────────────────────────────────
    REINFORCE:    advantage = G_t
    Actor-Critic: advantage = G_t − V(s_t),  V(s_t) 도 업데이트
    """
    rng_agent = RandomAgent()
    win_rates: list[float] = []
    mode_str = "AC" if agent.use_baseline else "RF"

    for episode in range(n_episodes):

        state = env.reset()
        # 궤적: (player, action, prob_snapshot, feat_snapshot)
        traj: list[tuple] = []

        while True:
            player = env.current_player
            legal = env.legal_actions(state)
            action, prob, feat = agent.select_action(state, player, legal)

            next_state, _, done, info = env.step(action)
            # prob, feat 을 copy 해야 이후 업데이트가 이전 값을 덮어쓰지 않음
            traj.append((player, action, prob.copy(), feat.copy()))
            state = next_state

            if done:
                winner = info["winner"]
                break

        # ── Monte Carlo 소급 업데이트 ──────────────────────────────
        for t, (player, action, prob, feat) in enumerate(reversed(traj)):
            if winner == EMPTY:
                G = 0.0
            elif player == winner:
                G = agent.gamma ** t
            else:
                G = -(agent.gamma ** t)

            agent.update(feat, action, prob, G)

        # ── 평가 ──────────────────────────────────────────────────
        if (episode + 1) % eval_interval == 0:
            wr = evaluate_vs_random(env, agent, rng_agent, eval_games)
            win_rates.append(wr)
            print(
                f"  [{mode_str}] ep {episode + 1:>6}/{n_episodes}"
                f"  승률={wr:.3f}"
            )

    return win_rates


# ======================================================================
# 평가 (Q-러닝과 동일한 인터페이스)
# ======================================================================

def evaluate_vs_random(
    env: GomokuEnv,
    agent: PolicyGradientAgent,
    random_agent: RandomAgent,
    n_games: int = 300,
) -> float:
    """
    에이전트 vs 랜덤봇  n_games판 진행, 승률 반환.
    절반은 흑(선공), 절반은 백(후공)으로 진행해 색깔 편향 제거.
    평가 시 greedy=True (확률 최대 행동 선택).
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
                action, _, _ = agent.select_action(state, p, legal, greedy=True)
            else:
                action = random_agent.select_action(state, legal)

            state, _, done, info = env.step(action)
            if done:
                if info["winner"] == agent_color:
                    wins += 1
                break

    return wins / n_games

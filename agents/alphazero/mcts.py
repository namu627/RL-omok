"""
PUCT 기반 Monte Carlo Tree Search (AlphaZero 스타일).

핵심 규칙
---------
1. child.current_player = -parent.current_player  (항상 강제 교대)
2. Q[node] = 해당 노드의 current_player 시점에서의 기대 승률
3. 부모 시점에서 자식을 선택할 때: score = -child.Q + U  (부호 반전)
4. 백업: 매 레벨마다 value = -value  (상대 시점으로 전환)
5. 종료 노드의 리프 값: parent.current_player가 이겼으면 자식 시점 = -1,
                         무승부 = 0, parent가 지면(상대가 이기면) = +1

렌주 금수 처리 (방안 B):
  - 루트 노드: env.legal_actions()로 금수 제외 — 실제 착수는 항상 합법
  - 트리 내부 노드: _raw_legal_actions()로 빈 칸 전체 허용 (금수 필터 없음)
    → 흑이 트리 내부에서 금수 자리를 두면 env.step()이 reward=-1 반환
    → _expand가 해당 자식을 winner=-BLACK 종료 노드로 생성
    → 역전파로 Q값이 -1 방향으로 수렴하여 MCTS가 자연히 금수를 회피
  - legal_actions()는 루트에서 1회만 호출 (방안 A: 중복 호출 제거)
"""

import math
import numpy as np

from env.gomoku import GomokuEnv
from agents.alphazero.network import AlphaZeroNet, board_to_tensor


# ──────────────────────────────────────────────────────────────────────
# MCTS 노드
# ──────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = (
        "parent", "action",
        "children",
        "N", "W", "Q", "P",
        "board", "current_player",
        "is_terminal", "winner",
    )

    def __init__(
        self,
        parent: "MCTSNode | None",
        action: int | None,
        board: np.ndarray,
        current_player: int,
        prior: float = 0.0,
        is_terminal: bool = False,
        winner: int = 0,
    ):
        self.parent = parent
        self.action = action
        self.children: dict[int, "MCTSNode"] = {}

        self.N = 0      # 방문 횟수
        self.W = 0.0    # 누적 가치
        self.Q = 0.0    # 평균 가치 (= W / N)  — 이 노드의 current_player 시점
        self.P = prior  # 사전 확률 (네트워크 정책 헤드)

        self.board = board
        self.current_player = current_player
        self.is_terminal = is_terminal
        self.winner = winner    # 0=진행중/무 1=BLACK -1=WHITE

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(self, c_puct: float, parent_N: int) -> float:
        """부모 시점의 PUCT 점수. 부모가 이 자식을 얼마나 원하는가."""
        U = c_puct * self.P * math.sqrt(parent_N) / (1 + self.N)
        return -self.Q + U   # 자식.Q는 자식 플레이어 시점 → 부모 시점 반전


# ──────────────────────────────────────────────────────────────────────
# MCTS
# ──────────────────────────────────────────────────────────────────────

class MCTS:
    """
    Parameters
    ----------
    network     : AlphaZeroNet  (eval 모드로 사용)
    env         : GomokuEnv     (시뮬레이션용 — 내부에서 deepcopy)
    n_simulations : 루트 당 시뮬레이션 횟수
    c_puct      : 탐색/활용 균형 계수
    device      : 'cpu' 또는 'cuda'
    """

    def __init__(
        self,
        network: AlphaZeroNet,
        env: GomokuEnv,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        device: str = "cpu",
    ):
        self.net = network
        self.env = env
        self.n_sim = n_simulations
        self.c_puct = c_puct
        self.device = device

    # ── 공개 인터페이스 ───────────────────────────────────────────────

    def get_action_probs(
        self,
        board: np.ndarray,
        current_player: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        루트 상태에서 n_simulations 회 시뮬레이션 실행 후
        방문 횟수로부터 착수 확률 분포를 반환.

        temperature : 1.0 = 정상 탐색, →0 = greedy (가장 방문 많은 칸)
        반환: (board_size²,) numpy float32
        """
        root = self._build_root(board, current_player)

        for _ in range(self.n_sim):
            self._simulate(root)

        return self._visit_counts_to_probs(root, temperature)

    # ── 내부 메서드 ───────────────────────────────────────────────────

    def _build_root(self, board: np.ndarray, current_player: int) -> MCTSNode:
        """루트 노드를 생성하고 네트워크로 사전 확률을 채운다."""
        root = MCTSNode(
            parent=None, action=None,
            board=board.copy(),
            current_player=current_player,
        )
        policy, _ = self.net.predict(board, current_player, device=self.device)
        # 합법 착수 외 마스킹
        legal = self.env.legal_actions(board)
        mask = np.zeros_like(policy)
        mask[legal] = 1.0
        policy = policy * mask
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            policy[legal] = 1.0 / len(legal)

        self._expand(root, policy, legal)   # legal 재전달 (재계산 없음)
        return root

    def _simulate(self, root: MCTSNode) -> None:
        """선택 → 평가·확장 → 역전파 한 사이클."""
        node = root
        path: list[MCTSNode] = [node]

        # ── 선택 ──────────────────────────────────────────────────────
        while not node.is_leaf() and not node.is_terminal:
            node = self._select_child(node)
            path.append(node)

        # ── 평가·확장 ────────────────────────────────────────────────
        if node.is_terminal:
            # 종료 노드: 부모가 이겼으면 자식 시점 = -1
            if node.winner == 0:
                leaf_value = 0.0          # 무승부
            elif node.winner == node.parent.current_player:
                leaf_value = -1.0         # 부모(승자) 시점에서 자식은 패자
            else:
                leaf_value = 1.0          # 부모가 진 경우
        else:
            # 리프: 네트워크 평가 + 확장
            policy, value = self.net.predict(
                node.board, node.current_player, device=self.device
            )
            # 트리 내부: 금수 필터 없이 빈 칸 전체를 후보로 삼는다 (방안 B).
            # 금수 착수 시 _apply_action → env.step()이 reward=-1 반환 → 역전파로 회피 학습.
            raw_legal = self._raw_legal_actions(node.board)
            mask = np.zeros_like(policy)
            mask[raw_legal] = 1.0
            policy = policy * mask
            s = policy.sum()
            if s > 0:
                policy /= s
            else:
                policy[raw_legal] = 1.0 / len(raw_legal)

            self._expand(node, policy, raw_legal)   # legal 전달 (방안 A: 재계산 없음)
            leaf_value = float(value)   # 이 노드의 current_player 시점

        # ── 역전파 ────────────────────────────────────────────────────
        self._backup(path, leaf_value)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """PUCT 점수가 가장 높은 자식 선택."""
        best_score = -float("inf")
        best_child = None
        parent_N = node.N
        for child in node.children.values():
            score = child.ucb_score(self.c_puct, parent_N)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _raw_legal_actions(self, board: np.ndarray) -> list[int]:
        """빈 칸 전체 인덱스 반환 (렌주 금수 필터 없음). 트리 내부 노드 전용."""
        n = self.env.board_size
        return [int(r) * n + int(c) for r, c in np.argwhere(board == 0)]

    def _expand(self, node: MCTSNode, policy: np.ndarray, legal: list[int]) -> None:
        """합법 착수마다 자식 노드 생성. 종료 상태 감지도 수행."""
        for action in legal:
            new_board, reward, done, info = self._apply_action(
                node.board, action, node.current_player
            )
            # 자식의 current_player 는 항상 반전
            child_player = -node.current_player

            winner = 0
            if done:
                if reward > 0:
                    winner = node.current_player    # 착수한 쪽이 이김
                elif reward < 0:
                    winner = -node.current_player   # 착수한 쪽이 짐 (금수 등)

            child = MCTSNode(
                parent=node,
                action=action,
                board=new_board,
                current_player=child_player,
                prior=float(policy[action]),
                is_terminal=done,
                winner=winner,
            )
            node.children[action] = child

    def _apply_action(
        self, board: np.ndarray, action: int, player: int
    ) -> tuple:
        """보드 배열만 복사해 착수 결과 계산 — deepcopy(env) 제거."""
        return self.env.step_board(board, action, player)

    def _backup(self, path: list[MCTSNode], leaf_value: float) -> None:
        """
        leaf_value = 경로 끝 노드의 current_player 시점 가치.
        올라가면서 매 레벨 부호를 반전해 각 노드 자신의 시점으로 변환.
        """
        value = leaf_value
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # 상위 노드는 반대 시점

    def _visit_counts_to_probs(
        self, root: MCTSNode, temperature: float
    ) -> np.ndarray:
        n = self.env.board_size ** 2
        counts = np.zeros(n, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.N

        if temperature < 1e-3:
            # greedy: 가장 방문 많은 칸에만 확률 1
            best = int(counts.argmax())
            probs = np.zeros(n, dtype=np.float32)
            probs[best] = 1.0
            return probs

        counts = counts ** (1.0 / temperature)
        return counts / counts.sum()

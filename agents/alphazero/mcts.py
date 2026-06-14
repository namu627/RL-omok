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
    → 흑이 트리 내부에서 금수 자리를 두면 step_board가 reward=-1 반환
    → _create_child가 해당 자식을 winner=-BLACK 종료 노드로 생성
    → 역전파로 Q값이 -1 방향으로 수렴하여 MCTS가 자연히 금수를 회피

Lazy Expansion (성능):
  - 자식 노드를 전부 미리 만들지 않고 PUCT가 처음 선택할 때 1개씩 생성.
  - 기존 eager 방식: _expand 1회 = step_board × 220회 (15×15 빈 칸 수)
  - lazy 방식: _create_child 1회 = step_board × 1회
  - step_board(BLACK) = classify_move 1회 → 40~220× 감소 효과.
  - 수학적 동등성: PUCT 점수 = eager와 동일 (미방문 자식: Q=0, N=0으로 스코어링)
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
        "_policy",          # (board_size²,) float32 — predict 후 저장. None이면 미평가 리프.
        "_legal_actions",   # list[int] — 이 노드에서 합법 착수 목록
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
        self._policy: np.ndarray | None = None   # 평가 후 세팅
        self._legal_actions: list[int] = []      # 평가 후 세팅

        self.N = 0      # 방문 횟수
        self.W = 0.0    # 누적 가치
        self.Q = 0.0    # 평균 가치 (= W / N)  — 이 노드의 current_player 시점
        self.P = prior  # 사전 확률 (부모의 정책 헤드)

        self.board = board
        self.current_player = current_player
        self.is_terminal = is_terminal
        self.winner = winner    # 0=진행중/무 1=BLACK -1=WHITE

    def is_leaf(self) -> bool:
        """_policy가 None이면 아직 네트워크 평가를 받지 않은 리프."""
        return self._policy is None

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
    env         : GomokuEnv     (시뮬레이션용)
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
        """
        root = self._build_root(board, current_player)

        for _ in range(self.n_sim):
            self._simulate(root)

        return self._visit_counts_to_probs(root, temperature)

    # ── 내부 메서드 ───────────────────────────────────────────────────

    def _build_root(self, board: np.ndarray, current_player: int) -> MCTSNode:
        """루트 노드를 생성하고 네트워크 정책을 저장한다 (lazy expansion)."""
        root = MCTSNode(
            parent=None, action=None,
            board=board.copy(),
            current_player=current_player,
        )
        policy, _ = self.net.predict(board, current_player, device=self.device)
        legal = self.env.legal_actions(board)
        mask = np.zeros_like(policy)
        mask[legal] = 1.0
        policy = policy * mask
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            policy[legal] = 1.0 / len(legal)
        root._policy = policy
        root._legal_actions = legal
        return root

    def _simulate(self, root: MCTSNode) -> None:
        """선택 → 평가·확장 → 역전파 한 사이클 (래퍼)."""
        leaf, path = self._select_leaf(root)

        if leaf.is_terminal:
            leaf_value = self._terminal_value(leaf)
            self._backup(path, leaf_value)
        else:
            policy, value = self.net.predict(
                leaf.board, leaf.current_player, device=self.device
            )
            self._expand_and_backup(leaf, path, policy, value)

    def _select_leaf(self, root: MCTSNode) -> tuple["MCTSNode", list["MCTSNode"]]:
        """루트에서 PUCT로 미평가 리프 또는 종료 노드까지 내려가 (leaf, path) 반환.

        Lazy Expansion: 미방문 액션이 선택되면 그 자리에서 자식 1개를 생성해 반환.
        평가·역전파는 하지 않는다.
        """
        node = root
        path: list[MCTSNode] = [node]

        while not node.is_leaf() and not node.is_terminal:
            action = self._select_action(node)
            if action not in node.children:
                # 처음 선택된 액션 → 자식 1개 생성 (step_board 1회)
                child = self._create_child(node, action)
                node.children[action] = child
                path.append(child)
                return child, path
            node = node.children[action]
            path.append(node)

        return node, path

    def _terminal_value(self, leaf: MCTSNode) -> float:
        """종료 노드의 leaf_value 계산 (leaf의 current_player 시점)."""
        if leaf.winner == 0:
            return 0.0
        if leaf.winner == leaf.parent.current_player:
            return -1.0   # 부모(착수자)가 이김 → 자식은 패자
        return 1.0        # 부모가 짐 (금수 자폭 등) → 자식에게 유리

    def _expand_and_backup(
        self,
        leaf: MCTSNode,
        path: list["MCTSNode"],
        policy: np.ndarray,
        value: float,
    ) -> None:
        """비종료 리프에 정책을 저장하고 역전파.

        Lazy Expansion: 자식을 미리 생성하지 않는다.
        _select_leaf가 다음 방문 시 _select_action → _create_child로 1개씩 생성.

        policy : net.predict_batch 등 외부 평가기가 반환한 raw (board_size²,) 배열.
                 마스킹·정규화를 여기서 수행한다.
        value  : 이 노드의 current_player 시점 가치 ∈ [-1, 1].
        """
        raw_legal = self._raw_legal_actions(leaf.board)
        mask = np.zeros_like(policy)
        mask[raw_legal] = 1.0
        policy = policy * mask
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            policy[raw_legal] = 1.0 / len(raw_legal)

        leaf._policy = policy
        leaf._legal_actions = raw_legal
        self._backup(path, float(value))

    def _select_action(self, node: MCTSNode) -> int:
        """PUCT 점수 최대 액션 반환.

        방문한 자식(N>0)은 child.ucb_score로, 미방문 액션은 prior × c_puct × sqrt(N)으로 스코어링.
        수학적으로 eager expansion의 _select_child와 동일한 결과를 반환한다.
        """
        best_score = -float("inf")
        best_action: int = node._legal_actions[0]
        parent_N = node.N
        sqrt_N = math.sqrt(parent_N)
        c = self.c_puct
        p = node._policy
        visited = node.children

        for action in node._legal_actions:
            if action in visited:
                child = visited[action]
                score = -child.Q + c * child.P * sqrt_N / (1 + child.N)
            else:
                # 미방문: Q=0, N=0 → PUCT = prior × c × sqrt(parent_N)
                score = c * p[action] * sqrt_N
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _create_child(self, parent: MCTSNode, action: int) -> MCTSNode:
        """지정 액션으로 자식 1개 생성 (step_board 1회 호출)."""
        new_board, reward, done, info = self._apply_action(
            parent.board, action, parent.current_player
        )
        child_player = -parent.current_player
        winner = 0
        if done:
            if reward > 0:
                winner = parent.current_player
            elif reward < 0:
                winner = -parent.current_player
        return MCTSNode(
            parent=parent,
            action=action,
            board=new_board,
            current_player=child_player,
            prior=float(parent._policy[action]),
            is_terminal=done,
            winner=winner,
        )

    def _raw_legal_actions(self, board: np.ndarray) -> list[int]:
        """빈 칸 전체 인덱스 반환 (렌주 금수 필터 없음). 트리 내부 노드 전용."""
        n = self.env.board_size
        return [int(r) * n + int(c) for r, c in np.argwhere(board == 0)]

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
            value = -value

    def _visit_counts_to_probs(
        self, root: MCTSNode, temperature: float
    ) -> np.ndarray:
        n = self.env.board_size ** 2
        counts = np.zeros(n, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.N

        if temperature < 1e-3:
            best = int(counts.argmax())
            probs = np.zeros(n, dtype=np.float32)
            probs[best] = 1.0
            return probs

        counts = counts ** (1.0 / temperature)
        return counts / counts.sum()

    # ── 하위 호환 / 성능 비교용 ───────────────────────────────────────

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """기존 코드 호환용 — 방문한 자식만 스코어링한다.
        새 코드는 _select_action + _create_child 조합을 사용한다."""
        best_score = -float("inf")
        best_child = None
        parent_N = node.N
        for child in node.children.values():
            score = child.ucb_score(self.c_puct, parent_N)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: MCTSNode, policy: np.ndarray, legal: list[int]) -> None:
        """Eager expansion — 합법 착수마다 자식 노드를 한꺼번에 생성.

        성능 주의: step_board를 len(legal)회 호출 (BLACK이면 classify_move도 동반).
        새 코드에서는 루트에 _policy/_legal_actions만 저장하는 방식을 권장.
        테스트 호환성을 위해 유지.
        """
        for action in legal:
            new_board, reward, done, info = self._apply_action(
                node.board, action, node.current_player
            )
            child_player = -node.current_player
            winner = 0
            if done:
                if reward > 0:
                    winner = node.current_player
                elif reward < 0:
                    winner = -node.current_player
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
        # eager expand 후에도 _policy/_legal_actions 저장 (is_leaf() 일관성)
        node._policy = policy
        node._legal_actions = legal

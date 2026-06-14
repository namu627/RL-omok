import numpy as np

try:
    from env.renju_rules import classify_move as _renju_classify
except ImportError:
    from renju_rules import classify_move as _renju_classify

EMPTY = 0
BLACK = 1   # 선공
WHITE = -1  # 후공

# renju_rules.py의 WHITE 표현 (별도 상수, 내부용)
_RENJU_WHITE = 2


class GomokuEnv:
    """
    오목 환경 (OpenAI Gym 스타일).

    상태: (board_size, board_size) numpy array. 1=흑, -1=백, 0=빈칸.
    행동: int, 0 ~ board_size²-1  (row*board_size + col)
    보상: 승 +1 / 패 -1 / 무 0 / 진행 중 0  (현재 플레이어 시점)

    renju=True 옵션:
      - board_size=15, n_in_row=5 필수.
      - 흑의 착수에 렌주 금수(장목·44·33) 적용. 금수 착수 → 즉시 패배.
      - 5목 우선 원칙: 5연속을 완성하면 금수 여부 무관하게 승리.
      - 흑의 legal_actions()에서 금수 위치 제외.
      - 백은 6목 이상도 승리(제약 없음).
    """

    def __init__(self, board_size: int = 6, n_in_row: int = 4, renju: bool = False):
        if renju and board_size != 15:
            raise ValueError("Renju 모드는 board_size=15만 지원합니다.")
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.renju = renju
        self.board: np.ndarray = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player: int = BLACK
        self.done: bool = False
        self.winner: int = EMPTY
        self.move_count: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = BLACK
        self.done = False
        self.winner = EMPTY
        self.move_count = 0
        return self.board.copy()

    def step(self, action: int):
        """
        action: int  (row * board_size + col)
        returns: (next_state, reward, done, info)
          reward: 현재 플레이어(step 호출 시점) 기준
        """
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")

        row, col = divmod(action, self.board_size)

        if self.board[row, col] != EMPTY:
            raise ValueError(f"Illegal action {action}: cell ({row},{col}) is not empty.")

        # ── 렌주 금수 판정 (흑 전용, 착수 전 보드 기준) ──────────────────
        if self.renju and self.current_player == BLACK:
            result = _renju_classify(self._to_renju_board(self.board), row, col)
            if result in ("금수-장목", "금수-44", "금수-33"):
                self.board[row, col] = BLACK
                self.move_count += 1
                self.done = True
                self.winner = WHITE
                info = {
                    "winner": self.winner,
                    "move_count": self.move_count,
                    "reason": result,
                }
                return self.board.copy(), -1.0, True, info

        # ── 정상 착수 ─────────────────────────────────────────────────────
        self.board[row, col] = self.current_player
        self.move_count += 1

        reward, self.done = self._check_terminal(row, col)

        info = {"winner": self.winner, "move_count": self.move_count}
        next_state = self.board.copy()

        # 턴 교체 (게임이 끝나도 current_player는 마지막 착수 플레이어 유지)
        if not self.done:
            self.current_player = -self.current_player

        return next_state, reward, self.done, info

    def legal_actions(self, state: np.ndarray = None) -> list[int]:
        """
        합법 수 목록 반환.

        renju=True이고 흑 차례일 때는 금수 위치를 제외.
        (state가 주어져도 current_player 기준으로 금수 필터 적용)

        [설계 결정] 방식 1 채택: 금수를 합법 수에서 제외 (학습 안정).
          에이전트는 금수를 "존재하지 않는 자리"로 인식하고 탐색 예산을 낭비하지 않음.
          향후 "상대를 금수로 유도하는 전략"까지 학습하려면,
          방식 2(금수 허용 + 착수 즉시 패배)로 전환 고려.
        """
        board = state if state is not None else self.board
        indices = np.argwhere(board == EMPTY)
        actions = [int(r) * self.board_size + int(c) for r, c in indices]

        if self.renju and self.current_player == BLACK:
            renju_b = self._to_renju_board(board)
            actions = [
                a for a in actions
                if _renju_classify(renju_b, a // self.board_size, a % self.board_size)
                not in ("금수-장목", "금수-44", "금수-33")
            ]

        return actions

    def render(self, mode: str = "human") -> None:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        col_header = "  " + " ".join(str(c) for c in range(self.board_size))
        print(col_header)
        for r in range(self.board_size):
            row_str = " ".join(symbols[v] for v in self.board[r])
            print(f"{r} {row_str}")
        player_name = "X(Black)" if self.current_player == BLACK else "O(White)"
        print(f"Turn: {player_name}  Moves: {self.move_count}")
        if self.done:
            if self.winner == EMPTY:
                print("Result: Draw")
            else:
                w = "X(Black)" if self.winner == BLACK else "O(White)"
                print(f"Result: {w} wins!")
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_renju_board(self, board: np.ndarray) -> np.ndarray:
        """gomoku 보드(WHITE=-1)를 renju_rules 형식(WHITE=2)으로 변환."""
        return np.where(board == WHITE, _RENJU_WHITE, board).astype(np.int8)

    def _check_terminal(self, row: int, col: int):
        """착수 직후 호출. (reward, done) 반환."""
        player = self.board[row, col]

        if self._count_streak(row, col, player) >= self.n_in_row:
            self.winner = player
            return 1.0, True

        # 무승부: 빈 칸 없음
        if self.move_count == self.board_size ** 2:
            return 0.0, True

        return 0.0, False

    def _count_streak(self, row: int, col: int, player: int) -> int:
        """4방향 중 최대 연속 길이를 반환."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        best = 0
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            best = max(best, count)
        return best

    # ------------------------------------------------------------------
    # MCTS 전용 경량 착수 시뮬레이션 (self 상태 변경 없음)
    # ------------------------------------------------------------------

    def step_board(self, board: np.ndarray, action: int, player: int) -> tuple:
        """
        주어진 보드에서 착수를 시뮬레이션. self.board 등 인스턴스 상태를 변경하지 않음.
        MCTS _apply_action 전용 — deepcopy(env) 대신 board 배열만 복사해 처리.
        Returns: (new_board, reward, done, info)
        """
        bs = self.board_size
        row, col = divmod(action, bs)
        new_board = board.copy()

        # 렌주 금수 판정 (흑 전용)
        if self.renju and player == BLACK:
            result = _renju_classify(self._to_renju_board(board), row, col)
            if result in ("금수-장목", "금수-44", "금수-33"):
                new_board[row, col] = BLACK
                return new_board, -1.0, True, {"winner": WHITE, "reason": result}

        # 정상 착수
        new_board[row, col] = player

        # 승리 판정
        if self._count_streak_on(new_board, row, col, player) >= self.n_in_row:
            return new_board, 1.0, True, {"winner": player}

        # 무승부 (빈 칸 없음)
        if not (new_board == EMPTY).any():
            return new_board, 0.0, True, {"winner": EMPTY}

        return new_board, 0.0, False, {"winner": EMPTY}

    def _count_streak_on(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """주어진 board 배열에서 연속 길이 계산 — self.board 비참조."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        best = 0
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            best = max(best, count)
        return best

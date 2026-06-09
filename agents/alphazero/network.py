"""
AlphaZero 신경망 (PyTorch).

입력 텐서 형식:  (batch, 3, board_size, board_size)
  ch0 : 현재 플레이어의 돌 위치   (1/0)
  ch1 : 상대 플레이어의 돌 위치   (1/0)
  ch2 : 턴 표시 (현재 플레이어가 BLACK이면 1, WHITE이면 0)

출력:
  log_policy : (batch, board_size²)   log-softmax — 각 칸의 착수 확률
  value      : (batch, 1)             tanh [-1, 1] — 현재 플레이어 시점 국면 평가

보드 크기, 잔차 블록 수, 필터 수를 파라미터화해 6×6 → 9×9 전환 시 재사용 가능.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 상수 (env/gomoku.py 와 동일) ──────────────────────────────────
BLACK = 1
WHITE = -1


# ======================================================================
# 상태 → 텐서 변환 (공용 유틸)
# ======================================================================

def board_to_tensor(board: np.ndarray, current_player: int) -> torch.Tensor:
    """
    (board_size, board_size) numpy array + current_player
    → (3, board_size, board_size) float32 tensor.

    current_player : BLACK(1) 또는 WHITE(-1)
    """
    # 현재 플레이어 시점으로 정규화: 내 돌=+1, 상대=-1
    normed = board * current_player          # (H, W) int8
    ch0 = (normed  > 0).astype(np.float32)  # 내 돌
    ch1 = (normed  < 0).astype(np.float32)  # 상대 돌
    ch2 = np.full_like(ch0, fill_value=1.0 if current_player == BLACK else 0.0)
    stacked = np.stack([ch0, ch1, ch2], axis=0)  # (3, H, W)
    return torch.from_numpy(stacked)


# ======================================================================
# 네트워크 구성 요소
# ======================================================================

class ResBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN + 잔차 → ReLU"""

    def __init__(self, n_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)          # 잔차 연결


# ======================================================================
# 메인 네트워크
# ======================================================================

class AlphaZeroNet(nn.Module):
    """
    AlphaZero 축소판 신경망.

    Parameters
    ----------
    board_size    : 보드 한 변 (6 또는 9 등)
    n_res_blocks  : 잔차 블록 수 (6×6은 3, 9×9는 5 권장)
    n_filters     : 각 Conv 레이어의 필터 수
    """

    def __init__(
        self,
        board_size: int = 6,
        n_res_blocks: int = 3,
        n_filters: int = 64,
    ):
        super().__init__()
        self.board_size = board_size
        n = board_size ** 2

        # ── 입력 레이어 ──────────────────────────────────────────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, n_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )

        # ── 잔차 블록 스택 ───────────────────────────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlock(n_filters) for _ in range(n_res_blocks)]
        )

        # ── 정책 헤드: 1×1 conv → flatten → linear → log_softmax ────
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * n, n),
        )

        # ── 가치 헤드: 1×1 conv → flatten → linear → tanh ───────────
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, 3, H, W)
        반환:
          log_policy : (batch, H*W)   log-softmax
          value      : (batch, 1)     tanh
        """
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        log_policy = F.log_softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return log_policy, value

    @torch.no_grad()
    def predict(
        self,
        board: np.ndarray,
        current_player: int,
        device: str = "cpu",
    ) -> tuple[np.ndarray, float]:
        """
        단일 보드 상태 → (policy_probs, value) 반환. 추론 전용.

        policy_probs : (board_size²,) numpy float32  — 각 칸의 착수 확률
        value        : float  ∈ [-1, 1]             — 현재 플레이어 시점
        """
        self.eval()
        tensor = board_to_tensor(board, current_player).unsqueeze(0).to(device)
        log_policy, val = self(tensor)
        policy = log_policy.exp().squeeze(0).cpu().numpy()
        return policy, float(val.item())

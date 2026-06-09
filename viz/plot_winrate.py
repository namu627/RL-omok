import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def _set_korean_font() -> None:
    """Windows에서 사용 가능한 한글 폰트를 찾아 matplotlib에 설정."""
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    # 한글 폰트를 못 찾으면 영문 레이블로 폴백 (경고 없앰)
    plt.rcParams["axes.unicode_minus"] = False


def _smooth(rates: np.ndarray, episodes: list[int], window: int):
    """이동 평균 적용. (smoothed, smooth_episodes) 반환."""
    if len(rates) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rates, kernel, mode="valid")
        return smoothed, episodes[window - 1:]
    return rates, episodes


def plot_winrate(
    win_rates: list[float],
    eval_interval: int,
    title: str = "Q-Learning Win Rate (vs Random Agent)",
    save_path: str = "winrate_phase1.png",
    smooth_window: int = 5,
) -> None:
    """단일 에이전트 승률 곡선 저장."""
    _set_korean_font()
    episodes = [(i + 1) * eval_interval for i in range(len(win_rates))]
    rates = np.array(win_rates, dtype=float)
    smoothed, smooth_ep = _smooth(rates, episodes, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rates, alpha=0.3, linewidth=1, color="steelblue", label="Win Rate")
    ax.plot(smooth_ep, smoothed, linewidth=2, color="steelblue",
            label=f"Moving Avg (w={smooth_window})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Win Rate vs Random Agent")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if win_rates:
        ax.annotate(
            f"final: {win_rates[-1]:.3f}",
            xy=(episodes[-1], win_rates[-1]),
            xytext=(-60, 15), textcoords="offset points",
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Graph saved: {os.path.abspath(save_path)}")
    plt.close(fig)


def plot_comparison(
    series: list[tuple],
    save_path: str = "comparison_ph1_ph2.png",
    title: str = "Phase 1 vs Phase 2: Win Rate vs Random Agent",
    smooth_window: int = 7,
) -> None:
    """
    여러 에이전트의 승률 곡선을 한 그래프에 비교.

    Parameters
    ----------
    series : list of (win_rates, eval_interval, label, color)
    """
    _set_korean_font()
    fig, ax = plt.subplots(figsize=(12, 6))

    for win_rates, eval_interval, label, color in series:
        if not win_rates:
            continue
        episodes = [(i + 1) * eval_interval for i in range(len(win_rates))]
        rates = np.array(win_rates, dtype=float)
        smoothed, smooth_ep = _smooth(rates, episodes, smooth_window)

        # 원본: 반투명
        ax.plot(episodes, rates, alpha=0.15, linewidth=1, color=color)
        # 이동평균: 진하게
        ax.plot(smooth_ep, smoothed, linewidth=2, color=color, label=label)
        # 최종값 주석
        ax.annotate(
            f"{win_rates[-1]:.3f}",
            xy=(episodes[-1], win_rates[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=color, fontsize=9, fontweight="bold",
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Win Rate vs Random Agent")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison graph saved: {os.path.abspath(save_path)}")
    plt.close(fig)

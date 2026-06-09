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


def plot_winrate(
    win_rates: list[float],
    eval_interval: int,
    title: str = "Q-Learning 승률 (vs Random Agent)",
    save_path: str = "winrate_phase1.png",
    smooth_window: int = 5,
) -> None:
    """
    승률 곡선을 저장.

    win_rates     : evaluate_vs_random() 이 기록한 승률 리스트
    eval_interval : 평가 간격 (에피소드 단위)
    smooth_window : 이동 평균 윈도우 크기
    """
    _set_korean_font()
    episodes = [(i + 1) * eval_interval for i in range(len(win_rates))]
    rates = np.array(win_rates, dtype=float)

    # 이동 평균
    if len(rates) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(rates, kernel, mode="valid")
        smooth_ep = episodes[smooth_window - 1:]
    else:
        smoothed, smooth_ep = rates, episodes

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, rates, alpha=0.3, linewidth=1, color="steelblue", label="원본 승률")
    ax.plot(smooth_ep, smoothed, linewidth=2, color="steelblue",
            label=f"이동 평균 (w={smooth_window})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% 기준선")

    ax.set_xlabel("학습 에피소드")
    ax.set_ylabel("승률 vs 랜덤봇")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 최종 승률 주석
    if win_rates:
        final_ep = episodes[-1]
        final_wr = win_rates[-1]
        ax.annotate(
            f"최종: {final_wr:.3f}",
            xy=(final_ep, final_wr),
            xytext=(-60, 15),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장: {os.path.abspath(save_path)}")
    plt.close(fig)

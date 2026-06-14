"""
Ablation OOD bar chart — team 달팽이질주 톤앤매너 통일판.
T2V 파트 디자인(라벤더 배경 + 일렉트릭 블루 에스컬레이션)에 맞춤.
출력: ablation_ood_v2.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch

# --- Korean font ---
font_path = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

# --- team 달팽이질주 palette ---
MAIN_BLUE = "#1B22E5"   # 진블루 (메인/최종)
MID_BLUE  = "#5A63EC"   # 중간 단계
LITE_BLUE = "#AEB4F2"   # 연블루 (시작)
LAVENDER  = "#F4F5FC"   # 배경
GREY_TXT  = "#3A3A44"
GREY_SUB  = "#8A8FA3"

stages = ["Phase1 baseline\n(주파수 모델 + SNS 증강)",
          "+ DF40 데이터\n(fake 24만/40종 교체)",
          "+ SNS real 파인튜닝\n(real 5x 오버샘플)"]
vals   = [0.41, 0.67, 0.87]
colors = [LITE_BLUE, MID_BLUE, MAIN_BLUE]

fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
fig.patch.set_facecolor(LAVENDER)
ax.set_facecolor(LAVENDER)

x = range(len(vals))
bars = ax.bar(x, vals, width=0.56, color=colors, zorder=3,
              edgecolor="white", linewidth=1.5)

# rounded-ish look: subtle shadow baseline
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=MAIN_BLUE if i == 2 else GREY_TXT, zorder=4)

# random baseline
ax.axhline(0.5, ls="--", lw=1.4, color=GREY_SUB, zorder=2)
ax.text(2.42, 0.505, "랜덤 (0.5)", ha="right", va="bottom",
        fontsize=10.5, color=GREY_SUB)

# improvement arrows + deltas
def delta_arrow(i0, i1, dval):
    y0, y1 = vals[i0], vals[i1]
    arr = FancyArrowPatch((i0 + 0.18, y0 + 0.05), (i1 - 0.18, y1 + 0.05),
                          connectionstyle="arc3,rad=-0.25",
                          arrowstyle="-|>", mutation_scale=18,
                          lw=2.2, color=MAIN_BLUE, zorder=5)
    ax.add_patch(arr)
    mx = (i0 + i1) / 2
    my = max(y0, y1) + 0.135
    ax.text(mx, my, f"+{dval:.2f}", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=MAIN_BLUE, zorder=6)

delta_arrow(0, 1, 0.26)
delta_arrow(1, 2, 0.20)

# axes cosmetics
ax.set_ylim(0, 1.04)
ax.set_xticks(list(x))
ax.set_xticklabels(stages, fontsize=11, color=GREY_TXT)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.0"], fontsize=10, color=GREY_SUB)
ax.set_ylabel("OOD AUC (SNS 화면녹화 140셋)", fontsize=12, color=GREY_TXT)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#D5D8EC")
ax.tick_params(length=0)
ax.grid(axis="y", color="#E3E5F2", lw=1, zorder=0)

ax.set_title("OOD를 끌어올린 주역은 화려한 구조가 아니라 '데이터 전략'",
             fontsize=15.5, fontweight="bold", color=GREY_TXT, pad=16)

plt.tight_layout()
out = "/home/t26106/deepfake/submission/문서/ablation_ood_v2.png"
plt.savefig(out, facecolor=LAVENDER, bbox_inches="tight")
print("saved:", out)

"""Рисунки для статьи: примеры синтетических экспериментов и схема каскада.

Запуск из корня репозитория:
    python code/make_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from calib_synth import generate

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

TITLES = {
    "ok": "норма",
    "low_snr": "сильный шум",
    "no_signal": "нет сигнала",
    "out_of_range": "вне диапазона",
    "undersampled": "редкая сетка",
    "glitch": "скачок / выбросы",
}
FAMILY_TITLES = {
    "resonator": "Спектроскопия резонатора",
    "qubit_spec": "Спектроскопия кубита",
    "rabi": "Раби",
    "ramsey": "Рамзи",
    "t1": "T1",
}


def fig_examples():
    rng = np.random.default_rng(7)
    grid = [
        [("rabi", "ok"), ("rabi", "out_of_range"), ("rabi", "undersampled")],
        [("ramsey", "ok"), ("ramsey", "low_snr"), ("t1", "glitch")],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), dpi=300)
    for r, row in enumerate(grid):
        for c, (fam, out) in enumerate(row):
            x, y, s = generate(fam, out, rng)
            ax = axes[r][c]
            ax.plot(x, y, "o", ms=2.2, color="#1f4e79")
            ax.set_title(f"{FAMILY_TITLES[fam]}: {TITLES[out]}")
            ax.set_xlabel(s.x_label)
            if c == 0:
                ax.set_ylabel(s.y_label)
            ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_examples.png")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                         fc=fc, ec="#333333", lw=0.8)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=7)


def _arrow(ax, p, q, text=None, offset=(0.0, 0.0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333"))
    if text:
        mx, my = (p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=6.5, style="italic")


def fig_cascade():
    fig, ax = plt.subplots(figsize=(3.4, 3.9), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    ax.axis("off")

    w, h = 0.62, 0.11
    x0 = (1 - w) / 2
    _box(ax, (x0, 0.98), w, h, "Сырые данные эксперимента\n(x, y) + тип эксперимента", "#eef3f8")
    _box(ax, (x0, 0.79), w, h, "Ступень 1: аппроксимация\nфизической моделью", "#dfeadf")
    _box(ax, (x0, 0.60), w, h, "Признаки качества фита\n→ классификатор исхода", "#dfeadf")
    _box(ax, (0.02, 0.30), 0.42, 0.14, "Уверенно:\nисход, параметр,\nдействие", "#f4f4f4")
    _box(ax, (0.56, 0.30), 0.42, 0.14, "Ступень 2: VLM\n(график + результат\nфита + контекст)", "#f8eadf")
    _box(ax, (x0, 0.04), w, h, "Решение и объяснение\nдля инженера", "#eef3f8")

    cx = 0.5
    _arrow(ax, (cx, 0.98), (cx, 0.90))
    _arrow(ax, (cx, 0.79), (cx, 0.71))
    _arrow(ax, (0.40, 0.60), (0.23, 0.44), "p ≥ τ", offset=(-0.07, 0.0))
    _arrow(ax, (0.60, 0.60), (0.77, 0.44), "p < τ", offset=(0.07, 0.0))
    _arrow(ax, (0.23, 0.30), (0.40, 0.15))
    _arrow(ax, (0.77, 0.30), (0.60, 0.15))
    fig.savefig(FIG_DIR / "fig_cascade.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_examples()
    fig_cascade()
    print("Рисунки сохранены в", FIG_DIR)

"""Генератор синтетических калибровочных экспериментов сверхпроводящих кубитов.

Для каждого примера генерируются сырые данные (x, y), график и разметка:
семейство эксперимента, класс исхода, истинные параметры и рекомендуемое действие.

Семейства экспериментов:
    resonator  - спектроскопия резонатора считывания (лоренцев провал)
    qubit_spec - спектроскопия кубита (лоренцев пик)
    rabi       - осцилляции Раби по амплитуде импульса
    ramsey     - эксперимент Рамзи (затухающие осцилляции)
    t1         - релаксация (экспоненциальный спад)

Классы исхода и рекомендуемые действия:
    ok           - accept          (принять параметр)
    low_snr      - more_averages   (увеличить число усреднений)
    no_signal    - check_setup     (проверить настройку / расширить поиск)
    out_of_range - shift_range     (сдвинуть или расширить диапазон сканирования)
    undersampled - denser_sweep    (уменьшить шаг сканирования)
    glitch       - repeat          (повторить измерение)

Пример запуска:
    python calib_synth.py --n 3000 --out data --plots 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

FAMILIES = ["resonator", "qubit_spec", "rabi", "ramsey", "t1"]
OUTCOMES = ["ok", "low_snr", "no_signal", "out_of_range", "undersampled", "glitch"]
ACTIONS = {
    "ok": "accept",
    "low_snr": "more_averages",
    "no_signal": "check_setup",
    "out_of_range": "shift_range",
    "undersampled": "denser_sweep",
    "glitch": "repeat",
}
# Доля классов: рутинные исходы преобладают, как в реальной калибровке.
OUTCOME_PROBS = [0.55, 0.12, 0.06, 0.12, 0.08, 0.07]


@dataclass
class Sample:
    family: str
    outcome: str
    action: str
    params: dict
    x_label: str
    y_label: str


# ---------------------------------------------------------------------------
# Физические модели сигналов
# ---------------------------------------------------------------------------

def lorentz(x, x0, gamma, amp, offset):
    return offset + amp * (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)


def rabi(x, period, amp, offset, phase=0.0):
    return offset + amp * 0.5 * (1 - np.cos(2 * np.pi * x / period + phase))


def ramsey(x, t2, detuning, amp, offset, phase=0.0):
    return offset + amp * 0.5 * (1 + np.exp(-x / t2) * np.cos(2 * np.pi * detuning * x + phase))


def t1_decay(x, t1, amp, offset):
    return offset + amp * np.exp(-x / t1)


MODELS = {
    "resonator": lorentz,
    "qubit_spec": lorentz,
    "rabi": rabi,
    "ramsey": ramsey,
    "t1": t1_decay,
}


# ---------------------------------------------------------------------------
# Генерация одного эксперимента
# ---------------------------------------------------------------------------

def _grid(rng, start, stop, n):
    return np.linspace(start, stop, n)


def generate(family: str, outcome: str, rng: np.random.Generator):
    """Возвращает (x, y, Sample)."""
    n_points = int(rng.integers(61, 121))
    noise = rng.uniform(0.01, 0.04)          # шум для «хорошего» эксперимента
    signal_scale = 1.0

    if outcome == "low_snr":
        noise = rng.uniform(0.18, 0.35)
    if outcome == "no_signal":
        signal_scale = 0.0
    if outcome == "undersampled":
        n_points = int(rng.integers(9, 16))

    if family in ("resonator", "qubit_spec"):
        # Частота в ГГц; резонатор ~7 ГГц, кубит (флаксониум/трансмон) 0.5–6 ГГц
        center = rng.uniform(6.8, 7.4) if family == "resonator" else rng.uniform(0.5, 6.0)
        span = rng.uniform(0.01, 0.03) if family == "resonator" else rng.uniform(0.05, 0.2)
        gamma = span * rng.uniform(0.04, 0.12)
        amp = -rng.uniform(0.5, 0.9) if family == "resonator" else rng.uniform(0.5, 0.9)
        offset = 1.0 if family == "resonator" else 0.05
        x0 = center
        if outcome == "out_of_range":
            # центр резонанса за пределами окна сканирования
            x0 = center + rng.choice([-1, 1]) * span * rng.uniform(0.7, 1.5)
        if outcome == "undersampled":
            gamma = span * rng.uniform(0.01, 0.03)  # узкая линия проскакивает между точками
        x = _grid(rng, center - span / 2, center + span / 2, n_points)
        y = lorentz(x, x0, gamma, amp * signal_scale, offset)
        params = {"f0_GHz": x0, "linewidth_GHz": gamma}
        x_label, y_label = "Частота, ГГц", "Амплитуда, отн. ед."

    elif family == "rabi":
        period = rng.uniform(0.3, 0.8)       # амплитуда pi-импульса = period / 2
        amp, offset = rng.uniform(0.7, 0.95), rng.uniform(0.02, 0.1)
        x_max = rng.uniform(2.0, 3.0) * period
        if outcome == "out_of_range":
            x_max = period * rng.uniform(0.15, 0.35)   # не видно даже половины периода
        if outcome == "undersampled":
            x_max = period * rng.uniform(6, 10)        # много периодов на мало точек
        x = _grid(rng, 0.0, x_max, n_points)
        y = rabi(x, period, amp * signal_scale, offset)
        params = {"pi_amp": period / 2}
        x_label, y_label = "Амплитуда импульса, отн. ед.", "Вероятность |1⟩"

    elif family == "ramsey":
        t2 = rng.uniform(5.0, 40.0)                 # мкс
        amp, offset = rng.uniform(0.7, 0.95), rng.uniform(0.02, 0.1)
        x_max = rng.uniform(1.0, 2.5) * t2
        detuning = rng.uniform(3.0, 8.0) / x_max    # МГц: 3–8 периодов в окне
        if outcome == "out_of_range":
            x_max = t2 * rng.uniform(0.02, 0.06)        # окно короче половины периода
            detuning = rng.uniform(0.1, 0.4) / x_max
        if outcome == "undersampled":
            # частота выше частоты Найквиста: меньше двух точек на период
            detuning = n_points * rng.uniform(0.6, 1.4) / x_max
        x = _grid(rng, 0.0, x_max, n_points)
        y = ramsey(x, t2, detuning, amp * signal_scale, offset)
        params = {"t2_star_us": t2, "detuning_MHz": detuning}
        x_label, y_label = "Задержка, мкс", "Вероятность |1⟩"

    elif family == "t1":
        t1 = rng.uniform(10.0, 300.0)               # мкс (флаксониумы — до ~1 мс)
        amp, offset = rng.uniform(0.7, 0.95), rng.uniform(0.02, 0.08)
        x_max = rng.uniform(3.0, 5.0) * t1
        if outcome == "out_of_range":
            x_max = t1 * rng.choice([rng.uniform(0.05, 0.15), rng.uniform(30, 60)])
        if outcome == "undersampled":
            x_max = t1 * rng.uniform(25, 40)            # спад укладывается в 1–2 точки
        x = _grid(rng, 0.0, x_max, n_points)
        y = t1_decay(x, t1, amp * signal_scale, offset)
        params = {"t1_us": t1}
        x_label, y_label = "Задержка, мкс", "Вероятность |1⟩"
    else:
        raise ValueError(family)

    y = y + rng.normal(0.0, noise, size=x.shape)

    if outcome == "glitch":
        # скачок уровня или выбросы (например, флуктуации TLS-дефектов)
        if rng.random() < 0.5:
            k = int(rng.integers(len(x) // 3, 2 * len(x) // 3))
            y[k:] += rng.uniform(0.15, 0.3)
        else:
            idx = rng.choice(len(x), size=max(3, len(x) // 15), replace=False)
            y[idx] += rng.normal(0.0, 0.6, size=idx.shape)

    params = {k: float(v) for k, v in params.items()}
    params["noise_sigma"] = float(noise)
    sample = Sample(family, outcome, ACTIONS[outcome], params, x_label, y_label)
    return x, y, sample


def generate_dataset(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        outcome = rng.choice(OUTCOMES, p=OUTCOME_PROBS)
        x, y, s = generate(family, str(outcome), rng)
        items.append((x, y, s))
    return items


def save_dataset(items, out_dir: Path, n_plots: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    arrays = {}
    for i, (x, y, s) in enumerate(items):
        arrays[f"x_{i}"] = x
        arrays[f"y_{i}"] = y
        meta.append({"id": i, **asdict(s)})
    np.savez_compressed(out_dir / "raw.npz", **arrays)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    if n_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        for i, (x, y, s) in enumerate(items[:n_plots]):
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            ax.plot(x, y, "o", ms=3)
            ax.set_xlabel(s.x_label)
            ax.set_ylabel(s.y_label)
            ax.set_title(s.family)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{i:05d}.png")
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3000, help="число экспериментов")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--plots", type=int, default=0, help="сколько графиков сохранить в PNG")
    args = p.parse_args()
    items = generate_dataset(args.n, args.seed)
    save_dataset(items, args.out, args.plots)
    counts = {}
    for _, _, s in items:
        counts[s.outcome] = counts.get(s.outcome, 0) + 1
    print(f"Сохранено {len(items)} экспериментов в {args.out}: {counts}")


if __name__ == "__main__":
    main()

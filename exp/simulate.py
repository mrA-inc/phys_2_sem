"""
simulate.py
===========
Параметрическая симуляция формы капли с CLI и валидацией физических параметров.

Использование:
    python simulate.py --volume 30 --alpha 35 --theta 100 --fluid water
    python simulate.py --volume 40 --alpha 0  --theta 112 --3d --save /tmp/drop.pdf
    python simulate.py --cv                   # ввод CV-данных и сравнение с теорией

API:
    from simulate import run_simulation
    result = run_simulation(30, 35, 100, fluid=WATER)
"""
import sys
import os
import argparse

# Переключаем stdin в UTF-8 на Windows (убирает BOM от PowerShell echo)
if sys.platform == "win32":
    import io
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

from physics import WATER, GLYCERIN, ETHYLENE_GLYCOL, MERCURY, Fluid
from drop_axisym import fit_b_for_volume, drop_volume
from inclined import (
    hysteresis_symmetric, two_arc_profile_2d,
    max_volume_3d, sliding_angle_2d,
)


# ===========================================================================
# Исключения
# ===========================================================================
class DropPhysicsError(ValueError):
    """Базовый класс ошибок физической невозможности."""


class DropSlidesError(DropPhysicsError):
    """Капля соскальзывает при данных параметрах."""


class VolumeError(DropPhysicsError):
    """Объём вне допустимого диапазона."""


class AngleError(DropPhysicsError):
    """Угол (наклона или краевой) вне допустимого диапазона."""


# ===========================================================================
# Валидация
# ===========================================================================
def validate_params(volume_ul: float, alpha_deg: float,
                    theta_deg: float, fluid: Fluid,
                    width_mm: float | None = None) -> dict:
    """
    Проверяет физическую допустимость параметров капли.
    Возвращает словарь с предупреждениями и вспомогательными числами.
    Бросает исключение при невозможных параметрах.
    """
    warnings_list = []

    if not (0.01 <= volume_ul <= 10000):
        raise VolumeError(
            f"Объём {volume_ul:.3f} мкл вне диапазона [0.01, 10000] мкл"
        )

    if not (0.0 <= alpha_deg < 90.0):
        raise AngleError(
            f"Угол наклона {alpha_deg:.1f}° вне диапазона [0, 90)°"
        )

    if not (1.0 < theta_deg < 179.0):
        raise AngleError(
            f"Краевой угол {theta_deg:.1f}° вне диапазона (1, 179)°"
        )

    V = volume_ul * 1e-9   # мкл → м³
    theta = np.deg2rad(theta_deg)
    alpha = np.deg2rad(alpha_deg)

    V_max = None
    if alpha_deg > 0.01:
        if width_mm is None:
            width_mm = 2.0 * (3 * V / (2 * np.pi)) ** (1 / 3) * 1e3
        w = width_mm * 1e-3

        delta_lim = np.deg2rad(30.0)
        thA = theta + delta_lim
        thR = theta - delta_lim

        V_max = max_volume_3d(alpha, thA, thR, fluid, w)

        if V > V_max:
            alpha_c = None
            for a_deg in np.linspace(0.5, 89, 500):
                if max_volume_3d(np.deg2rad(a_deg), thA, thR, fluid, w) <= V:
                    alpha_c = a_deg
                    break

            msg = (
                f"Капля объёмом {volume_ul:.1f} мкл соскальзывает при "
                f"α={alpha_deg:.1f}°.\n"
                f"  Макс. удерживаемый объём при α={alpha_deg:.1f}°: "
                f"{V_max * 1e9:.1f} мкл.\n"
            )
            if alpha_c is not None:
                msg += f"  Критический угол для данного объёма: α_c ≈ {alpha_c:.1f}°."
            raise DropSlidesError(msg)

    R_est = (3 * V / (2 * np.pi)) ** (1 / 3)
    Bo = fluid.bond_number(R_est)
    if Bo > 5:
        warnings_list.append(
            f"Предупреждение: Bo={Bo:.2f} > 5; "
            f"гравитация сильно искажает форму, модель двух дуг менее точна."
        )

    return {
        "warnings": warnings_list,
        "bond_number": Bo,
        "V_max_ul": V_max * 1e9 if V_max is not None else None,
    }


# ===========================================================================
# Основная симуляция
# ===========================================================================
def run_simulation(volume_ul: float, alpha_deg: float,
                   theta_deg: float, fluid: Fluid = WATER,
                   width_mm: float | None = None,
                   theta_A_deg: float | None = None,
                   theta_R_deg: float | None = None) -> dict:
    """
    Запускает симуляцию для заданных параметров.
    Возвращает словарь с профилями и геометрией.
    Бросает DropPhysicsError при невозможных параметрах.

    Если заданы theta_A_deg и theta_R_deg (например, измеренные CV),
    профиль строится из них НАПРЯМУЮ через две дуги, а не пересчитывается
    через симметричную модель гистерезиса. Это нужно для воспроизведения
    реальной асимметричной капли с известными краевыми углами.
    """
    meta = validate_params(volume_ul, alpha_deg, theta_deg, fluid, width_mm)

    V = volume_ul * 1e-9
    theta = np.deg2rad(theta_deg)
    alpha = np.deg2rad(alpha_deg)

    measured_angles = theta_A_deg is not None and theta_R_deg is not None

    if alpha_deg < 0.5 and not measured_angles:
        # горизонталь: осесимметрия
        prof = fit_b_for_volume(V, fluid.c, theta)
        thA = thR = theta_deg
        profile_x = prof.x * 1e3    # м → мм
        profile_z = (prof.apex_height - prof.z) * 1e3
    else:
        # наклон (или заданы измеренные углы): две дуги
        if width_mm is None:
            width_mm = 2.0 * (3 * V / (2 * np.pi)) ** (1 / 3) * 1e3
        w = width_mm * 1e-3
        if measured_angles:
            # строим профиль из ИЗМЕРЕННЫХ краевых углов (CV), без гистерезиса
            thA, thR = theta_A_deg, theta_R_deg
            thA_r, thR_r = np.deg2rad(thA), np.deg2rad(thR)
            # ширину основания подгоняем под измеренную (через zeta = base/2)
            zeta = w / 2
        else:
            thA_r, thR_r, _ = hysteresis_symmetric(alpha, V, theta, fluid, w)
            thA, thR = np.rad2deg(thA_r), np.rad2deg(thR_r)
            zeta = w / 2 * 1.1
        xs, zs, info = two_arc_profile_2d(zeta, thA_r, thR_r)
        profile_x = xs * 1e3
        profile_z = zs * 1e3
        prof = None

    base_width = profile_x.max() - profile_x.min()
    apex_height = profile_z.max()
    R_est = base_width / 2 * 1e-3
    Bo = fluid.bond_number(R_est)

    geometry = {
        "volume_ul":            volume_ul,
        "alpha_deg":            alpha_deg,
        "theta_Y_deg":          theta_deg,
        "theta_A_deg":          thA,
        "theta_R_deg":          thR,
        "base_width_mm":        base_width,
        "apex_height_mm":       apex_height,
        "capillary_length_mm":  fluid.capillary_length * 1e3,
        "bond_number":          Bo,
        "fluid":                fluid.name,
    }
    geometry.update({k: v for k, v in meta.items() if k != "warnings"})

    return {
        "profile_x":      profile_x,
        "profile_z":      profile_z,
        "axisym_profile": prof,
        "geometry":       geometry,
        "warnings":       meta["warnings"],
        "fluid":          fluid,
        "alpha_deg":      alpha_deg,
    }


# ===========================================================================
# Вид сбоку
# ===========================================================================
def plot_sideview(result: dict, ax=None, show: bool = True, save_path=None):
    """
    Вид сбоку в системе ПОДЛОЖКИ: основание капли лежит горизонтально на
    линии поверхности (как на реальном снимке), а наклон передаётся стрелкой
    силы тяжести под углом alpha к нормали. Жёсткий поворот всего профиля
    НЕ применяется — иначе плоская капля выглядит косой «лодочкой» в воздухе.
    """
    g = result["geometry"]
    x = result["profile_x"]
    z = result["profile_z"]
    alpha = result["alpha_deg"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    # Горизонтальная капля: profile_x — полуширина (0..R), зеркалим в полный контур.
    # Наклонная капля: profile_x уже полный асимметричный контур (-L2..+L1,
    # два разных краевых угла), зеркалить НЕЛЬЗЯ — иначе теряется асимметрия.
    if x.min() < -1e-9:
        x_full, z_full = x, z                       # уже полный контур (наклон)
    else:
        x_full = np.concatenate([-x[::-1], x])      # зеркало (горизонталь)
        z_full = np.concatenate([z[::-1], z])

    # капля в системе подложки: основание на z=0, вершина вверху
    ax.fill(x_full, z_full, color="#4a90d9", alpha=0.35)
    ax.plot(x_full, z_full, color="#1a5fa8", lw=2)

    # линия подложки (поверхность зеркала) — горизонтальна в этой системе
    ax.axhline(0, color="#888", lw=1.6)

    xlo, xhi = x_full.min(), x_full.max()
    zmax = z_full.max()
    xpad = (xhi - xlo) * 0.06
    zpad = zmax * 0.22

    # вектор силы тяжести: под углом alpha к вертикали (downhill — в сторону +x)
    if alpha > 0.5:
        gx = np.sin(np.deg2rad(alpha))
        gz = -np.cos(np.deg2rad(alpha))
        L = 0.65 * zmax if zmax > 0 else 0.3
        x0 = xlo + (xhi - xlo) * 0.08
        z0 = zmax * 1.05
        ax.annotate("", xy=(x0 + gx*L, z0 + gz*L), xytext=(x0, z0),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.text(x0 - 0.2, z0, "g", color="#c0392b",
                fontsize=12, ha="right", va="center")
        ax.text(xhi - 0.05, -zpad * 0.5, "downhill", fontsize=8,
                color="#555", ha="right", va="top")
        ax.text(xlo + 0.05, -zpad * 0.5, "uphill", fontsize=8,
                color="#555", ha="left", va="top")

    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(-zpad, zmax + zpad * 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x, мм (вдоль подложки)")
    ax.set_ylabel("z, мм (от подложки)")
    title = (
        f"Капля {g['fluid']} | V={g['volume_ul']:.1f} мкл | "
        f"α={g['alpha_deg']:.0f}° | θ_A/θ_R={g['theta_A_deg']:.0f}/{g['theta_R_deg']:.0f}°\n"
        f"Bo={g['bond_number']:.2f} | "
        f"основание={g['base_width_mm']:.2f} мм | "
        f"h={g['apex_height_mm']:.2f} мм"
    )
    ax.set_title(title, fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# ===========================================================================
# CLI
# ===========================================================================
FLUIDS = {
    "water":          WATER,
    "glycerin":       GLYCERIN,
    "ethylene_glycol": ETHYLENE_GLYCOL,
    "mercury":        MERCURY,
}

def _prompt_float(prompt: str, lo: float, hi: float, default: float | None = None) -> float:
    """Запрашивает число с клавиатуры, пока оно не попадает в [lo, hi]."""
    hint = f"  (рекомендуется {lo}-{hi}"
    if default is not None:
        hint += f", Enter = {default}"
    hint += "): "
    while True:
        raw = input(prompt + hint).strip().lstrip("﻿")
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
        except ValueError:
            print(f"    WARN: Введите число.")
            continue
        if not (lo <= val <= hi):
            print(f"    WARN: Значение вне диапазона [{lo}, {hi}].")
            continue
        return val


# ===========================================================================
# Авто-режим: читаем cv_result.json и воспроизводим реальную каплю
# ===========================================================================
def _load_cv_json(path: str | None = None) -> dict:
    """Читает числа из cv_result.json (вывод dropcv.py)."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "out", "cv_result.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл {path} не найден. Сначала запустите dropcv.py "
            f"(он создаёт cv_result.json)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_comparison_pdf(cv: dict, result_cv: dict, result_th: dict, save_path: str):
    """
    PDF с двумя панелями:
      Левая  — таблица: CV-измерение vs теория (hysteresis_symmetric).
      Правая — наложенные профили на общих осях:
                 красный сплошной  = CV (из измеренных theta_A/theta_R)
                 синий пунктир     = теория Юнга-Лапласа (hysteresis_symmetric)
    Два профиля реально разные: CV даёт то, что видно на снимке,
    теория предсказывает то, что должно быть при symmetric hysteresis.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig,
                           left=0.03, right=0.97, wspace=0.06)
    ax_tbl  = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1])

    # ── извлекаем числа ──────────────────────────────────────────────────────
    g_cv = result_cv["geometry"]
    g_th = result_th["geometry"]

    theta_L  = cv["theta_L_deg"]
    theta_R  = cv["theta_R_deg"]
    thA_cv   = max(theta_L, theta_R)
    thR_cv   = min(theta_L, theta_R)
    base_cv  = cv["base_width_mm"]
    h_cv     = cv["apex_height_mm"]
    alpha_p  = cv["alpha_phys_deg"]
    R_m = base_cv * 1e-3 / 2
    h_m = h_cv * 1e-3
    V_cv = np.pi * h_m * (3*R_m**2 + h_m**2) / 6 * 1e9

    def _pct(cv_v, th_v):
        if cv_v is None or th_v is None or abs(th_v) < 1e-12:
            return "—"
        return f"{100*(cv_v - th_v)/abs(th_v):+.1f}%"

    rows = [
        ["Параметр",         "CV (снимок)",         "Теория (YL)",         "delta"],
        ["theta_A, grad",    f"{thA_cv:.1f}",        f"{g_th['theta_A_deg']:.1f}",    _pct(thA_cv, g_th['theta_A_deg'])],
        ["theta_R, grad",    f"{thR_cv:.1f}",        f"{g_th['theta_R_deg']:.1f}",    _pct(thR_cv, g_th['theta_R_deg'])],
        ["theta_Y, grad",    f"{0.5*(thA_cv+thR_cv):.1f}", f"{g_th['theta_Y_deg']:.1f}", _pct(0.5*(thA_cv+thR_cv), g_th['theta_Y_deg'])],
        ["alpha_phys, grad", f"{alpha_p:.1f}",       f"{g_th['alpha_deg']:.1f}",      "—"],
        ["base_width, mm",   f"{base_cv:.2f}",       f"{g_th['base_width_mm']:.2f}",  _pct(base_cv, g_th['base_width_mm'])],
        ["apex_height, mm",  f"{h_cv:.2f}",          f"{g_th['apex_height_mm']:.2f}", _pct(h_cv, g_th['apex_height_mm'])],
        ["V_est, mkl",       f"{V_cv:.2f}",          f"{g_th['volume_ul']:.2f}",      _pct(V_cv, g_th['volume_ul'])],
        ["Bo",               "—",                    f"{g_th['bond_number']:.3f}",    "—"],
        ["a (kap.), mm",     "—",                    f"{g_th['capillary_length_mm']:.2f}", "—"],
    ]

    # ── таблица ──────────────────────────────────────────────────────────────
    ax_tbl.axis("off")
    col_w = [0.38, 0.22, 0.22, 0.16]
    col_x = [0.01, 0.39, 0.63, 0.85]
    row_h = 1.0 / (len(rows) + 0.5)

    for r_i, row in enumerate(rows):
        y = 1.0 - (r_i + 0.5) * row_h
        for c_i, cell in enumerate(row):
            weight = "bold" if r_i == 0 else "normal"
            bg = "#e8f0fe" if r_i == 0 else ("#f7f7f7" if r_i % 2 == 0 else "white")
            # подсветить строки с большим расхождением
            if r_i > 0 and c_i == 3 and "%" in cell:
                try:
                    pct_val = float(cell.replace("%","").replace("+",""))
                    if abs(pct_val) > 20:
                        bg = "#fff3cd"
                except ValueError:
                    pass
            ax_tbl.text(col_x[c_i] + col_w[c_i]*0.5, y, cell,
                        ha="center", va="center", fontsize=8,
                        fontweight=weight, transform=ax_tbl.transAxes,
                        bbox=dict(boxstyle="square,pad=0.18", fc=bg,
                                  ec="#bbbbbb", lw=0.5))

    ax_tbl.set_title(
        "CV (снимок) vs теория Юнга-Лапласа\n"
        "(1 снимок, вода на стекле, sigma=72.8 мН/м)\n"
        "Теория: hysteresis_symmetric(V, alpha, theta_Y)",
        fontsize=8.5, pad=6
    )

    # ── профили ──────────────────────────────────────────────────────────────
    def _full(x, z):
        if x.min() < -1e-9:
            return x, z
        return np.concatenate([-x[::-1], x]), np.concatenate([z[::-1], z])

    # CV-профиль: из ИЗМЕРЕННЫХ углов (то, что видно на снимке)
    x_cv_f, z_cv_f = _full(result_cv["profile_x"], result_cv["profile_z"])
    # Теоретический профиль: из hysteresis_symmetric
    x_th_f, z_th_f = _full(result_th["profile_x"], result_th["profile_z"])

    ax_prof.fill(x_cv_f, z_cv_f, color="#e74c3c", alpha=0.20, zorder=1)
    ax_prof.plot(x_cv_f, z_cv_f, color="#c0392b", lw=2.2,
                 label=f"CV: thA={thA_cv:.1f}, thR={thR_cv:.1f} grad, base={base_cv:.2f} mm",
                 zorder=3)

    ax_prof.fill(x_th_f, z_th_f, color="#3498db", alpha=0.20, zorder=2)
    ax_prof.plot(x_th_f, z_th_f, color="#1a5fa8", lw=2.2, ls="--",
                 label=f"Теория: thA={g_th['theta_A_deg']:.1f}, "
                       f"thR={g_th['theta_R_deg']:.1f} grad, "
                       f"base={g_th['base_width_mm']:.2f} mm",
                 zorder=4)

    ax_prof.axhline(0, color="#666", lw=1.4)

    # стрелка g
    alpha_r = np.deg2rad(alpha_p)
    # охват по x/z: берём максимум обоих профилей + поле
    x_all = np.concatenate([x_cv_f, x_th_f])
    z_all = np.concatenate([z_cv_f, z_th_f])
    xlo, xhi = x_all.min(), x_all.max()
    zmax = z_all.max()
    xpad = (xhi - xlo) * 0.08
    zpad = zmax * 0.22

    L_arr = 0.45 * zmax if zmax > 0.05 else 0.3
    x0 = xlo + (xhi - xlo) * 0.08
    z0 = zmax * 1.05
    ax_prof.annotate("", xy=(x0 + np.sin(alpha_r)*L_arr, z0 - np.cos(alpha_r)*L_arr),
                     xytext=(x0, z0),
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8))
    ax_prof.text(x0 - 0.12, z0, "g", color="#c0392b", fontsize=11,
                 ha="right", va="center")

    # подписи uphill/downhill внутри поля (не за пределами)
    ax_prof.text(xhi - 0.05, -zpad * 0.55, "downhill", fontsize=7.5,
                 color="#555", ha="right", va="top")
    ax_prof.text(xlo + 0.05, -zpad * 0.55, "uphill", fontsize=7.5,
                 color="#555", ha="left", va="top")

    ax_prof.set_xlim(xlo - xpad, xhi + xpad)
    ax_prof.set_ylim(-zpad, zmax + zpad * 1.5)
    ax_prof.set_aspect("equal")
    ax_prof.set_xlabel("x, мм")
    ax_prof.set_ylabel("z, мм")
    ax_prof.legend(fontsize=7.5, loc="upper center")
    ax_prof.set_title(
        f"Наложение профилей  (alpha={alpha_p:.1f} grad)\n"
        f"CV h={h_cv:.2f} mm  |  Теория h={g_th['apex_height_mm']:.2f} mm",
        fontsize=9
    )

    fig.suptitle(
        f"Капля воды на наклонном зеркале: эксперимент vs теория Юнга-Лапласа  "
        f"(sigma={WATER.sigma*1e3:.1f} мН/м)",
        fontsize=10, y=1.01
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cv_auto_mode(json_path: str | None = None):
    """
    Читает cv_result.json, строит два профиля и сохраняет comparison.pdf.
      result_cv  — из ИЗМЕРЕННЫХ краевых углов (что видно на снимке)
      result_th  — из hysteresis_symmetric(V, alpha, theta_Y) (что предсказывает теория)
    Никакого интерактивного ввода.
    """
    print(f"\n{'='*60}")
    print("  Авто-режим: cv_result.json -> comparison.pdf")
    print(f"  Жидкость: вода (sigma={WATER.sigma*1e3:.1f} мН/м, известная)")
    print(f"{'='*60}\n")

    try:
        cv = _load_cv_json(json_path)
    except FileNotFoundError as e:
        print(f"  ОШИБКА: {e}")
        return

    theta_L   = cv["theta_L_deg"]
    theta_R   = cv["theta_R_deg"]
    alpha_phys = cv["alpha_phys_deg"]
    base_mm   = cv["base_width_mm"]
    h_mm      = cv["apex_height_mm"]
    theta_Y   = 0.5 * (theta_L + theta_R)
    thA_cv    = max(theta_L, theta_R)
    thR_cv    = min(theta_L, theta_R)

    R_m = base_mm * 1e-3 / 2
    h_m = h_mm  * 1e-3
    V_est_ul = np.pi * h_m * (3*R_m**2 + h_m**2) / 6 * 1e9

    print(f"  CV: theta_A={thA_cv:.1f}, theta_R={thR_cv:.1f} grad, "
          f"alpha={alpha_phys:.1f} grad")
    print(f"       base={base_mm:.2f} мм, h={h_mm:.2f} мм, V~{V_est_ul:.2f} мкл\n")

    # Профиль 1 — из ИЗМЕРЕННЫХ углов (воспроизводит снимок)
    try:
        result_cv = run_simulation(
            V_est_ul, alpha_phys, theta_Y, fluid=WATER,
            width_mm=base_mm, theta_A_deg=thA_cv, theta_R_deg=thR_cv,
        )
    except DropPhysicsError as e:
        print(f"  ОШИБКА (CV-профиль): {e}")
        return

    # Профиль 2 — из теоретических углов (hysteresis_symmetric)
    try:
        result_th = run_simulation(
            V_est_ul, alpha_phys, theta_Y, fluid=WATER,
            width_mm=base_mm,
            # theta_A_deg/theta_R_deg НЕ задаём — используется hysteresis_symmetric
        )
    except DropPhysicsError as e:
        print(f"  ОШИБКА (теор. профиль): {e}")
        return

    g_cv = result_cv["geometry"]
    g_th = result_th["geometry"]
    print(f"  CV   профиль: base={g_cv['base_width_mm']:.2f} мм, "
          f"h={g_cv['apex_height_mm']:.2f} мм, "
          f"thA={g_cv['theta_A_deg']:.1f}, thR={g_cv['theta_R_deg']:.1f} grad")
    print(f"  Теор профиль: base={g_th['base_width_mm']:.2f} мм, "
          f"h={g_th['apex_height_mm']:.2f} мм, "
          f"thA={g_th['theta_A_deg']:.1f}, thR={g_th['theta_R_deg']:.1f} grad")
    print(f"  Bo={g_th['bond_number']:.3f}")

    figs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs"
    )
    os.makedirs(figs_dir, exist_ok=True)

    cmp_path = os.path.join(figs_dir, "comparison.pdf")
    _make_comparison_pdf(cv, result_cv, result_th, cmp_path)
    print(f"  Сохранено: {cmp_path}")

    sv_path = os.path.join(figs_dir, "cv_theory_sideview.pdf")
    plot_sideview(result_cv, show=False, save_path=sv_path)
    print(f"  Профиль CV: {sv_path}")

    dr_path = os.path.join(figs_dir, "drop_result.pdf")
    plot_sideview(result_cv, show=False, save_path=dr_path)
    print(f"  drop_result: {dr_path}")

    return result_cv, result_th


# ===========================================================================
# Режим CV-сравнения (ручной ввод)
# ===========================================================================
def _cv_compare_mode():
    """
    Интерактивный ввод CV-измерений и сравнение с теорией Юнга–Лапласа.
    Использует сигму воды как известную константу (WATER.sigma = 72.8 мН/м).

    CV-данные:
      theta_L, theta_R  — левый/правый краевые углы из dropcv
      alpha_screen      — угол синей базовой линии на экране (0..90 deg)
      base_mm           — ширина основания, мм
      h_mm              — высота капли, мм

    Физический угол наклона:
      alpha_phys = 90 - alpha_screen  (камера вертикальна, зеркало горизонтально)

    Объём оценивается из геометрии: V ~ pi*h*(3*R^2+h^2)/6, где R = base/2.
    """
    print(f"\n{'='*60}")
    print("  Режим CV-сравнения: CV-измерение vs теория Юнга-Лапласа")
    print(f"  Жидкость: вода (sigma={WATER.sigma*1e3:.1f} мН/м, известная)")
    print(f"{'='*60}\n")

    print("  Введите данные из dropcv.py (exp_data.jpg).")
    print("  Если значение неизвестно - нажмите Enter для пропуска.\n")

    def _prompt_opt(prompt, lo, hi, default=None):
        """Запрашивает число или пустую строку (возвращает None)."""
        hint = f"  ({lo}-{hi}"
        if default is not None:
            hint += f", Enter = {default}"
        hint += "): "
        while True:
            raw = input(prompt + hint).strip().lstrip("﻿")
            if raw == "" and default is not None:
                return default
            if raw == "":
                return None
            try:
                val = float(raw)
            except ValueError:
                print("    WARN: Введите число.")
                continue
            if not (lo <= val <= hi):
                print(f"    WARN: Вне диапазона [{lo}, {hi}].")
                continue
            return val

    theta_L = _prompt_opt("  theta_L (лев. краевой угол), grad", 1.0, 179.0, default=None)
    theta_R = _prompt_opt("  theta_R (прав. краевой угол), grad", 1.0, 179.0, default=None)
    alpha_screen = _prompt_opt("  alpha_screen (угол синей линии на экране), grad",
                               0.0, 89.9, default=None)
    base_mm = _prompt_opt("  Ширина основания (base_mm), мм", 0.1, 50.0, default=None)
    h_mm    = _prompt_opt("  Высота капли (h_mm), мм", 0.01, 20.0, default=None)

    # вычисляем производные
    if theta_L is not None and theta_R is not None:
        theta_Y = 0.5 * (theta_L + theta_R)
    elif theta_L is not None:
        theta_Y = theta_L
    elif theta_R is not None:
        theta_Y = theta_R
    else:
        print("\n  ОШИБКА: краевые углы не заданы, сравнение невозможно.")
        return

    alpha_phys = (90.0 - alpha_screen) if alpha_screen is not None else 0.0

    # оценка объёма
    V_est_ul = None
    if base_mm is not None and h_mm is not None:
        R_m = (base_mm * 1e-3) / 2
        h_m = h_mm * 1e-3
        V_est = np.pi * h_m * (3 * R_m**2 + h_m**2) / 6
        V_est_ul = V_est * 1e9

    print(f"\n{'-'*60}")
    print(f"  Входные CV-данные:")
    print(f"    theta_L       = {theta_L:.1f} grad" if theta_L else "    theta_L       = (не задан)")
    print(f"    theta_R       = {theta_R:.1f} grad" if theta_R else "    theta_R       = (не задан)")
    print(f"    theta_Y       = {theta_Y:.1f} grad  (среднее краевого угла)")
    print(f"    alpha_screen  = {alpha_screen:.1f} grad" if alpha_screen is not None else
          "    alpha_screen  = (не задан)")
    print(f"    alpha_phys    = {alpha_phys:.1f} grad  (физический наклон зеркала)")
    print(f"    base_mm (CV)  = {base_mm:.2f} мм" if base_mm else "    base_mm (CV)  = (не задан)")
    print(f"    h_mm   (CV)   = {h_mm:.2f} мм" if h_mm else "    h_mm   (CV)   = (не задан)")
    print(f"    V_est         = {V_est_ul:.1f} мкл" if V_est_ul else "    V_est         = (не вычислен)")

    if V_est_ul is None:
        print("\n  Объём не вычислен (нужны base_mm и h_mm).")
        print("  Введите объём вручную:")
        V_est_ul = _prompt_float("  V, мкл", 0.01, 10000.0, default=10.0)

    # теоретическая симуляция
    print(f"\n  Запускаем теорию (alpha={alpha_phys:.1f} grad, theta_Y={theta_Y:.1f} grad, "
          f"V={V_est_ul:.1f} мкл)...")
    try:
        result = run_simulation(V_est_ul, alpha_phys, theta_Y, fluid=WATER)
    except DropPhysicsError as e:
        print(f"\n  ОШИБКА теории: {e}")
        print("  Попробуйте уменьшить объём или угол наклона.")
        return

    g = result["geometry"]
    for w in result["warnings"]:
        print(f"  WARN: {w}")

    # таблица сравнения
    print(f"\n{'='*60}")
    print(f"  {'Параметр':<28} {'CV (измерение)':>14}  {'Теория':>10}")
    print(f"  {'-'*56}")

    def _row(name, cv_val, th_val, fmt=".2f", unit=""):
        cv_s = f"{cv_val:{fmt}}{unit}" if cv_val is not None else "n/a"
        th_s = f"{th_val:{fmt}}{unit}" if th_val is not None else "n/a"
        diff = ""
        if cv_val is not None and th_val is not None and th_val != 0:
            pct = 100 * (cv_val - th_val) / abs(th_val)
            diff = f"  ({pct:+.1f}%)"
        print(f"  {name:<28} {cv_s:>14}  {th_s:>10}{diff}")

    _row("theta_Y, grad", theta_Y, g["theta_Y_deg"], ".1f", " gr")
    _row("theta_A, grad", theta_R, g["theta_A_deg"] if alpha_phys > 0.5 else None, ".1f", " gr")
    _row("theta_R, grad", theta_L, g["theta_R_deg"] if alpha_phys > 0.5 else None, ".1f", " gr")
    _row("base_width, mm", base_mm, g["base_width_mm"], ".2f", " mm")
    _row("h (vysota), mm", h_mm, g["apex_height_mm"], ".2f", " mm")
    _row("alpha_phys, grad", alpha_phys, alpha_phys, ".1f", " gr")
    print(f"  {'-'*56}")
    print(f"  {'Bo (теория)':<28} {'':>14}  {g['bond_number']:>10.3f}")
    print(f"  {'V (теория)':<28} {'':>14}  {g['volume_ul']:>10.1f} мкл")
    print(f"  {'a (капилл. длина)':<28} {'':>14}  {g['capillary_length_mm']:>10.2f} мм")
    print(f"{'='*60}")

    print("\n  Примечание: 1 снимок - сравнение предварительное.")
    if alpha_phys < 0.5 and alpha_screen is not None and alpha_screen > 1:
        print(f"  Замечание: alpha_screen={alpha_screen:.1f} grad -> alpha_phys={alpha_phys:.1f} grad")
        print("  Горизонтальная капля (theta_A=theta_R не вычисляется).")

    # сохранить вид сбоку
    out_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(os.path.dirname(out_dir), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    save_path = os.path.join(figs_dir, "cv_theory_sideview.pdf")
    plot_sideview(result, show=False, save_path=save_path)
    print(f"\n  Профиль (теория) сохранён: {save_path}")


def _interactive_mode(fluid: "Fluid", do_3d: bool, save: str | None, width_mm: float | None):
    """Интерактивный ввод параметров с рекомендациями и повтором при ошибке."""
    a_mm = fluid.capillary_length * 1e3
    print(f"\n{'='*54}")
    print(f"  Жидкость: {fluid.name}  |  капиллярная длина a = {a_mm:.2f} мм")
    print(f"{'='*54}")

    # --- рекомендации ---
    # Типичный лабораторный диапазон: Bo < 4 → V < 4a³ * (2π/3)
    V_rec_max = 4 * fluid.capillary_length**3 * (2 * np.pi / 3) * 1e9   # мкл
    print(f"\n  Объём V:")
    print(f"    0.1-{V_rec_max:.0f} мкл  - умеренное влияние гравитации (Bo < 4)")
    print(f"    > {V_rec_max:.0f} мкл    - сильное уплощение, модель менее точна")

    print(f"\n  Краевой угол theta:")
    print(f"    10-80 grad   - гидрофильная поверхность")
    print(f"    80-110 grad  - нейтральная / слабо гидрофобная")
    print(f"    110-170 grad - гидрофобная (лотос-эффект > 150)")

    print(f"\n  Угол наклона alpha:")
    print(f"    0 grad       - горизонтальная поверхность (осесимметрия)")
    print(f"    1-30 grad    - умеренный наклон")
    print(f"    > 45 grad    - высокий наклон, риск соскальзывания")
    print()

    while True:
        volume_ul = _prompt_float("  V, мкл", 0.1, 10000, default=10.0)
        theta_deg = _prompt_float("  theta, grad", 1.0,  179.0, default=100.0)
        alpha_deg = _prompt_float("  alpha, grad", 0.0,   89.9, default=0.0)
        print()

        try:
            result = run_simulation(volume_ul, alpha_deg, theta_deg,
                                    fluid=fluid, width_mm=width_mm)
        except DropSlidesError as e:
            print(f"  ОШИБКА: {e}")
            print("  Попробуйте уменьшить объём или угол наклона.\n")
            continue
        except DropPhysicsError as e:
            print(f"  ОШИБКА: {e}\n")
            continue

        for w in result["warnings"]:
            print(f"  WARN: {w}")

        g = result["geometry"]
        print(f"\n{'-'*54}")
        print(f"  Жидкость:          {g['fluid']}")
        print(f"  Объём:             {g['volume_ul']:.2f} мкл")
        print(f"  Угол наклона alpha:  {g['alpha_deg']:.1f} grad")
        print(f"  Краевой угол theta:  {g['theta_Y_deg']:.1f} grad")
        if abs(g["theta_A_deg"] - g["theta_R_deg"]) > 0.1:
            print(f"  thetaA / thetaR:     {g['theta_A_deg']:.1f} / {g['theta_R_deg']:.1f} grad")
        print(f"  Ширина основания:    {g['base_width_mm']:.2f} мм")
        print(f"  Высота вершины:      {g['apex_height_mm']:.2f} мм")
        print(f"  Капилл. длина a:     {g['capillary_length_mm']:.2f} мм")
        print(f"  Число Бонда Bo:      {g['bond_number']:.3f}")
        if g.get("V_max_ul"):
            print(f"  V_max при alpha:     {g['V_max_ul']:.1f} мкл")
        print(f"{'-'*54}\n")

        if do_3d:
            from render3d import render_and_save
            render_and_save(result, path=save, view="iso")
        else:
            out = save or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "drop_result.pdf")
            plot_sideview(result, show=False, save_path=out)
            print(f"  Сохранено: {out}")

        again = input("\n  Ещё раз? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Симуляция формы капли жидкости",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--volume", type=float, default=None,   help="объём, мкл (без - интерактивный режим)")
    parser.add_argument("--alpha",  type=float, default=0.0,    help="угол наклона, °")
    parser.add_argument("--theta",  type=float, default=None,   help="краевой угол, ° (без - интерактивный режим)")
    parser.add_argument("--fluid",  default="water",
                        choices=list(FLUIDS.keys()),             help="жидкость")
    parser.add_argument("--width",  type=float, default=None,   help="ширина основания, мм")
    parser.add_argument("--3d",     dest="do_3d", action="store_true",
                                                                 help="добавить 3D-рендер")
    parser.add_argument("--save",   default=None,               help="путь к файлу вывода")
    parser.add_argument("--cv",     action="store_true",
                                                                 help="режим CV-сравнения: ручной ввод данных dropcv и сравнение с теорией")
    parser.add_argument("--cv-auto", dest="cv_auto", action="store_true",
                                                                 help="авто: читает out/cv_result.json и строит реальную каплю")
    parser.add_argument("--cv-json", dest="cv_json", default=None,
                                                                 help="путь к cv_result.json (для --cv-auto)")
    args = parser.parse_args()

    # авто-режим: читаем JSON и воспроизводим каплю (+ --cv-auto для совместимости)
    # Запуск без аргументов = авто-режим (нет интерактивного ввода)
    no_args = (args.volume is None and args.theta is None
               and not args.cv and not args.cv_auto)
    if args.cv_auto or no_args:
        _cv_auto_mode(args.cv_json)
        sys.exit(0)

    # режим CV-сравнения (ручной ввод)
    if args.cv:
        _cv_compare_mode()
        sys.exit(0)

    fluid = FLUIDS[args.fluid]

    # интерактивный режим если volume или theta не заданы явно
    if args.volume is None or args.theta is None:
        _interactive_mode(fluid, args.do_3d, args.save, args.width)
        sys.exit(0)

    try:
        result = run_simulation(args.volume, args.alpha, args.theta,
                                fluid=fluid, width_mm=args.width)
    except DropPhysicsError as e:
        print(f"\nОШИБКА: {e}\n", file=sys.stderr)
        sys.exit(1)

    for w in result["warnings"]:
        print(f"WARN: {w}")

    g = result["geometry"]
    print(f"\n{'-'*50}")
    print(f"  Жидкость:          {g['fluid']}")
    print(f"  Объём:             {g['volume_ul']:.2f} мкл")
    print(f"  Угол наклона alpha:  {g['alpha_deg']:.1f} grad")
    print(f"  Краевой угол theta:  {g['theta_Y_deg']:.1f} grad")
    if abs(g["theta_A_deg"] - g["theta_R_deg"]) > 0.1:
        print(f"  thetaA / thetaR:     {g['theta_A_deg']:.1f} / {g['theta_R_deg']:.1f} grad")
    print(f"  Ширина основания:    {g['base_width_mm']:.2f} мм")
    print(f"  Высота вершины:      {g['apex_height_mm']:.2f} мм")
    print(f"  Капилл. длина a:     {g['capillary_length_mm']:.2f} мм")
    print(f"  Число Бонда Bo:      {g['bond_number']:.3f}")
    if g.get("V_max_ul"):
        print(f"  V_max при alpha:     {g['V_max_ul']:.1f} мкл")
    print(f"{'-'*50}\n")

    if args.do_3d:
        from render3d import render_and_save
        render_and_save(result, path=args.save, view="iso")
    else:
        out = args.save or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "drop_result.pdf")
        plot_sideview(result, show=False, save_path=out)
        print(f"Сохранено: {out}")

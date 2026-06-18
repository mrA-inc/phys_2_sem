"""
simulate.py
===========
Параметрическая симуляция формы капли с CLI и валидацией физических параметров.

Использование:
    python3 simulate.py --volume 30 --alpha 35 --theta 100 --fluid water
    python3 simulate.py --volume 40 --alpha 0  --theta 112 --3d --save /tmp/drop.pdf

API:
    from simulate import run_simulation
    result = run_simulation(30, 35, 100, fluid=WATER)
"""
import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

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
                   width_mm: float | None = None) -> dict:
    """
    Запускает симуляцию для заданных параметров.
    Возвращает словарь с профилями и геометрией.
    Бросает DropPhysicsError при невозможных параметрах.
    """
    meta = validate_params(volume_ul, alpha_deg, theta_deg, fluid, width_mm)

    V = volume_ul * 1e-9
    theta = np.deg2rad(theta_deg)
    alpha = np.deg2rad(alpha_deg)

    if alpha_deg < 0.5:
        # горизонталь: осесимметрия
        prof = fit_b_for_volume(V, fluid.c, theta)
        thA = thR = theta_deg
        profile_x = prof.x * 1e3    # м → мм
        profile_z = (prof.apex_height - prof.z) * 1e3
    else:
        # наклон: две дуги
        if width_mm is None:
            width_mm = 2.0 * (3 * V / (2 * np.pi)) ** (1 / 3) * 1e3
        w = width_mm * 1e-3
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
    """Вид сбоку: профиль капли с подложкой и аннотациями."""
    g = result["geometry"]
    x = result["profile_x"]
    z = result["profile_z"]
    alpha = result["alpha_deg"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # замкнутый контур (зеркало по x)
    x_full = np.concatenate([-x[::-1], x])
    z_full = np.concatenate([z[::-1], z])

    # поворот на угол наклона для отображения
    a = np.deg2rad(alpha)
    xr = x_full * np.cos(a) + z_full * np.sin(a)
    zr = -x_full * np.sin(a) + z_full * np.cos(a)

    ax.fill(xr, zr, color="#4a90d9", alpha=0.35)
    ax.plot(xr, zr, color="#1a5fa8", lw=2)

    # линия подложки
    hw = g["base_width_mm"] * 0.7
    ax.axline((-hw, 0), (hw, 0), color="#888", lw=1.5)

    ax.set_aspect("equal")
    ax.set_xlabel("x, мм")
    ax.set_ylabel("z, мм")
    title = (
        f"Капля {g['fluid']} | V={g['volume_ul']:.1f} мкл | "
        f"α={g['alpha_deg']:.0f}° | θ={g['theta_Y_deg']:.0f}°\n"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Симуляция формы капли жидкости",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--volume", type=float, required=True,  help="объём, мкл")
    parser.add_argument("--alpha",  type=float, default=0.0,    help="угол наклона, °")
    parser.add_argument("--theta",  type=float, required=True,  help="краевой угол, °")
    parser.add_argument("--fluid",  default="water",
                        choices=list(FLUIDS.keys()),             help="жидкость")
    parser.add_argument("--width",  type=float, default=None,   help="ширина основания, мм")
    parser.add_argument("--3d",     dest="do_3d", action="store_true",
                                                                 help="добавить 3D-рендер")
    parser.add_argument("--save",   default=None,               help="путь к файлу вывода")
    args = parser.parse_args()

    fluid = FLUIDS[args.fluid]

    try:
        result = run_simulation(args.volume, args.alpha, args.theta,
                                fluid=fluid, width_mm=args.width)
    except DropPhysicsError as e:
        print(f"\n❌  {e}\n", file=sys.stderr)
        sys.exit(1)

    for w in result["warnings"]:
        print(f"⚠️   {w}")

    g = result["geometry"]
    print(f"\n{'─'*50}")
    print(f"  Жидкость:          {g['fluid']}")
    print(f"  Объём:             {g['volume_ul']:.2f} мкл")
    print(f"  Угол наклона α:    {g['alpha_deg']:.1f}°")
    print(f"  Краевой угол θ:    {g['theta_Y_deg']:.1f}°")
    if abs(g["theta_A_deg"] - g["theta_R_deg"]) > 0.1:
        print(f"  θA / θR:           {g['theta_A_deg']:.1f}° / {g['theta_R_deg']:.1f}°")
    print(f"  Ширина основания:  {g['base_width_mm']:.2f} мм")
    print(f"  Высота вершины:    {g['apex_height_mm']:.2f} мм")
    print(f"  Капилл. длина a:   {g['capillary_length_mm']:.2f} мм")
    print(f"  Число Бонда Bo:    {g['bond_number']:.3f}")
    if g.get("V_max_ul"):
        print(f"  V_max при α:       {g['V_max_ul']:.1f} мкл")
    print(f"{'─'*50}\n")

    if args.do_3d:
        from render3d import render_and_save
        render_and_save(result, path=args.save, view="iso")
    else:
        out = args.save or "/tmp/drop_result.pdf"
        plot_sideview(result, show=False, save_path=out)
        print(f"Сохранено: {out}")

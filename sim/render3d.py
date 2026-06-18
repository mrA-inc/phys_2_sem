"""
render3d.py
===========
3D-рендер поверхности капли через matplotlib (plot_surface).

Горизонтальная капля  — вращение осесимметричного меридионального профиля.
Наклонная капля       — метод двух окружностей: для каждого азимута ψ
                        строится дуга с локальным краевым углом θ(ψ).

Вызов из CLI:
    python3 simulate.py --volume 40 --alpha 0 --theta 112 --3d --save /tmp/out.pdf

API:
    from render3d import render_and_save
    render_and_save(result, path="/tmp/drop_3d.pdf", view="iso")
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from drop_axisym import DropProfile


# ===========================================================================
# 1. Горизонтальная капля — тело вращения
# ===========================================================================
def surface_axisym(prof: DropProfile, n_phi: int = 120):
    """
    Поверхность осесимметричной капли из меридионального профиля.
    Возвращает (X, Y, Z) — сетки для ax.plot_surface(), координаты в мм.
    Z=0 у основания, Z=apex_height у вершины.
    """
    x = prof.x                            # полуширина, м
    z_h = prof.apex_height - prof.z       # высота над подложкой, м
    phi = np.linspace(0, 2 * np.pi, n_phi)

    X = np.outer(x, np.cos(phi)) * 1e3   # (n_arc, n_phi), мм
    Y = np.outer(x, np.sin(phi)) * 1e3
    Z = np.outer(z_h, np.ones(n_phi)) * 1e3
    return X, Y, Z


# ===========================================================================
# 2. Наклонная капля — метод двух окружностей (приближение)
# ===========================================================================
def surface_inclined(alpha_deg: float, volume_ul: float, theta_Y_deg: float,
                     fluid, width_mm: float,
                     n_phi: int = 90, n_arc: int = 60):
    """
    Приближённая 3D-поверхность наклонной капли.
    Для каждого азимута ψ строится дуга окружности с локальным θ(ψ).
    Возвращает (X, Y, Z) в мм; ось X — вдоль склона (вниз +X), Y — поперёк.
    """
    from inclined import contact_angle_azimuthal, hysteresis_symmetric, two_arc_profile_2d

    V = volume_ul * 1e-9
    theta = np.deg2rad(theta_Y_deg)
    alpha = np.deg2rad(alpha_deg)
    w = width_mm * 1e-3

    thA, thR, _ = hysteresis_symmetric(alpha, V, theta, fluid, w)

    zeta = w / 2 * 1.1
    _, _, info0 = two_arc_profile_2d(zeta, thA, thR, n=n_arc)
    L1, L2 = info0["L1"], info0["L2"]

    psis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    thetas_psi = contact_angle_azimuthal(psis, thA, thR)

    X_all, Y_all, Z_all = [], [], []

    for psi, th_psi in zip(psis, thetas_psi):
        if abs(np.sin(th_psi)) < 1e-9:
            continue
        weight = 0.5 * (1 - np.cos(psi))
        L_long = (1 - weight) * L1 + weight * L2

        r = L_long / np.sin(th_psi)
        H = r * (1 - np.cos(th_psi))

        t = np.linspace(0, th_psi, n_arc)
        xp = r * np.sin(t)
        zp = H - r * (1 - np.cos(t))

        sign = np.cos(psi)
        X_col = sign * xp * np.cos(psi) * 1e3
        Y_col = xp * np.sin(psi) * 1e3
        Z_col = zp * 1e3

        X_all.append(X_col)
        Y_all.append(Y_col)
        Z_all.append(Z_col)

    X = np.column_stack(X_all)
    Y = np.column_stack(Y_all)
    Z = np.column_stack(Z_all)
    return X, Y, Z


# ===========================================================================
# 3. Рисование одного 3D-вида
# ===========================================================================
def plot_drop_3d(ax, X, Y, Z, alpha_deg: float = 0.0, view: str = "iso"):
    """
    Рисует 3D-поверхность капли на осях ax (Axes3D).
    view: 'iso' | 'side' | 'top' | 'front'
    """
    views = {
        "iso":   (30, 45),
        "side":  (5,  0),
        "top":   (90, 0),
        "front": (10, 90),
    }
    elev, azim = views.get(view, (30, 45))
    ax.view_init(elev=elev, azim=azim)

    ax.plot_surface(X, Y, Z, cmap="Blues", alpha=0.82,
                    linewidth=0, antialiased=True, rcount=60, ccount=60)

    # подложка
    lim = max(np.abs(X).max(), np.abs(Y).max()) * 1.25
    xs = np.linspace(-lim, lim, 4)
    ys = np.linspace(-lim, lim, 4)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = -Xg * np.tan(np.deg2rad(alpha_deg))
    ax.plot_surface(Xg, Yg, Zg, color="gray", alpha=0.25,
                    linewidth=0, rcount=2, ccount=2)

    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_zlabel("z, мм")
    ax.set_box_aspect([1, 1, 0.5])


# ===========================================================================
# 4. Комплексный рендер (2×2)
# ===========================================================================
def render_and_save(result: dict, path: str | None = None, view: str = "iso"):
    """
    Строит рисунок 2×2: вид сбоку + три 3D-вида.
    result — словарь из run_simulation().
    """
    from simulate import plot_sideview

    g = result["geometry"]
    alpha = g["alpha_deg"]
    fluid = result["fluid"]

    if alpha < 0.5:
        prof = result["axisym_profile"]
        X, Y, Z = surface_axisym(prof)
    else:
        X, Y, Z = surface_inclined(
            alpha, g["volume_ul"], g["theta_Y_deg"], fluid,
            g["base_width_mm"],
        )

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f"Капля {g['fluid']} | V={g['volume_ul']:.1f} мкл | "
        f"α={alpha:.0f}° | θ={g['theta_Y_deg']:.0f}°",
        fontsize=12,
    )

    ax0 = fig.add_subplot(2, 2, 1)
    plot_sideview(result, ax=ax0, show=False)
    ax0.set_title("Вид сбоку")

    for idx, (v, title) in enumerate([
        ("iso",  "Изометрия"),
        ("side", "Сбоку (3D)"),
        ("top",  "Сверху (3D)"),
    ]):
        ax3 = fig.add_subplot(2, 2, idx + 2, projection="3d")
        plot_drop_3d(ax3, X, Y, Z, alpha_deg=alpha, view=v)
        ax3.set_title(title, fontsize=10)

    plt.tight_layout()

    out = path or "/tmp/drop_3d.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"3D-рендер сохранён: {out}")

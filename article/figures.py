r"""
figures.py
==========
Генерация всех рисунков статьи из провалидированного симуляционного ядра.
Сохраняет векторные PDF (+ PNG-превью) в /home/claude/droplet/figs/.
Кириллица берётся из DejaVu Sans (шрифт matplotlib по умолчанию).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from physics import (WATER, ETHYLENE_GLYCOL, GLYCERIN, Fluid,
                     sphere_cap_volume_from_R, sphere_cap_base_radius,
                     sphere_cap_height)
from drop_2d import (profile_2d_exact, area_2d_exact, height_for_area_2d,
                     integrate_arclength, saturation_height,
                     area_2d_arclength, profile_2d_for_area)
from drop_axisym import axisym_profile, fit_b_for_volume, drop_volume, equivalent_diameter
from inclined import (contact_angle_azimuthal, two_circle_volume,
                      spherical_cap_volume_equal, sliding_angle_2d, max_volume_3d,
                      hysteresis_symmetric, two_arc_profile_2d)

FIGDIR = "/home/claude/droplet/figs"
os.makedirs(FIGDIR, exist_ok=True)

# --- единый стиль ---
rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "legend.framealpha": 0.92,
    "legend.fontsize": 9.5,
    "axes.titlesize": 12,
    "figure.constrained_layout.use": True,
})
C = dict(blue="#1f5fb0", red="#c0392b", green="#1e8449", orange="#e08a1e",
         purple="#7d3c98", gray="#555555", teal="#138d90")


def _save(fig, name):
    pdf = os.path.join(FIGDIR, name + ".pdf")
    png = os.path.join(FIGDIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# ---------------------------------------------------------------------------
# 1. 2D-профили на горизонтали для разных объёмов (площадей сечения)
# ---------------------------------------------------------------------------
def fig_2d_profiles():
    fl = WATER
    theta = np.deg2rad(110.0)
    a = fl.capillary_length
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    areas = [2e-6, 8e-6, 20e-6, 60e-6, 150e-6]      # м^2 (на единицу длины)
    colors = [C["blue"], C["teal"], C["green"], C["orange"], C["red"]]
    for A, col in zip(areas, colors):
        prof = profile_2d_for_area(theta, A, fl.c)
        h = prof.apex_height
        # профиль: x от оси, z вниз от вершины -> высота = h - z
        x = prof.x; z = prof.apex_height - prof.z
        X = np.concatenate([-x[::-1], x]) * 1e3
        Z = np.concatenate([z[::-1], z]) * 1e3
        ax.plot(X, Z, color=col, label=f"$S$ = {A*1e6:.0f} мм$^2$, $h$ = {h*1e3:.2f} мм")
    h_star = saturation_height(fl, theta) * 1e3
    ax.axhline(h_star, ls="--", color=C["gray"], lw=1.3,
               label=f"$h^* = 2a\\,\\sin(\\theta/2)$ = {h_star:.2f} мм")
    ax.set_xlabel("$x$, мм")
    ax.set_ylabel("$z$, мм")
    ax.set_title(f"2D-профиль капли на горизонтали ({fl.name}, $\\theta$={np.rad2deg(theta):.0f}°)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5)
    _save(fig, "fig_2d_profiles")


# ---------------------------------------------------------------------------
# 2. Насыщение высоты: apex height vs объём (площадь) -> h*
# ---------------------------------------------------------------------------
def fig_height_saturation():
    theta = np.deg2rad(110.0)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for fl, col in ((WATER, C["blue"]), (GLYCERIN, C["green"]),
                    (ETHYLENE_GLYCOL, C["orange"])):
        # прямой проход по кривизне вершины b: получаем (площадь, высота) без поиска корня
        bs = np.logspace(np.log10(40.0), np.log10(6000.0), 45)
        areas, hs = [], []
        for b in bs:
            prof = integrate_arclength(b, fl.c, theta, axisymmetric=False)
            areas.append(area_2d_arclength(prof) * 1e6)   # мм^2
            hs.append(prof.apex_height * 1e3)              # мм
        order = np.argsort(areas)
        areas = np.array(areas)[order]; hs = np.array(hs)[order]
        ax.plot(areas, hs, color=col, label=f"{fl.name} ($a$={fl.capillary_length*1e3:.2f} мм)")
        hstar = saturation_height(fl, theta) * 1e3
        ax.axhline(hstar, ls="--", color=col, lw=1.1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("площадь сечения $S$, мм$^2$  (аналог объёма)")
    ax.set_ylabel("высота вершины $h$, мм")
    ax.set_title(f"Насыщение высоты к пределу $h^*=2a\\sin(\\theta/2)$ ($\\theta$={np.rad2deg(theta):.0f}°)")
    ax.legend(loc="lower right")
    _save(fig, "fig_height_saturation")


# ---------------------------------------------------------------------------
# 3. 3D осесимметричные профили для разных чисел Бонда
# ---------------------------------------------------------------------------
def fig_axisym_bond():
    fl = WATER
    theta = np.deg2rad(100.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    vols_ul = [1, 5, 20, 50, 100]
    colors = [C["blue"], C["teal"], C["green"], C["orange"], C["red"]]
    for V_ul, col in zip(vols_ul, colors):
        V = V_ul * 1e-9
        prof = fit_b_for_volume(V, fl.c, theta)
        Bo = fl.bond_number(prof.base_radius)
        X = np.concatenate([-prof.x[::-1], prof.x]) * 1e3
        # z вниз от вершины -> отображаем высоту = (apex - z)
        Zh = (prof.apex_height - np.concatenate([prof.z[::-1], prof.z])) * 1e3
        ax1.plot(X, Zh, color=col,
                 label=f"{V_ul} мкл, Bo={Bo:.2f}")
    ax1.set_xlabel("$x$, мм"); ax1.set_ylabel("высота, мм")
    ax1.set_title("Осесимметричные профили (вода)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper right", fontsize=8.5)

    # правый: высота и диаметр основания vs объём, сравнение со сфер. шапкой
    vols = np.logspace(np.log10(0.3e-9), np.log10(200e-9), 30)
    h_num, d_num, h_cap = [], [], []
    for V in vols:
        prof = fit_b_for_volume(V, fl.c, theta)
        h_num.append(prof.apex_height * 1e3)
        d_num.append(2 * prof.base_radius * 1e3)
        # сфер. шапка того же объёма
        R = (3*V/(np.pi*(2-3*np.cos(theta)+np.cos(theta)**3)))**(1/3)
        h_cap.append(sphere_cap_height(R, theta) * 1e3)
    ax2.plot(vols*1e9, h_num, color=C["blue"], label="высота $h$ (с гравитацией)")
    ax2.plot(vols*1e9, h_cap, ls="--", color=C["blue"], alpha=0.7, label="высота $h$ (сфер. шапка)")
    ax2.plot(vols*1e9, d_num, color=C["red"], label="диаметр основания $2x_0$")
    ax2.set_xscale("log")
    ax2.set_xlabel("объём $V$, мкл"); ax2.set_ylabel("размер, мм")
    ax2.set_title("Отклонение от сферической шапки с ростом $V$")
    ax2.legend(loc="upper left", fontsize=9)
    _save(fig, "fig_axisym_bond")


# ---------------------------------------------------------------------------
# 4. Кубическое распределение краевого угла по периметру
# ---------------------------------------------------------------------------
def fig_angle_distribution():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    psi = np.linspace(0, 2*np.pi, 400)
    cases = [(120, 60), (130, 80), (110, 95)]
    colors = [C["red"], C["green"], C["blue"]]
    for (tmax, tmin), col in zip(cases, colors):
        th = contact_angle_azimuthal(psi, np.deg2rad(tmax), np.deg2rad(tmin))
        ax.plot(np.rad2deg(psi), np.rad2deg(th), color=col,
                label=f"$\\theta_{{max}}$={tmax}°, $\\theta_{{min}}$={tmin}°")
    ax.set_xlabel("азимут $\\psi$, град (0 — вниз по склону)")
    ax.set_ylabel("краевой угол $\\theta(\\psi)$, град")
    ax.set_title("Распределение краевого угла по периметру (кубика, ElSherbini)")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend()
    _save(fig, "fig_angle_distribution")


# ---------------------------------------------------------------------------
# 5. Отклонение сферической шапки от двух окружностей vs гистерезис
# ---------------------------------------------------------------------------
def fig_sphcap_deviation():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    beta = 1.2
    rb_eq = 2.0e-3
    w = rb_eq / np.sqrt(beta); L = rb_eq * np.sqrt(beta)
    means = [70, 90, 110, 130]
    colors = [C["blue"], C["teal"], C["orange"], C["red"]]
    hyst = np.arange(0, 81, 5)
    for m, col in zip(means, colors):
        dev = []
        hh = []
        for hd in hyst:
            tmax = np.deg2rad(m + hd/2); tmin = np.deg2rad(m - hd/2)
            if m - hd/2 < 5:
                continue
            Vtc = two_circle_volume(tmax, tmin, L, w, n_phi=360)
            Vcap = spherical_cap_volume_equal(tmax, tmin, L, w)
            dev.append((Vcap - Vtc)/Vtc*100); hh.append(hd)
        ax.plot(hh, dev, color=col, marker="o", ms=3,
                label=f"средний угол {m}°")
    ax.axhline(0, color=C["gray"], lw=0.8)
    ax.set_xlabel("гистерезис $\\theta_{max}-\\theta_{min}$, град")
    ax.set_ylabel("отклонение объёма $\\dfrac{V_{cap}-V_{2C}}{V_{2C}}$, %")
    ax.set_title("Ошибка приближения сферической шапки ($\\beta$=1.2)")
    ax.legend()
    _save(fig, "fig_sphcap_deviation")


# ---------------------------------------------------------------------------
# 6. Гистерезис θA, θR vs угол наклона (симметричная модель)
# ---------------------------------------------------------------------------
def fig_hysteresis_incline():
    fl = WATER
    theta_Y = np.deg2rad(95.0)
    # материальные пределы (симметрично относительно theta_Y): гистерезис +-28 deg
    dlim = np.deg2rad(28.0)
    thA_max = theta_Y + dlim
    thR_min = theta_Y - dlim
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    vols_ul = [10, 30, 60]
    colors = [C["blue"], C["green"], C["red"]]
    for V_ul, col in zip(vols_ul, colors):
        V = V_ul * 1e-9
        w = 2.0 * (3*V/(2*np.pi))**(1/3)
        alphas = np.deg2rad(np.linspace(0, 90, 400))
        tA, tR, a_plot = [], [], []
        a_slide = None
        for a in alphas:
            A, R, slid = hysteresis_symmetric(a, V, theta_Y, fl, w)
            if slid:
                break
            # достигнут материальный предел -> соскальзывание
            if A >= thA_max or R <= thR_min:
                a_slide = np.rad2deg(a)
                tA.append(np.rad2deg(min(A, thA_max)))
                tR.append(np.rad2deg(max(R, thR_min)))
                a_plot.append(np.rad2deg(a))
                break
            tA.append(np.rad2deg(A)); tR.append(np.rad2deg(R)); a_plot.append(np.rad2deg(a))
        lbl = f"{V_ul} мкл" + (f", скольжение при {a_slide:.0f}°" if a_slide else "")
        ax.plot(a_plot, tA, color=col, label=lbl)
        ax.plot(a_plot, tR, color=col, ls="--")
        if a_slide is not None:
            ax.plot(a_slide, tA[-1], "o", color=col, ms=7, zorder=5)
            ax.plot(a_slide, tR[-1], "o", color=col, ms=7, zorder=5)
    ax.axhline(np.rad2deg(theta_Y), color=C["gray"], lw=0.9, ls=":")
    ax.axhline(np.rad2deg(thA_max), color="k", lw=1.0, alpha=0.5)
    ax.axhline(np.rad2deg(thR_min), color="k", lw=1.0, alpha=0.5)
    ax.text(91, np.rad2deg(thA_max), "$\\theta_{A}^{\\max}$", va="center", fontsize=9)
    ax.text(91, np.rad2deg(thR_min), "$\\theta_{R}^{\\min}$", va="center", fontsize=9)
    ax.text(91, np.rad2deg(theta_Y), "$\\theta_Y$", va="center", color=C["gray"], fontsize=9)
    ax.set_xlabel("угол наклона $\\alpha$, град")
    ax.set_ylabel("краевые углы $\\theta_A$ (—), $\\theta_R$ (- -), град")
    ax.set_title("Развитие гистерезиса с наклоном (вода, $\\theta_Y$=95°)\nкружок — начало соскальзывания (достигнут материальный предел)")
    ax.set_xlim(0, 96)
    ax.legend(loc="lower left", fontsize=8.5)
    _save(fig, "fig_hysteresis_incline")


# ---------------------------------------------------------------------------
# 7. Критическое соскальзывание: alpha_c vs объём; V_max vs alpha
# ---------------------------------------------------------------------------
def fig_critical_sliding():
    fl = WATER
    thA, thR = np.deg2rad(100.0), np.deg2rad(65.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # alpha_c vs объём
    vols = np.linspace(2e-9, 120e-9, 120)
    ac = []
    for V in vols:
        w = 2.0 * (3*V/(2*np.pi))**(1/3)
        found = np.nan
        for a_deg in np.linspace(0.5, 90, 600):
            if max_volume_3d(np.deg2rad(a_deg), thA, thR, fl, w) <= V:
                found = a_deg; break
        ac.append(found)
    ax1.plot(vols*1e9, ac, color=C["blue"])
    ax1.set_xlabel("объём $V$, мкл"); ax1.set_ylabel("критический угол $\\alpha_c$, град")
    ax1.set_title("Угол начала соскальзывания\n(вода, $\\theta_A$=100°, $\\theta_R$=65°)")
    ax1.set_ylim(0, 90)

    # V_max vs alpha для разных гистерезисов
    alphas = np.deg2rad(np.linspace(2, 90, 200))
    for (tA_d, tR_d), col in zip([(100,65),(95,75),(110,60)],
                                 [C["blue"], C["green"], C["red"]]):
        tA, tR = np.deg2rad(tA_d), np.deg2rad(tR_d)
        Vmax = []
        for a in alphas:
            # эквивалентная ширина для оценки (берём фиксированную w=4 мм для наглядности)
            Vmax.append(max_volume_3d(a, tA, tR, fl, 4e-3) * 1e9)
        ax2.plot(np.rad2deg(alphas), Vmax, color=col,
                 label=f"$\\theta_A$={tA_d}°, $\\theta_R$={tR_d}°")
    ax2.set_yscale("log")
    ax2.set_xlabel("угол наклона $\\alpha$, град"); ax2.set_ylabel("$V_{max}$, мкл ($w$=4 мм)")
    ax2.set_title("Макс. удерживаемый объём (Фурмидж)")
    ax2.legend(fontsize=9)
    _save(fig, "fig_critical_sliding")


# ---------------------------------------------------------------------------
# 8. Формы капли на наклоне (две дуги, наклонено на alpha)
# ---------------------------------------------------------------------------
def fig_inclined_shapes():
    fl = WATER
    theta_Y = np.deg2rad(95.0)
    V = 30e-9
    w = 2.0 * (3*V/(2*np.pi))**(1/3)
    zeta = w / 2.0 * 1.15      # полудлина базы (немного вытянута)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    for ax, a_deg in zip(axes, [0, 20, 40, 60]):
        a = np.deg2rad(a_deg)
        A, R, slid = hysteresis_symmetric(a, V, theta_Y, fl, w)
        if slid:
            A, R = np.deg2rad(theta_Y*0+115), np.deg2rad(60)  # предельные
        x, y, info = two_arc_profile_2d(zeta, A, R, n=200)
        # поворот на угол наклона (downhill — вправо)
        ca, sa = np.cos(-a), np.sin(-a)
        xr = x*ca - y*sa
        yr = x*sa + y*ca
        ax.plot(xr*1e3, yr*1e3, color=C["blue"], lw=2)
        ax.fill(xr*1e3, yr*1e3, color=C["blue"], alpha=0.15)
        # линия поверхности
        L = max(zeta*1.6, 3e-3)
        sx = np.array([-L, L]); sy = np.array([0, 0])
        sxr = sx*ca - sy*sa; syr = sx*sa + sy*ca
        ax.plot(sxr*1e3, syr*1e3, color=C["gray"], lw=1.5)
        ttl = f"$\\alpha$={a_deg}°"
        if not slid:
            ttl += f"\n$\\theta_A$={np.rad2deg(A):.0f}°, $\\theta_R$={np.rad2deg(R):.0f}°"
        ax.set_title(ttl, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-4.3, 4.3); ax.set_ylim(-4.3, 3.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Форма капли воды (30 мкл) на наклонной поверхности (метод двух дуг)", fontsize=12)
    _save(fig, "fig_inclined_shapes")


def main():
    print("Генерация рисунков ->", FIGDIR)
    fig_2d_profiles()
    fig_height_saturation()
    fig_axisym_bond()
    fig_angle_distribution()
    fig_sphcap_deviation()
    fig_hysteresis_incline()
    fig_critical_sliding()
    fig_inclined_shapes()
    print("Готово.")


if __name__ == "__main__":
    main()

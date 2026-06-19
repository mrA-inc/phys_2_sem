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
                     n_phi: int = 120, n_arc: int = 80):
    """
    Приближённая 3D-поверхность наклонной капли.

    Строится из того же 2D-профиля двух дуг, что и боковой вид: downhill-сторона
    (psi=0) — дуга C1 с краевым углом theta_A, uphill (psi=pi) — дуга C2 с theta_R.
    Ключевое отличие от наивного «купола»: вершина капли СМЕЩЕНА по X (в сторону
    uphill), потому что downhill-радиус L1 != uphill-радиус L2. Каждый азимутальный
    меридиан — отдельная дуга, проходящая через общую вершину (x_apex, H), что и
    даёт видимую асимметрию формы, а не симметричный полусферический купол.

    Возвращает (X, Y, Z) в мм; ось X — вдоль склона (вниз +X), Y — поперёк.
    """
    from inclined import contact_angle_azimuthal, hysteresis_symmetric, two_arc_profile_2d

    V = volume_ul * 1e-9
    theta = np.deg2rad(theta_Y_deg)
    alpha = np.deg2rad(alpha_deg)
    w = width_mm * 1e-3

    thA, thR, slid = hysteresis_symmetric(alpha, V, theta, fluid, w)
    if slid or thA is None:
        thA, thR = theta + np.deg2rad(20), theta - np.deg2rad(20)

    # Опорный 2D-профиль (от uphill-контакта через вершину к downhill-контакту).
    zeta = w / 2 * 1.1
    _, _, info0 = two_arc_profile_2d(zeta, thA, thR, n=n_arc)
    L1, L2, H = info0["L1"], info0["L2"], info0["H"]
    # Вершина по X: контакт downhill в +L1, uphill в -L2; вершина (общая точка
    # касания дуг) лежит над точкой x_apex со смещением в сторону более пологой
    # (uphill) дуги. Для дуги C1: центр в x=0, apex над x=0 -> apex_x=0 в системе
    # two_arc. Берём смещение из положения проекции вершины: x_apex = (L1 - L2)/2.
    x_apex = 0.5 * (L1 - L2)

    psis = np.linspace(0, 2 * np.pi, n_phi)
    # Полудлина основания в направлении psi: downhill (psi=0) -> L1, uphill -> L2.
    # Эллиптическая интерполяция радиуса основания по азимуту.
    weight = 0.5 * (1 - np.cos(psis))           # 0 при psi=0 (down), 1 при psi=pi (up)
    L_psi = (1 - weight) * L1 + weight * L2
    thetas_psi = contact_angle_azimuthal(psis, thA, thR)

    X_all, Y_all, Z_all = [], [], []

    for psi, L_long, th_psi in zip(psis, L_psi, thetas_psi):
        th_psi = max(th_psi, np.deg2rad(2.0))
        r = L_long / np.sin(th_psi)             # радиус дуги этого меридиана
        # дуга от вершины (t=0) до контакта (t=th_psi)
        t = np.linspace(0, th_psi, n_arc)
        s = r * np.sin(t)                        # отступ от вершины вдоль основания
        zp = H * (1 - (1 - np.cos(t)) / (1 - np.cos(th_psi)))  # высота: H у вершины -> 0 у контакта

        # точка основания в направлении psi от вершины, смещённой на x_apex по X
        X_col = (x_apex + s * np.cos(psi)) * 1e3
        Y_col = (s * np.sin(psi)) * 1e3
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

    # Поворачиваем каплю на угол наклона вокруг оси Y (склон вниз по +X),
    # чтобы её основание лежало в наклонной плоскости подложки, а не «висело»
    # горизонтально. Без поворота наклон капли визуально теряется и форма
    # кажется симметричным куполом.
    a = np.deg2rad(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    Xr = X * ca + Z * sa
    Zr = -X * sa + Z * ca

    ax.plot_surface(Xr, Y, Zr, cmap="Blues", alpha=0.82,
                    linewidth=0, antialiased=True, rcount=60, ccount=60)

    # подложка — наклонная плоскость, в которой лежит основание капли
    lim = max(np.abs(Xr).max(), np.abs(Y).max()) * 1.25
    xs = np.linspace(-lim, lim, 4)
    ys = np.linspace(-lim, lim, 4)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = -Xg * np.tan(a)
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

    out = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "drop_3d.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"3D-рендер сохранён: {out}")


# ===========================================================================
# 5. Интерактивный CLI (запуск напрямую: python render3d.py)
#    Вводятся ТОЛЬКО объём и угол наклона. Жидкость — вода (известный sigma),
#    равновесный краевой угол воды на стекле theta_Y фиксирован (материаловедение).
#    Асимметрия наклонной капли (theta_A != theta_R) вычисляется из физики
#    гистерезиса (hysteresis_symmetric), а не задаётся пользователем.
# ===========================================================================
THETA_WATER_GLASS_DEG = 72.8   # равновесный краевой угол воды на стекле

if __name__ == "__main__":
    import argparse
    import io as _io
    if sys.platform == "win32":
        sys.stdin = _io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig", errors="replace")

    # render3d нуждается в simulate; добавляем exp/ в путь если нужно
    _exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp")
    if os.path.isdir(_exp_dir):
        sys.path.insert(0, os.path.abspath(_exp_dir))
    from physics import WATER

    parser = argparse.ArgumentParser(
        description="3D-рендер капли воды: вводятся только объём и угол наклона.",
    )
    parser.add_argument("--volume", type=float, default=None, help="объём, мкл")
    parser.add_argument("--alpha",  type=float, default=None, help="угол наклона, °")
    parser.add_argument("--theta",  type=float, default=THETA_WATER_GLASS_DEG,
                        help="равновесный краевой угол, ° (по умолч. вода/стекло 72.8)")
    parser.add_argument("--view",   default="iso",
                        choices=["iso", "side", "top", "front"],
                        help="проекция 3D-вида")
    parser.add_argument("--save",   default=None, help="путь сохранения (PDF/PNG)")
    args = parser.parse_args()

    fluid = WATER
    a_mm = fluid.capillary_length * 1e3
    theta_deg = args.theta

    def _ask(prompt, lo, hi, default):
        hint = f"  ({lo}-{hi}, Enter={default}): "
        while True:
            raw = input(prompt + hint).strip().lstrip("﻿")
            if raw == "":
                return default
            try:
                v = float(raw)
            except ValueError:
                print("    Введите число.")
                continue
            if not (lo <= v <= hi):
                print(f"    Вне диапазона [{lo}, {hi}].")
                continue
            return v

    # интерактивно спрашиваем ТОЛЬКО объём и наклон
    if args.volume is None or args.alpha is None:
        print(f"\n{'='*60}")
        print(f"  3D-рендер капли воды")
        print(f"  sigma = {fluid.sigma*1e3:.1f} мН/м (известный),  a = {a_mm:.2f} мм")
        print(f"  краевой угол воды на стекле theta_Y = {theta_deg:.1f} grad")
        print(f"{'='*60}")
        V_rec = 4 * fluid.capillary_length**3 * (2 * np.pi / 3) * 1e9
        print(f"  Объём V: 0.1-{V_rec:.0f} мкл (Bo<4, умеренная гравитация)")
        print(f"  Наклон alpha: 0 grad = горизонт, >45 grad = высокий риск соскальзывания")
        print(f"  (на наклоне асимметрия theta_A/theta_R считается автоматически)\n")
        volume_ul = args.volume if args.volume is not None else _ask("  V, мкл     ", 0.1, 10000, 10.0)
        alpha_deg = args.alpha  if args.alpha  is not None else _ask("  alpha, grad", 0.0, 89.9, 30.0)
    else:
        volume_ul, alpha_deg = args.volume, args.alpha

    # запуск симуляции
    try:
        from simulate import run_simulation
    except ImportError:
        print("ОШИБКА: simulate.py не найден. Запускайте из sim/ или корня проекта.")
        sys.exit(1)

    print(f"\n  Симуляция: V={volume_ul:.1f} мкл, alpha={alpha_deg:.1f} grad, "
          f"theta_Y={theta_deg:.1f} grad...")
    try:
        result = run_simulation(volume_ul, alpha_deg, theta_deg, fluid=fluid)
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        sys.exit(1)

    for w in result.get("warnings", []):
        print(f"  WARN: {w}")
    g = result["geometry"]
    print(f"  Bo={g['bond_number']:.3f}, основание={g['base_width_mm']:.2f} мм, "
          f"h={g['apex_height_mm']:.2f} мм")
    if abs(g["theta_A_deg"] - g["theta_R_deg"]) > 0.1:
        print(f"  Асимметрия: theta_A(downhill)={g['theta_A_deg']:.1f} grad, "
              f"theta_R(uphill)={g['theta_R_deg']:.1f} grad")

    out = args.save or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "drop_3d.pdf"
    )
    render_and_save(result, path=out, view=args.view)
    print(f"  Готово: {out}")

r"""
inclined.py
===========
Капля на НАКЛОННОЙ поверхности: гистерезис краевого угла, условие начала
соскальзывания, метод двух окружностей (ElSherbini 2004) для объёма 3D-капли.

Содержит три блока:

A. Критерии удержания / соскальзывания
   - 2D (Френкель 1948):     rho g V sin(alpha) = sigma (cos theta_R - cos theta_A)
   - 3D (Фурмидж/Extrand):   rho g V sin(alpha) = k w sigma (cos theta_R - cos theta_A)
   Отсюда критический угол наклона alpha_c или максимальный объём V_max.

B. Распределение краевого угла по периметру (ElSherbini, ур. 41):
   cos theta(psi) = 2 D/pi^3 psi^3 - 3 D/pi^2 psi^2 + cos theta_max,
   D = cos theta_max - cos theta_min, psi отсчитывается от «нижней» точки (downhill).

C. Метод двух окружностей: профиль каждого диаметрального сечения —
   две дуги окружностей с общей касательной в вершине. Радиальный профиль в
   сторону азимута phi — одна дуга (C1 вниз по склону / C2 вверх). Объём:
       V = \int_0^{2pi} m(phi) d phi,    m(phi) = \int_0^{edge} x * y(x) dx   (Паппус).
   Условие общей вершины L1 = Lf * L2 (ур. 19-20) обеспечивает единую высоту.
   Проверка: круговой контур + единый угол -> сферическая шапка.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

from physics import Fluid, sphere_cap_volume_from_base, sphere_cap_volume_from_R


# ===========================================================================
# A. Критерии соскальзывания
# ===========================================================================
def sliding_angle_2d(area: float, theta_A: float, theta_R: float, fluid: Fluid):
    """
    Критический угол наклона для 2D-капли (площадь сечения area, м^2/м):
        sin(alpha_c) = sigma (cos theta_R - cos theta_A) / (rho g area)
    Возвращает alpha_c (рад) или None, если капля удерживается при любом наклоне.
    """
    rhs = fluid.sigma * (np.cos(theta_R) - np.cos(theta_A)) / (fluid.delta_rho * fluid.g * area)
    if rhs >= 1.0:
        return None
    return np.arcsin(rhs)


def max_volume_3d(alpha: float, theta_A: float, theta_R: float, fluid: Fluid,
                  width: float, k: float = 1.0):
    """
    Максимальный объём 3D-капли, удерживаемой на склоне угла alpha (рад):
        V_max = k w sigma (cos theta_R - cos theta_A) / (rho g sin alpha)
    width w — ширина контактного пятна поперёк направления скатывания (м).
    k — численный коэффициент формы (Furmidge k=1; Extrand–Kumagai k~ ).
    """
    if np.sin(alpha) <= 1e-9:
        return np.inf
    return (k * width * fluid.sigma * (np.cos(theta_R) - np.cos(theta_A))
            / (fluid.delta_rho * fluid.g * np.sin(alpha)))


def critical_volume_curve(theta_A: float, theta_R: float, fluid: Fluid,
                          width_func, k: float = 1.0, n: int = 120):
    """Кривая V_max(alpha) для alpha от ~1° до 90°. width_func(V)->w либо const."""
    alphas = np.deg2rad(np.linspace(1.0, 90.0, n))
    Vmax = []
    for a in alphas:
        # если width зависит от V, берём грубую оценку через эквивалентную ширину
        w = width_func if np.isscalar(width_func) else width_func(a)
        Vmax.append(max_volume_3d(a, theta_A, theta_R, fluid, w, k))
    return alphas, np.array(Vmax)


# ===========================================================================
# B. Распределение краевого угла по периметру
# ===========================================================================
def contact_angle_azimuthal(psi: np.ndarray, theta_max: float, theta_min: float):
    """
    Краевой угол как функция азимута psi (рад), отсчитываемого от downhill-точки.
    Кубическая аппроксимация ElSherbini (ур. 41). Симметрия: theta(2pi-psi)=theta(psi).
    """
    psi = np.asarray(psi, dtype=float)
    psi_fold = np.where(psi > np.pi, 2*np.pi - psi, psi)   # [0, pi]
    D = np.cos(theta_max) - np.cos(theta_min)
    cos_t = 2.0 * D / np.pi**3 * psi_fold**3 - 3.0 * D / np.pi**2 * psi_fold**2 + np.cos(theta_max)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return np.arccos(cos_t)


# ===========================================================================
# C. Метод двух окружностей (объём 3D-капли)
# ===========================================================================
def _Lf(theta1: float, theta2: float) -> float:
    """Коэффициент Lf (ElSherbini ур. 20): L1 = Lf * L2, обеспечивает общую вершину."""
    return (np.sin(theta1) * (1.0 - np.cos(theta2))
            / (np.sin(theta2) * (1.0 - np.cos(theta1))))


def ellipse_radius(psi: float, semi_major_L: float, semi_minor_w: float) -> float:
    """Радиус эллипса (от центра до контура) в направлении азимута psi.
    psi=0 -> вдоль большой оси (downhill), psi=pi/2 -> малая ось."""
    return (semi_major_L * semi_minor_w
            / np.sqrt((semi_minor_w * np.cos(psi))**2 + (semi_major_L * np.sin(psi))**2))


def _arc_moment(edge: float, theta: float) -> float:
    r"""
    Первый момент радиального профиля (одной дуги) о вершинной оси:
        m_arc = \int_0^{edge} x * y(x) dx = V_cap_arc / (2 pi),
    где радиальный профиль — меридиан сферической шапки с базовым радиусом `edge`
    и краевым углом theta (радиус сферы r = edge/sin theta). Через объём шапки

        V_cap = (pi/3) r^3 (2 - 3 cos theta + cos^3 theta),

    что корректно и для theta > 90° (нависающая капля), где y(x) многозначна,
    тогда как прямой интеграл по x неприменим.
    """
    if edge <= 0:
        return 0.0
    r = edge / np.sin(theta)
    V_cap = (np.pi / 3.0) * r**3 * (2.0 - 3.0*np.cos(theta) + np.cos(theta)**3)
    return V_cap / (2.0 * np.pi)


def two_circle_volume(theta_max: float, theta_min: float,
                      semi_major_L: float, semi_minor_w: float,
                      n_phi: int = 720) -> float:
    """
    Объём 3D-капли методом двух окружностей.
    Контур — эллипс (semi_major_L вдоль склона, semi_minor_w поперёк).
    Краевой угол меняется по азимуту согласно кубической формуле.
    Интегрируется первый момент радиального профиля по всем азимутам (Паппус).
    """
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    V = 0.0
    for phi in phis:
        # хорда в ориентации (phi mod pi); полудлина от центра до контура
        zeta = ellipse_radius(phi, semi_major_L, semi_minor_w)
        # краевые углы на двух концах хорды
        th_here = contact_angle_azimuthal(np.array([phi]), theta_max, theta_min)[0]
        th_opp = contact_angle_azimuthal(np.array([(phi + np.pi) % (2*np.pi)]),
                                         theta_max, theta_min)[0]
        # downhill-конец имеет больший угол -> это theta1 (C1, меньшая дуга)
        theta1, theta2 = max(th_here, th_opp), min(th_here, th_opp)
        Lf = _Lf(theta1, theta2)
        # L1 + L2 = 2 zeta, L1 = Lf L2
        L2 = 2.0 * zeta / (1.0 + Lf)
        L1 = 2.0 * zeta * Lf / (1.0 + Lf)
        # радиальный профиль в сторону phi: если угол здесь больше — это C1 (edge=L1),
        # иначе C2 (edge=L2)
        if th_here >= th_opp:
            edge, theta_edge = L1, theta1
        else:
            edge, theta_edge = L2, theta2
        V += _arc_moment(edge, theta_edge) * dphi
    return V


def spherical_cap_volume_equal(theta_max: float, theta_min: float,
                               semi_major_L: float, semi_minor_w: float) -> float:
    """
    Объём в приближении сферической шапки для тех же параметров:
    средний угол theta=(theta_max+theta_min)/2, контур — круг той же площади,
    что и эллипс (эквивалентный диаметр D = 2 sqrt(L w)).
    """
    theta = 0.5 * (theta_max + theta_min)
    D = 2.0 * np.sqrt(semi_major_L * semi_minor_w)   # диаметр круга равной площади
    return sphere_cap_volume_from_base(D, theta)


# ===========================================================================
# D. Гистерезис при наклоне (симметричная модель) и форма из двух дуг (2D)
# ===========================================================================
def hysteresis_symmetric(alpha: float, volume: float, theta_Y: float,
                         fluid: Fluid, width: float):
    """
    Симметричная модель гистерезиса (Lv 2017: theta_R=theta-Δθ/2, theta_A=theta+Δθ/2).
    Из баланса Френкеля/Фурмиджа: rho g V sin(alpha) = sigma w (cos theta_R - cos theta_A)
    при theta_A=theta_Y+delta, theta_R=theta_Y-delta:
        cos theta_R - cos theta_A = 2 sin(theta_Y) sin(delta)
    => sin(delta) = rho g V sin(alpha) / (2 sigma w sin theta_Y).
    Возвращает (theta_A, theta_R, slid) — slid=True, если sin(delta)>1 (капля скользит).
    """
    s = (fluid.delta_rho * fluid.g * volume * np.sin(alpha)
         / (2.0 * fluid.sigma * width * np.sin(theta_Y)))
    if s >= 1.0:
        return None, None, True
    delta = np.arcsin(s)
    return theta_Y + delta, theta_Y - delta, False


def two_arc_profile_2d(zeta: float, theta_down: float, theta_up: float,
                       n: int = 200):
    """
    Плоский профиль капли из двух дуг окружностей (метод ElSherbini, 2D),
    с общей касательной в вершине. theta_down (>=) — нижний по склону угол (C1),
    theta_up — верхний (C2). База от -L2 (uphill) до +L1 (downhill), общая высота H.
    Возвращает (x, y) контура (вершина наверху), x от -L2 до +L1.
    """
    theta1, theta2 = max(theta_down, theta_up), min(theta_down, theta_up)
    Lf = _Lf(theta1, theta2)
    L2 = 2.0 * zeta / (1.0 + Lf)
    L1 = 2.0 * zeta * Lf / (1.0 + Lf)
    r1 = L1 / np.sin(theta1)
    r2 = L2 / np.sin(theta2)
    H = r1 * (1.0 - np.cos(theta1))   # = r2 (1 - cos theta2), общая вершина
    # Параметризация по углу вокруг центра дуги (корректно для theta>90, нависание).
    # C1: правая (downhill) дуга, центр (0, H - r1); apex при phi=pi/2,
    #     контакт (L1, 0) при phi = pi/2 - theta1.
    yc1 = H - r1
    phi1 = np.linspace(np.pi/2, np.pi/2 - theta1, n)   # apex -> downhill contact
    x1 = r1 * np.cos(phi1)
    y1 = yc1 + r1 * np.sin(phi1)
    yc2 = H - r2
    phi2 = np.linspace(np.pi/2 + theta2, np.pi/2, n)    # uphill contact -> apex
    x2 = r2 * np.cos(phi2)
    y2 = yc2 + r2 * np.sin(phi2)
    # порядок: левый контакт (uphill) -> вершина -> правый контакт (downhill)
    x = np.concatenate([x2, x1[1:]])
    y = np.concatenate([y2, y1[1:]])
    return x, y, dict(L1=L1, L2=L2, H=H, r1=r1, r2=r2, theta1=theta1, theta2=theta2)


# ===========================================================================
# Проверки
# ===========================================================================
def validate():
    print("=== Проверка метода двух окружностей: круг + единый угол -> шапка ===")
    theta = np.deg2rad(75.0)
    rb = 1.5e-3  # базовый радиус 1.5 мм
    V_tc = two_circle_volume(theta, theta, rb, rb, n_phi=720)
    V_cap = sphere_cap_volume_from_R(rb / np.sin(theta), theta)
    print(f"theta={np.rad2deg(theta):.0f}°, базовый радиус={rb*1e3:.2f} мм")
    print(f"  V (две окружности) = {V_tc*1e9:.5f} мкл")
    print(f"  V (сфер. шапка)    = {V_cap*1e9:.5f} мкл")
    print(f"  расхождение        = {abs(V_tc-V_cap)/V_cap*100:.3f} %")

    print("\n=== Отклонение сферической шапки от двух окружностей vs гистерезис ===")
    print("    (воспроизводим Fig. 9 ElSherbini: beta=1.2)")
    beta = 1.2
    rb_eq = 2.0e-3
    theta_mean = np.deg2rad(90.0)
    for hyst_deg in (0, 20, 40, 50, 60, 80, 100):
        hyst = np.deg2rad(hyst_deg)
        th_max = theta_mean + hyst / 2
        th_min = theta_mean - hyst / 2
        if th_min <= np.deg2rad(2):
            continue
        # эллипс с aspect=beta, эквивалентная площадь как у круга rb_eq
        w = rb_eq / np.sqrt(beta)
        L = rb_eq * np.sqrt(beta)
        V_tc = two_circle_volume(th_max, th_min, L, w)
        V_cap = spherical_cap_volume_equal(th_max, th_min, L, w)
        dev = (V_cap - V_tc) / V_tc * 100
        print(f"  hyst={hyst_deg:3d}°: V_2C={V_tc*1e9:6.3f} мкл, "
              f"V_cap={V_cap*1e9:6.3f} мкл, отклонение={dev:+6.1f} %")

    print("\n=== Критерий соскальзывания (вода) ===")
    fl = Fluid("вода", 998.0, 0.0728)
    thA, thR = np.deg2rad(95.0), np.deg2rad(60.0)
    for V_ul in (5, 20, 50, 100):
        V = V_ul * 1e-9
        # ширина контакта ~ диаметр эквивалентной полусферы
        w = 2.0 * (3*V/(2*np.pi))**(1/3)
        a_c = None
        # перебор: при каком alpha V становится критическим
        for a_deg in np.linspace(1, 90, 900):
            a = np.deg2rad(a_deg)
            if max_volume_3d(a, thA, thR, fl, w) <= V:
                a_c = a_deg
                break
        ac_str = f"{a_c:.1f}°" if a_c else ">90° (держится)"
        print(f"  V={V_ul:4d} мкл (w≈{w*1e3:.2f} мм): крит. угол наклона = {ac_str}")


if __name__ == "__main__":
    validate()

r"""
drop_2d.py
==========
Двумерная (плоская) капля на ГОРИЗОНТАЛЬНОЙ поверхности в поле тяжести.

Реализованы два независимых способа построения профиля (для кросс-проверки):

1) Точное вариационное решение (sessile).
   Минимизация функционала энергии при фиксированной площади сечения
   приводит к первому интегралу уравнения Эйлера–Лагранжа (= Юнга–Лапласа):

        cos phi(z) = 1/sqrt(1+z'^2) = -c z^2/2 + lambda z + cos(theta),     (*)

   где z отсчитывается ВВЕРХ от линии контакта (z=0) до вершины (z=h),
   c = delta_rho g / sigma, а множитель Лагранжа

        lambda = c h / 2 + (1 - cos theta) / h.

   Профиль:  x(z) = \int_z^h  f/sqrt(1-f^2) dt,   f = cos phi.
   (Эквивалентно ур. (31)-(35) Матюхина–Фроленкова 2013.)

2) Численное интегрирование в дуговой параметризации (Bashforth–Adams, плоский случай):
        dx/ds = cos phi, dz/ds = sin phi, dphi/ds = b + c z,
   b — кривизна у вершины. Унифицировано с осесимметричным случаем.

Оба способа дают совпадающие профили (см. validate()).
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.integrate import IntegrationWarning
from scipy.optimize import brentq

from physics import Fluid


# ----------------------------------------------------------------------------
# Способ 1: точное вариационное решение (через высоту h и угол theta)
# ----------------------------------------------------------------------------
def _lambda_of(c: float, h: float, theta: float) -> float:
    return c * h / 2.0 + (1.0 - np.cos(theta)) / h


def profile_2d_exact(theta: float, h: float, c: float, n: int = 400):
    """
    Точный профиль 2D-капли по углу theta (рад) и высоте h (м).
    Возвращает (x, z): половина профиля от вершины (z=h, x=0) к контакту (z=0).
    """
    lam = _lambda_of(c, h, theta)

    def f(z):  # cos(phi)
        return -c * z**2 / 2.0 + lam * z + np.cos(theta)

    # узлы по z от 0 (контакт) до h (вершина)
    z = np.linspace(0.0, h, n)
    # x(z) = интеграл от z до h; у вершины подынтегральное ~ 1/sqrt(h-z)
    # (интегрируемая особенность) — гасим предупреждения quad
    x = np.empty_like(z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for i, zi in enumerate(z):
            val, _ = quad(lambda t: f(t) / np.sqrt(max(1.0 - f(t)**2, 1e-15)),
                          zi, h, limit=200)
            x[i] = val
    return x, z


def area_2d_exact(theta: float, h: float, c: float) -> float:
    """Площадь поперечного сечения 2D-капли (м^2) для точного профиля."""
    x, z = profile_2d_exact(theta, h, c, n=600)
    # полная площадь = 2 * интеграл x dz
    return 2.0 * np.trapezoid(x, z)


def height_for_area_2d(theta: float, area: float, c: float,
                       h_lo: float = 1e-7, h_hi: float = 1.0) -> float:
    """Подобрать высоту h так, чтобы площадь сечения равнялась area."""
    g = lambda h: area_2d_exact(theta, h, c) - area
    # расширяем верхнюю границу при необходимости
    hi = h_hi
    while g(hi) < 0 and hi < 1e3:
        hi *= 2.0
    return brentq(g, h_lo, hi, xtol=1e-12, rtol=1e-10)


# ----------------------------------------------------------------------------
# Способ 2: дуговая параметризация (унифицировано с 3D)
# ----------------------------------------------------------------------------
@dataclass
class DropProfile:
    s: np.ndarray      # дуговая координата
    x: np.ndarray      # горизонталь от оси симметрии
    z: np.ndarray      # вертикаль (вниз от вершины)
    phi: np.ndarray    # угол наклона касательной
    theta: float       # достигнутый краевой угол
    b: float           # кривизна у вершины

    @property
    def base_radius(self) -> float:
        return self.x[-1]

    @property
    def apex_height(self) -> float:
        return self.z[-1]


def integrate_arclength(b: float, c: float, theta_stop: float,
                        axisymmetric: bool, s_max: float = 50.0,
                        n_eval: int = 2000) -> DropProfile:
    """
    Интегрирование профиля от вершины (s=0) до достижения краевого угла theta_stop.
    axisymmetric=True  -> 3D осесимметрия:  dphi/ds = 2b + c z - sin(phi)/x
    axisymmetric=False -> 2D:               dphi/ds = b + c z
    z отсчитывается ВНИЗ от вершины (z>=0), phi растёт от 0.
    """
    def rhs(s, y):
        x, z, phi = y
        if axisymmetric:
            if x < 1e-12:
                dphi = b  # предел sin(phi)/x -> b у вершины
            else:
                dphi = 2.0 * b + c * z - np.sin(phi) / x
        else:
            dphi = b + c * z
        return [np.cos(phi), np.sin(phi), dphi]

    def event_theta(s, y):
        return y[2] - theta_stop
    event_theta.terminal = True
    event_theta.direction = 1

    y0 = [0.0, 0.0, 0.0]
    sol = solve_ivp(rhs, [0.0, s_max], y0, events=event_theta,
                    max_step=s_max / n_eval, rtol=1e-9, atol=1e-12, dense_output=True)
    s = sol.t
    x, z, phi = sol.y
    th = phi[-1]
    return DropProfile(s=s, x=x, z=z, phi=phi, theta=th, b=b)


def saturation_height(fluid: Fluid, theta: float) -> float:
    """
    Предельная (максимальная) высота большой 2D-капли (лужи):
        h* = 2 a sin(theta/2)           [Матюхин 2013, ур. (36); Lv: h ~ 2a sin(theta/2)]
    """
    return 2.0 * fluid.capillary_length * np.sin(theta / 2.0)


def area_2d_arclength(prof: DropProfile) -> float:
    r"""Быстрая площадь сечения 2D-капли по дуговому профилю: S = 2 \int x dz."""
    return 2.0 * np.trapezoid(prof.x, prof.z)


def profile_2d_for_area(theta: float, area: float, c: float,
                        b_lo: float = 1.0, b_hi: float = 1e6) -> DropProfile:
    """
    Быстро подобрать дуговой 2D-профиль с заданной площадью сечения (через b).
    Площадь монотонно убывает с ростом b. Один ODE-прогон на вычисление.
    """
    def area_of_b(b):
        return area_2d_arclength(integrate_arclength(b, c, theta, axisymmetric=False)) - area
    lo, hi = b_lo, b_hi
    while area_of_b(lo) < 0:
        lo /= 2.0
        if lo < 1e-3:
            break
    while area_of_b(hi) > 0:
        hi *= 2.0
        if hi > 1e9:
            break
    b = brentq(area_of_b, lo, hi, xtol=1e-4, rtol=1e-9)
    return integrate_arclength(b, c, theta, axisymmetric=False)


# ----------------------------------------------------------------------------
# Кросс-проверка
# ----------------------------------------------------------------------------
def validate():
    fl = WATER = Fluid("вода", 998.0, 0.0728)
    c = fl.c
    theta = np.deg2rad(80.0)
    h = 1.2e-3  # 1.2 мм

    # exact
    x_ex, z_ex = profile_2d_exact(theta, h, c, n=300)

    # arc-length: подобрать b так, чтобы высота совпала с h
    def apex_h_of_b(b):
        prof = integrate_arclength(b, c, theta, axisymmetric=False)
        return prof.apex_height
    # b ~ 1/радиус; ищем b такой, что apex height = h
    b = brentq(lambda b: apex_h_of_b(b) - h, 1.0, 1e5)
    prof = integrate_arclength(b, c, theta, axisymmetric=False)

    print("=== Кросс-проверка 2D (exact vs arc-length) ===")
    print(f"theta = {np.rad2deg(theta):.1f} deg, h = {h*1e3:.3f} мм, a = {fl.capillary_length*1e3:.3f} мм")
    print(f"  exact:      base half-width = {x_ex[0]*1e3:.4f} мм")
    print(f"  arclength:  base half-width = {prof.base_radius*1e3:.4f} мм")
    print(f"  arclength achieved theta = {np.rad2deg(prof.theta):.3f} deg")
    print(f"  относит. расхождение по ширине: "
          f"{abs(x_ex[0]-prof.base_radius)/x_ex[0]*100:.3f} %")

    # Проверка предельной высоты: большая капля
    big_area = 80e-6  # большая "лужа" в 2D (м^2 на единицу длины)
    h_big = height_for_area_2d(theta, big_area, c)
    print(f"\n  h(большая капля)   = {h_big*1e3:.4f} мм")
    print(f"  h* = 2a sin(θ/2)   = {saturation_height(fl, theta)*1e3:.4f} мм  (предел)")


if __name__ == "__main__":
    from physics import WATER
    validate()

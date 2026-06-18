r"""
drop_axisym.py
==============
Осесимметричная (3D) капля на ГОРИЗОНТАЛЬНОЙ поверхности — уравнение
Бэшфорта–Адамса в дуговой параметризации:

    dx/ds   = cos(phi)
    dz/ds   = sin(phi)
    dphi/ds = 2b + c z - sin(phi)/x            (у вершины sin(phi)/x -> b)

b — кривизна у вершины (1/м), c = delta_rho g / sigma.
Объём тела вращения:   V = \int pi x^2 dz = \int pi x^2 sin(phi) ds.

Проверка: при g -> 0 (c -> 0) профиль превращается в сферическую шапку,
объём совпадает с V = (pi R^3/3)(2 - 3 cos theta + cos^3 theta), R = 1/b.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

from physics import Fluid, sphere_cap_volume_from_R
from drop_2d import integrate_arclength, DropProfile


def drop_volume(prof: DropProfile) -> float:
    """Объём осесимметричной капли (м^3) по профилю (тело вращения вокруг оси z)."""
    return np.trapezoid(np.pi * prof.x**2 * np.sin(prof.phi), prof.s)


def axisym_profile(b: float, c: float, theta: float) -> DropProfile:
    """Осесимметричный профиль для кривизны b у вершины и краевого угла theta."""
    return integrate_arclength(b, c, theta, axisymmetric=True)


def fit_b_for_volume(volume: float, c: float, theta: float,
                     b_lo: float = 1.0, b_hi: float = 1e6) -> DropProfile:
    """
    Подобрать кривизну у вершины b так, чтобы объём капли равнялся volume (м^3)
    при заданном краевом угле theta.
    Чем больше b, тем меньше капля -> объём монотонно убывает по b.
    """
    def vol_of_b(b):
        return drop_volume(axisym_profile(b, c, theta)) - volume

    lo, hi = b_lo, b_hi
    # гарантируем смену знака
    while vol_of_b(lo) < 0:           # капля слишком мала даже при малом b
        lo /= 2.0
        if lo < 1e-3:
            break
    while vol_of_b(hi) > 0:           # капля слишком велика даже при большом b
        hi *= 2.0
        if hi > 1e9:
            break
    b = brentq(vol_of_b, lo, hi, xtol=1e-6, rtol=1e-10)
    return axisym_profile(b, c, theta)


def equivalent_diameter(volume: float) -> float:
    """Эквивалентный диаметр сферы того же объёма."""
    return 2.0 * (3.0 * volume / (4.0 * np.pi))**(1.0 / 3.0)


def validate():
    from physics import WATER
    print("=== Проверка 3D: предел без гравитации -> сферическая шапка ===")
    theta = np.deg2rad(70.0)
    c0 = 1e-9   # практически нулевая гравитация
    b = 1000.0  # 1/м  -> R = 1 мм
    prof = axisym_profile(b, c0, theta)
    V_num = drop_volume(prof)
    V_cap = sphere_cap_volume_from_R(1.0 / b, theta)
    print(f"theta={np.rad2deg(theta):.0f}°, R=1/b={1/b*1e3:.3f} мм")
    print(f"  V (численно)        = {V_num*1e9:.5f} мкл")
    print(f"  V (сфер. шапка)     = {V_cap*1e9:.5f} мкл")
    print(f"  относит. расхождение = {abs(V_num-V_cap)/V_cap*100:.4f} %")

    print("\n=== Реальная капля воды заданного объёма ===")
    for V_ul in (1.0, 5.0, 20.0, 50.0):
        V = V_ul * 1e-9   # мкл -> м^3
        prof = fit_b_for_volume(V, WATER.c, theta)
        D = 2 * prof.base_radius
        h = prof.apex_height
        Bo = WATER.bond_number(prof.base_radius)
        # сравнение с шапкой того же объёма (наивное приближение)
        from physics import sphere_cap_base_radius
        print(f"  V={V_ul:5.1f} мкл: основание 2x0={D*1e3:5.2f} мм, "
              f"h={h*1e3:5.3f} мм, Bo={Bo:.3f}, b у вершины={prof.b:.1f} 1/м")


if __name__ == "__main__":
    validate()

"""
physics.py
==========
Базовые физические величины и определения для задачи о форме капли.

Все формулы и обозначения согласованы со статьёй:
    - капиллярная длина  a = sqrt(sigma / (rho g))
    - число Бонда        Bo = rho g L^2 / sigma = (L/a)^2
    - капиллярная константа c = rho g / sigma = 1 / a^2   (знак '+' для сидящей капли)

Литература:
    [Lv 2017]            arXiv:1705.03548 — точные 2D-решения
    [ElSherbini 2004]    J. Colloid Interface Sci. 273, 566 — метод двух окружностей
    [Rotenberg 1983]     J. Colloid Interface Sci. 93, 169 — ADSA, дуговая параметризация
    [Matyukhin 2013]     Конд. среды и межф. границы 15, 292 — вариационный вывод
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G_EARTH = 9.80665  # м/с^2, стандартное ускорение свободного падения


@dataclass(frozen=True)
class Fluid:
    """Свойства жидкости (фазы) на границе с газом."""
    name: str
    rho: float       # плотность жидкости, кг/м^3
    sigma: float     # коэффициент поверхностного натяжения ж–г, Н/м
    rho_gas: float = 1.20   # плотность окружающего газа, кг/м^3 (воздух при 20 C)
    g: float = G_EARTH

    @property
    def delta_rho(self) -> float:
        """Разность плотностей через границу раздела, кг/м^3."""
        return self.rho - self.rho_gas

    @property
    def capillary_length(self) -> float:
        """Капиллярная длина a = sqrt(sigma / (delta_rho * g)), м."""
        return np.sqrt(self.sigma / (self.delta_rho * self.g))

    @property
    def c(self) -> float:
        """Капиллярная константа c = delta_rho * g / sigma = 1/a^2, 1/м^2."""
        return self.delta_rho * self.g / self.sigma

    def bond_number(self, length: float) -> float:
        """Число Бонда для характерного размера length (м)."""
        return self.delta_rho * self.g * length**2 / self.sigma


# --- Готовые наборы свойств (20 C) ---
WATER = Fluid(name="вода", rho=998.0, sigma=0.0728)
ETHYLENE_GLYCOL = Fluid(name="этиленгликоль", rho=1113.0, sigma=0.0477)
GLYCERIN = Fluid(name="глицерин", rho=1261.0, sigma=0.0634)
MERCURY = Fluid(name="ртуть", rho=13534.0, sigma=0.485, rho_gas=1.2)


def sphere_cap_volume_from_R(R: float, theta: float) -> float:
    """Объём сферического сегмента (шапки) с радиусом сферы R и краевым углом theta."""
    return np.pi * R**3 / 3.0 * (2.0 - 3.0*np.cos(theta) + np.cos(theta)**3)


def sphere_cap_volume_from_base(D: float, theta: float) -> float:
    """
    Объём сферической шапки через диаметр контактного пятна D и угол theta.
    Формула (42) из ElSherbini 2004.
    """
    return np.pi * D**3 / 24.0 * (2.0 - 3.0*np.cos(theta) + np.cos(theta)**3) / np.sin(theta)**3


def sphere_cap_base_radius(R: float, theta: float) -> float:
    """Радиус контактного пятна сферической шапки: x0 = R sin(theta)."""
    return R * np.sin(theta)


def sphere_cap_height(R: float, theta: float) -> float:
    """Высота сферической шапки: h = R (1 - cos theta)."""
    return R * (1.0 - np.cos(theta))


if __name__ == "__main__":
    for f in (WATER, ETHYLENE_GLYCOL, GLYCERIN, MERCURY):
        print(f"{f.name:14s}: a = {f.capillary_length*1e3:6.3f} мм, "
              f"c = {f.c:8.1f} 1/м^2, Bo(1мм) = {f.bond_number(1e-3):.4f}")

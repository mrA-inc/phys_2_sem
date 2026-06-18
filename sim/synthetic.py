r"""
synthetic.py
============
Генерация реалистичного синтетического «экспериментального» снимка капли
(вид сбоку) из физической модели (sim/), для отработки CV-обработки.
Силуэт капли = меридиональный профиль осесимметричной капли (проекция сбоку).
Добавляются: подложка, размытие и шум, имитирующие фотографию.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import numpy as np
import cv2

from physics import WATER, Fluid
from drop_axisym import fit_b_for_volume
from inclined import two_arc_profile_2d, hysteresis_symmetric


def _meridian_axisym(fluid: Fluid, volume: float, theta: float):
    """Меридиональный профиль (x,z) осесимметричной капли: x от оси, z — высота от базы."""
    prof = fit_b_for_volume(volume, fluid.c, theta)
    x = prof.x
    z = prof.apex_height - prof.z      # высота над подложкой
    return x, z, prof


def render_drop(volume=40e-9, theta_deg=110.0, alpha_deg=0.0,
                fluid: Fluid = WATER, px_per_mm=60.0,
                img_w=900, img_h=650, noise=6.0, blur=2.0,
                inclined_hysteresis=True, seed=0):
    """
    Рендер снимка капли. Для alpha>0 используется профиль из двух дуг
    (нижний/верхний краевые углы из симметричной модели гистерезиса).
    Возвращает (image_bgr, ground_truth_dict).
    """
    rng = np.random.default_rng(seed)
    a = np.deg2rad(alpha_deg)
    theta = np.deg2rad(theta_deg)

    if alpha_deg <= 1e-6:
        x, z, prof = _meridian_axisym(fluid, volume, theta)
        xs = np.concatenate([-x[::-1], x])
        zs = np.concatenate([z[::-1], z])
        gt = dict(theta_left=theta_deg, theta_right=theta_deg,
                  base_width_mm=2*prof.base_radius*1e3,
                  apex_height_mm=prof.apex_height*1e3,
                  alpha_deg=0.0, volume_ul=volume*1e9,
                  sigma=fluid.sigma, capillary_length_mm=fluid.capillary_length*1e3)
    else:
        w = 2.0 * (3*volume/(2*np.pi))**(1/3)
        if inclined_hysteresis:
            thA, thR, slid = hysteresis_symmetric(a, volume, theta, fluid, w)
            if slid or thA is None:
                thA, thR = theta + np.deg2rad(20), theta - np.deg2rad(20)
        else:
            thA, thR = theta, theta
        zeta = w/2 * 1.1
        xs, zs, info = two_arc_profile_2d(zeta, thA, thR, n=240)
        gt = dict(theta_left=np.rad2deg(thR), theta_right=np.rad2deg(thA),
                  base_width_mm=(info['L1']+info['L2'])*1e3,
                  apex_height_mm=info['H']*1e3,
                  alpha_deg=alpha_deg, volume_ul=volume*1e9,
                  sigma=fluid.sigma, capillary_length_mm=fluid.capillary_length*1e3)

    # поворот профиля на угол наклона (downhill — вправо)
    ca, sa = np.cos(-a), np.sin(-a)
    xr = xs*ca - zs*sa
    zr = xs*sa + zs*ca

    cx = img_w * 0.5
    cy = img_h * 0.72
    px = cx + xr * 1e3 * px_per_mm
    py = cy - zr * 1e3 * px_per_mm

    img = np.full((img_h, img_w), 235, np.uint8)   # светлый фон

    # подложка: полуплоскость под линией поверхности; направление совпадает
    # с базой капли в экранных координатах: (cos a, sin a) (downhill вправо-вниз)
    p1 = (int(cx - img_w*np.cos(a)), int(cy - img_w*np.sin(a)))
    p2 = (int(cx + img_w*np.cos(a)), int(cy + img_w*np.sin(a)))
    sub_poly = np.array([[p1[0], p1[1]], [p2[0], p2[1]],
                         [p2[0], img_h+200], [p1[0], img_h+200]], np.int32)
    cv2.fillPoly(img, [sub_poly], 205)
    cv2.line(img, p1, p2, 120, 2)

    # капля — тёмная заливка
    poly = np.stack([px, py], axis=1).astype(np.int32)
    cv2.fillPoly(img, [poly], 70)
    cv2.polylines(img, [poly], True, 45, 2)

    if blur > 0:
        k = int(blur*2)*2+1
        img = cv2.GaussianBlur(img, (k, k), blur)
    if noise > 0:
        img = np.clip(img.astype(np.float64) + rng.normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)

    gt.update(px_per_mm=px_per_mm, contact_center_px=(cx, cy))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr, gt


if __name__ == "__main__":
    os.makedirs("/home/claude/droplet/cv/out", exist_ok=True)
    for name, kw in [("drop_horizontal", dict(alpha_deg=0.0, volume=40e-9, theta_deg=115.0)),
                     ("drop_inclined", dict(alpha_deg=35.0, volume=30e-9, theta_deg=100.0))]:
        img, gt = render_drop(**kw)
        path = f"/home/claude/droplet/cv/out/{name}.png"
        cv2.imwrite(path, img)
        print(f"{name}: {img.shape}, GT theta L/R = "
              f"{gt['theta_left']:.1f}/{gt['theta_right']:.1f}, "
              f"base={gt['base_width_mm']:.2f} мм, h={gt['apex_height_mm']:.2f} мм")

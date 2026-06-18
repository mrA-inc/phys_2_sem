r"""
run_cv.py
=========
Полный CV-конвейер для статьи:
  - генерация двух «экспериментальных» снимков (горизонталь + наклон);
  - измерение геометрии и краевых углов, ADSA-восстановление sigma;
  - аннотированные рисунки в cv/out/;
  - таблица сравнения модель (GT) <-> CV.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
from synthetic import render_drop
from dropcv import measure, adsa_fit_surface_tension, visualize_measurement
from physics import WATER

OUT = "/home/claude/droplet/cv/out"
os.makedirs(OUT, exist_ok=True)


def process(name, kw, title, do_adsa=False):
    img, gt = render_drop(**kw)
    cv2.imwrite(f"{OUT}/{name}.png", img)
    m = measure(img, px_per_mm=gt["px_per_mm"])
    if do_adsa:
        adsa_fit_surface_tension(m, WATER, gt["px_per_mm"],
                                 theta_guess_deg=0.5*(m.theta_left_deg+m.theta_right_deg))
    visualize_measurement(img, m, save_path=f"{OUT}/{name}_annotated.png",
                          title=title, gt=gt)
    return gt, m


def main():
    print("CV-обработка ->", OUT)
    rows = []

    gt1, m1 = process("drop_horizontal",
                      dict(alpha_deg=0.0, volume=45e-9, theta_deg=112.0, px_per_mm=60.0),
                      "Горизонтальная капля (вода, 45 мкл)", do_adsa=True)
    rows.append(("Горизонталь", gt1, m1))

    gt2, m2 = process("drop_inclined",
                      dict(alpha_deg=35.0, volume=30e-9, theta_deg=100.0, px_per_mm=60.0),
                      "Наклонная капля (вода, 30 мкл, α=35°)", do_adsa=False)
    rows.append(("Наклон 35°", gt2, m2))

    # таблица
    print("\n" + "="*78)
    print(f"{'параметр':<22}{'GT (модель)':>18}{'CV (измерение)':>20}{'Δ':>14}")
    print("="*78)

    def line(label, gtv, cvv, unit="", fmt="{:.2f}"):
        d = cvv - gtv
        print(f"{label:<22}{fmt.format(gtv)+' '+unit:>18}{fmt.format(cvv)+' '+unit:>20}{('%+.2f'%d):>14}")

    for tag, gt, m in rows:
        print(f"\n--- {tag} ---")
        line("краевой угол лев.", gt["theta_left"], m.theta_left_deg, "°", "{:.1f}")
        line("краевой угол прав.", gt["theta_right"], m.theta_right_deg, "°", "{:.1f}")
        line("угол наклона α", gt["alpha_deg"], m.alpha_deg, "°", "{:.1f}")
        line("основание", gt["base_width_mm"], m.base_width_mm, "мм")
        line("высота", gt["apex_height_mm"], m.apex_height_mm, "мм")
        if np.isfinite(m.sigma_adsa):
            line("σ (ADSA)", gt["sigma"]*1e3, m.sigma_adsa*1e3, "мН/м", "{:.1f}")
    print("="*78)
    print("\nАннотированные снимки: drop_horizontal_annotated.png, drop_inclined_annotated.png")


if __name__ == "__main__":
    main()

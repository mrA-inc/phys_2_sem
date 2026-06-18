r"""
dropcv.py
=========
Компьютерное зрение для снимка капли (вид сбоку):
  1. извлечение контура капли (порог Оцу + морфология + контуры);
  2. детекция базовой линии (подложки) и угла наклона alpha;
  3. определение точек контакта и измерение левого/правого краевых углов
     (локальная полиномиальная аппроксимация в системе базовой линии);
  4. измерение геометрии: ширина основания, высота, радиус кривизны у вершины;
  5. ADSA-восстановление поверхностного натяжения из формы (осесимметрия);
  6. аннотированная визуализация + экспорт измерений.

Зависит от sim/ (модель Юнга–Лапласа для ADSA-подгонки).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.optimize import least_squares

from physics import WATER, Fluid
from drop_axisym import axisym_profile


# ===========================================================================
# 1. Контур капли
# ===========================================================================
def _multiotsu2(gray):
    """Два порога (3 класса) методом Оцу перебором. Возвращает (t_low, t_high)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    muT = mu[-1]
    best, best_var = (85, 170), -1.0
    idx = np.arange(256)
    for t1 in range(1, 254):
        if w[t1] < 1e-6:
            continue
        for t2 in range(t1 + 1, 255):
            w0 = w[t1]; w1 = w[t2] - w[t1]; w2 = 1.0 - w[t2]
            if w1 < 1e-6 or w2 < 1e-6:
                continue
            m0 = mu[t1] / w0
            m1 = (mu[t2] - mu[t1]) / w1
            m2 = (muT - mu[t2]) / w2
            var = w0*(m0-muT)**2 + w1*(m1-muT)**2 + w2*(m2-muT)**2
            if var > best_var:
                best_var, best = var, (t1, t2)
    return best


def extract_drop_contour(img_bgr):
    """Возвращает (contour Nx2 в пикселях, бинарная маска капли).
    Капля — самый тёмный класс (3-классовый порог Оцу), устойчиво к подложке."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    t_low, t_high = _multiotsu2(gray)
    # капля = пиксели темнее нижнего порога
    th = np.where(gray <= t_low, 255, 0).astype(np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("Контур капли не найден")
    cnt = max(cnts, key=cv2.contourArea)
    return cnt.reshape(-1, 2).astype(float), th


# ===========================================================================
# 2. Базовая линия (подложка) и угол наклона
# ===========================================================================
def detect_baseline(img_bgr):
    """
    Детекция линии подложки методом Хафа. Возвращает (point, direction, alpha_deg).
    direction — единичный вектор вдоль поверхности (downhill в сторону +x экрана).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=img_bgr.shape[1]//3, maxLineGap=30)
    if lines is None:
        return None
    # выбираем самую длинную почти-горизонтальную линию
    best, best_len = None, 0
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if abs(ang) < 50:  # подложка не круче 50°
            length = np.hypot(x2 - x1, y2 - y1)
            if length > best_len:
                best_len, best = length, l
    if best is None:
        return None
    x1, y1, x2, y2 = best
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    d = np.array([x2 - x1, y2 - y1], float); d /= np.linalg.norm(d)
    alpha = np.degrees(np.arctan2(d[1], d[0]))   # экранные y вниз -> downhill вправо
    return np.array([x1, y1], float), d, alpha


# ===========================================================================
# 3+4. Краевые углы и геометрия
# ===========================================================================
@dataclass
class CVMeasurement:
    alpha_deg: float
    theta_left_deg: float
    theta_right_deg: float
    base_width_px: float
    apex_height_px: float
    base_width_mm: float = np.nan
    apex_height_mm: float = np.nan
    apex_radius_mm: float = np.nan
    contact_left_px: tuple = (0, 0)
    contact_right_px: tuple = (0, 0)
    apex_px: tuple = (0, 0)
    sigma_adsa: float = np.nan
    extras: dict = field(default_factory=dict)


def _project(points, origin, d, n):
    """Координаты точек в системе базовой линии: s вдоль d, t по нормали n."""
    rel = points - origin
    s = rel @ d
    t = rel @ n
    return s, t


def measure(img_bgr, px_per_mm=None, fluid: Fluid = WATER, fit_window_px=45):
    """Полное измерение капли по снимку."""
    cnt, mask = extract_drop_contour(img_bgr)
    base = detect_baseline(img_bgr)
    if base is None:
        raise RuntimeError("Базовая линия не найдена")
    origin, d, alpha = base
    n = np.array([d[1], -d[0]])           # нормаль, указывает «вверх» от подложки
    # убедимся, что нормаль смотрит в сторону капли (где больше точек контура)
    s_all, t_all = _project(cnt, origin, d, n)
    if np.median(t_all) < 0:
        n = -n
        s_all, t_all = _project(cnt, origin, d, n)

    # точки контура НАД подложкой (t>0) — собственно капля
    drop_mask = t_all > -2.0
    s_d, t_d = s_all[drop_mask], t_all[drop_mask]
    cnt_d = cnt[drop_mask]

    # точки контакта: крайние слева/справа точки капли у подложки (малое t)
    near = t_d < (0.06 * (t_d.max() - t_d.min()) + t_d.min())
    s_near = s_d[near]
    s_left, s_right = s_near.min(), s_near.max()
    # вершина (макс. высота над подложкой)
    i_apex = np.argmax(t_d)
    apex_pt = cnt_d[i_apex]
    apex_h = t_d[i_apex]

    base_width = s_right - s_left

    def contact_point_px(s_target):
        idx = np.argmin(np.abs(s_d - s_target) + 5.0*np.abs(t_d))
        return cnt_d[idx], idx
    cl, il = contact_point_px(s_left)
    cr, ir = contact_point_px(s_right)

    # --- краевой угол: локальный фит окружности у линии контакта ---
    # касательная к окружности в точке контакта даёт угол, корректный и для theta>90.
    t_span = t_d.max() - min(t_d.min(), 0.0)
    R_local = max(0.30 * t_span, 20.0)        # радиус окрестности фита, px
    T_band = max(0.45 * t_span, 30.0)

    def _fit_circle(xs, ys):
        # алгебраический фит (Kåsa): x^2+y^2 = 2 a x + 2 b y + c
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        bb = xs**2 + ys**2
        sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
        xc = sol[0] / 2.0; yc = sol[1] / 2.0
        R = np.sqrt(sol[2] + xc**2 + yc**2)
        return xc, yc, R

    # --- краевой угол: локальный фит окружности к стенке у каждого контакта ---
    # (работает и для симметричной, и для асимметричной/наклонной капли).
    apex_h_val = t_d.max()
    t_band = max(0.42 * apex_h_val, 28.0)     # высота полосы стенки, px
    s_window = 0.45 * max(base_width, 1.0)    # ширина по s (не дотягиваться до др. контакта)

    def _fit_circle_st(ss, tt):
        A = np.column_stack([ss, tt, np.ones_like(ss)])
        bb = ss**2 + tt**2
        sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
        sc = sol[0] / 2.0; tc = sol[1] / 2.0
        R = np.sqrt(max(sol[2] + sc**2 + tc**2, 1e-9))
        return sc, tc, R

    def contact_angle(s_contact, side):
        sel = (t_d > 2.5) & (t_d < t_band) & (np.abs(s_d - s_contact) < s_window)
        if sel.sum() < 6:
            return np.nan, None, (np.nan, np.nan)
        ss, tt = s_d[sel], t_d[sel]
        sc, tc, R = _fit_circle_st(ss, tt)
        # точка контакта на окружности у t=0: ближайшая к s_contact
        # касательная к окружности в (s_contact,0): перпендикуляр к радиусу
        rad = np.array([s_contact - sc, 0.0 - tc])
        nr = np.hypot(*rad)
        if nr < 1e-9:
            return np.nan, None, (sc, tc)
        tau = np.array([-rad[1], rad[0]])
        if tau[1] < 0:
            tau = -tau                       # направить вверх (в каплю)
        ds, dt = tau
        if side == "left":
            theta = np.degrees(np.arctan2(dt, ds))
        else:
            theta = 180.0 - np.degrees(np.arctan2(dt, ds))
        theta = (theta + 360.0) % 360.0
        if theta > 180.0:
            theta = 360.0 - theta
        v = tau / np.hypot(*tau)
        return theta, v, (sc, tc)

    theta_left, vL, _ = contact_angle(s_left, "left")
    theta_right, vR, _ = contact_angle(s_right, "right")
    ellipse_params = None

    # радиус кривизны у вершины: круг по точкам верхушки (t близко к максимуму)
    apex_R_px = np.nan
    top_sel = t_d > (t_d.max() - max(0.25 * t_span, 18.0))
    if top_sel.sum() >= 5:
        xs_t = cnt_d[top_sel, 0]; ys_t = cnt_d[top_sel, 1]
        A = np.column_stack([xs_t, ys_t, np.ones_like(xs_t)])
        bb = xs_t**2 + ys_t**2
        try:
            sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
            xc, yc, cc = sol
            apex_R_px = np.sqrt(cc + xc**2/4*0 + (xc/2)**2 + (yc/2)**2)
            apex_R_px = np.sqrt((xc/2)**2 + (yc/2)**2 + cc)
        except Exception:
            pass

    # точки контакта в пикселях из пересечения эллипса с базовой линией (t=0)
    cl = origin + s_left * d
    cr = origin + s_right * d

    m = CVMeasurement(
        alpha_deg=alpha, theta_left_deg=theta_left, theta_right_deg=theta_right,
        base_width_px=base_width, apex_height_px=apex_h,
        contact_left_px=tuple(cl), contact_right_px=tuple(cr), apex_px=tuple(apex_pt),
    )
    if px_per_mm:
        m.base_width_mm = base_width / px_per_mm
        m.apex_height_mm = apex_h / px_per_mm
        if np.isfinite(apex_R_px):
            m.apex_radius_mm = apex_R_px / px_per_mm
    # касательные в экранных координатах для визуализации
    vL_img = (vL[0]*d + vL[1]*n) if vL is not None else None
    vR_img = (vR[0]*d + vR[1]*n) if vR is not None else None
    m.extras.update(origin=origin, d=d, n=n, contour=cnt_d,
                    s=s_d, t=t_d, px_per_mm=px_per_mm,
                    tangent_left=vL_img, tangent_right=vR_img,
                    ellipse=ellipse_params, s_left=s_left, s_right=s_right)
    return m


# ===========================================================================
# 5. ADSA: восстановление поверхностного натяжения из формы (осесимметрия)
# ===========================================================================
def adsa_fit_surface_tension(m: CVMeasurement, fluid: Fluid, px_per_mm: float,
                             theta_guess_deg: float = None):
    """
    Подгонка осесимметричного профиля Юнга–Лапласа к контуру (для alpha≈0).
    Свободные параметры: кривизна вершины b (1/м) и капиллярная константа c (1/м^2).
    sigma = delta_rho g / c. Положение/масштаб фиксированы по вершине и оси.
    """
    s = m.extras["s"]; t = m.extras["t"]
    # координаты в мм: r — горизонталь от оси, y — высота
    s_mm = s / px_per_mm
    y_mm = t / px_per_mm
    r0_mm = 0.5 * (np.min(s_mm) + np.max(s_mm))   # ось симметрии
    r_mm = s_mm - r0_mm
    y_top = y_mm.max()

    # эксперим. правый профиль y(|r|): усредняем левую/правую ветви
    R = np.abs(r_mm)
    order = np.argsort(R)
    Rs, Ys = R[order], y_mm[order]

    theta0 = np.deg2rad(theta_guess_deg if theta_guess_deg else
                        0.5*(m.theta_left_deg + m.theta_right_deg))

    def model_profile(b, c):
        prof = axisym_profile(b, c, theta0)
        xr = prof.x * 1e3            # мм
        yr = (prof.apex_height - prof.z) * 1e3
        return xr, yr

    def resid(params):
        b, c = params
        b = abs(b); c = abs(c)
        try:
            xr, yr = model_profile(b, c)
        except Exception:
            return np.full(40, 1e3)
        # сопоставляем по высоте: для набора высот сравниваем радиусы
        yq = np.linspace(0.05*y_top, 0.95*y_top, 40)
        r_model = np.interp(yq, yr, xr, left=np.nan, right=np.nan)
        # эксперимент: радиус как функция высоты (берём огибающую макс |r| при данной y)
        r_exp = np.array([np.max(R[np.abs(y_mm - yy) < 0.08]) if
                          np.any(np.abs(y_mm - yy) < 0.08) else np.nan for yy in yq])
        good = np.isfinite(r_model) & np.isfinite(r_exp)
        if good.sum() < 8:
            return np.full(40, 1e3)
        res = np.where(good, r_model - r_exp, 0.0)
        return res

    b0 = 1.0 / (max(m.apex_height_mm, 0.5) * 1e-3)   # грубо 1/высота
    c0 = fluid.c
    sol = least_squares(resid, [b0, c0], method="lm", max_nfev=200)
    b_fit, c_fit = abs(sol.x[0]), abs(sol.x[1])
    sigma = fluid.delta_rho * fluid.g / c_fit
    m.sigma_adsa = sigma
    m.extras["adsa"] = dict(b=b_fit, c=c_fit, theta0=np.rad2deg(theta0),
                            cost=sol.cost, a_mm=np.sqrt(1.0/c_fit)*1e3)
    return sigma


if __name__ == "__main__":
    from synthetic import render_drop
    img, gt = render_drop(alpha_deg=0.0, volume=45e-9, theta_deg=112.0, px_per_mm=60.0)
    m = measure(img, px_per_mm=gt["px_per_mm"])
    print("=== Горизонтальная капля ===")
    print(f"  GT:  theta={gt['theta_left']:.1f}°, base={gt['base_width_mm']:.2f} мм, h={gt['apex_height_mm']:.2f} мм")
    print(f"  CV:  theta L/R={m.theta_left_deg:.1f}/{m.theta_right_deg:.1f}°, "
          f"base={m.base_width_mm:.2f} мм, h={m.apex_height_mm:.2f} мм, alpha={m.alpha_deg:.1f}°")
    sigma = adsa_fit_surface_tension(m, WATER, gt["px_per_mm"], theta_guess_deg=112.0)
    print(f"  ADSA sigma={sigma*1e3:.1f} мН/м (истинное {WATER.sigma*1e3:.1f}), "
          f"a={m.extras['adsa']['a_mm']:.2f} мм")


# ===========================================================================
# 6. Визуализация результатов измерения
# ===========================================================================
def visualize_measurement(img_bgr, m: CVMeasurement, save_path=None,
                          title=None, gt=None):
    """Аннотированный рисунок: контур, базовая линия, точки контакта,
    касательные краевых углов, вершина, измеренные величины."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ex = m.extras
    origin = ex["origin"]; d = ex["d"]; n = ex["n"]
    cnt = ex["contour"]
    cl = np.array(m.contact_left_px); cr = np.array(m.contact_right_px)
    apex = np.array(m.apex_px)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)

    # контур капли
    ax.plot(cnt[:, 0], cnt[:, 1], color="#21d0d0", lw=1.6, label="контур капли")

    # базовая линия (вдоль d через точки контакта)
    base_pts = np.array([cl - d*60, cr + d*60])
    ax.plot(base_pts[:, 0], base_pts[:, 1], color="#ffb000", lw=1.8, label="базовая линия")

    # точки контакта и вершина
    ax.plot(*cl, "o", color="#ff4d4d", ms=7)
    ax.plot(*cr, "o", color="#ff4d4d", ms=7)
    ax.plot(*apex, "s", color="#7CFC00", ms=7, label="вершина")

    # касательные краевых углов
    Llen = 0.9 * max(m.base_width_px, 60) / 2
    for c, v, side in ((cl, ex.get("tangent_left"), "L"),
                       (cr, ex.get("tangent_right"), "R")):
        if v is None:
            continue
        p2 = c + v * Llen
        ax.plot([c[0], p2[0]], [c[1], p2[1]], color="#ff4d4d", lw=2.0)

    # высота: вертикаль от вершины к базе (по нормали)
    # проекция вершины на базовую линию
    s_apex = (apex - origin) @ d
    foot = origin + s_apex * d
    ax.plot([apex[0], foot[0]], [apex[1], foot[1]], ls=":", color="#cccccc", lw=1.4)

    # подписи углов
    txt_l = f"θ_L={m.theta_left_deg:.1f}°"
    txt_r = f"θ_R={m.theta_right_deg:.1f}°"
    ax.annotate(txt_l, cl, textcoords="offset points", xytext=(-70, 8),
                color="#ff4d4d", fontsize=11, fontweight="bold")
    ax.annotate(txt_r, cr, textcoords="offset points", xytext=(18, 8),
                color="#ff4d4d", fontsize=11, fontweight="bold")

    # сводный блок
    lines = [f"α = {m.alpha_deg:.1f}°",
             f"основание = {m.base_width_mm:.2f} мм",
             f"высота = {m.apex_height_mm:.2f} мм"]
    if np.isfinite(m.apex_radius_mm):
        lines.append(f"R(вершина) = {m.apex_radius_mm:.2f} мм")
    if np.isfinite(m.sigma_adsa):
        lines.append(f"σ(ADSA) = {m.sigma_adsa*1e3:.1f} мН/м")
    if gt is not None:
        lines.append("—")
        lines.append(f"GT θ = {gt['theta_left']:.1f}/{gt['theta_right']:.1f}°")
    ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round", fc="black", ec="#888888", alpha=0.55),
            color="white")

    ax.set_title(title or "CV-измерение капли", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
        print(f"  сохранено {os.path.basename(save_path)}")
    plt.close(fig)

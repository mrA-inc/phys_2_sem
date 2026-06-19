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



# ===========================================================================
# 7. Калибровка масштаба по линейке (шаг 1/64 дюйма)
#    Линейка наклонена ~-28° в кадре; штрихи под ~+61°.
#    Ищем период вдоль оси линейки через автокорреляцию проекций штрихов.
# ===========================================================================
RULER_STEP_MM = 25.4 / 64   # 1/64 дюйма (~0.397 мм)


def calibrate_ruler(img_bgr, step_mm: float = RULER_STEP_MM,
                    ruler_angle_deg: float = -28.0,
                    debug_path: str | None = None):
    """
    Вычисляет px_per_mm по периоду штрихов линейки.
    ruler_angle_deg — известный угол наклона оси линейки в кадре.
    Возвращает (px_per_mm, period_px).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # единичный вектор вдоль оси линейки
    a = np.deg2rad(ruler_angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-d[1], d[0]])   # перпендикуляр (вдоль штрихов)

    # ROI — нижние 60% кадра (там линейка)
    roi_y = int(h * 0.4)
    gray_roi = gray[roi_y:, :]
    edges = cv2.Canny(gray_roi, 20, 70)

    # проецируем точки края на ось линейки
    ys_idx, xs_idx = np.where(edges > 0)
    if len(xs_idx) < 20:
        raise RuntimeError("Слишком мало краёв в ROI")
    pts = np.stack([xs_idx, ys_idx + roi_y], axis=1).astype(float)
    origin = np.array([w / 2, h / 2], float)
    s = (pts - origin) @ d   # координата вдоль линейки
    t = (pts - origin) @ n   # поперёк (вдоль штрихов)

    # отбираем точки в полосе ±band вдоль оси линейки (сама линейка ~300 px шириной)
    band = 400
    sel = np.abs(t) < band
    if sel.sum() < 20:
        raise RuntimeError("Точки внутри полосы линейки не найдены")
    s_pts = s[sel]

    # автокорреляция гистограммы вдоль s
    s_min, s_max = s_pts.min(), s_pts.max()
    n_bins = min(int(s_max - s_min) + 1, 4000)
    hist, _ = np.histogram(s_pts, bins=n_bins, range=(s_min, s_max))
    hist = hist.astype(float)
    hist -= hist.mean()
    acorr = np.correlate(hist, hist, mode="full")[len(hist) - 1:]
    acorr[0] = 0

    # первый значимый пик (период штриха: ожидаем 10–200 px)
    peaks = [(acorr[i], i) for i in range(8, min(300, len(acorr) - 1))
             if acorr[i] > acorr[i-1] and acorr[i] > acorr[i+1]]
    if not peaks:
        raise RuntimeError("Период штрихов не найден")
    period_px = float(max(peaks, key=lambda x: x[0])[1])
    px_per_mm = period_px / step_mm

    if debug_path:
        dbg = img_bgr.copy()
        # рисуем ось линейки через центр
        p1 = (origin + d * (-w * 0.6)).astype(int)
        p2 = (origin + d * (w * 0.6)).astype(int)
        cv2.line(dbg, tuple(p1), tuple(p2), (0, 255, 0), 3)
        small = cv2.resize(dbg, (dbg.shape[1] // 4, dbg.shape[0] // 4))
        cv2.imwrite(debug_path, small)

    return px_per_mm, period_px


# ===========================================================================
# 8. Извлечение разметки из цветного фото (синяя линия + красный контур)
# ===========================================================================

def extract_colored_markup(img_bgr):
    """
    Извлекает из фото с ручной разметкой:
      - синюю линию  -> базовая линия (поверхность зеркала)
      - красную кривую -> профиль капли
    Возвращает (baseline_pts, drop_pts) — массивы пикселей Nx2.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # синий: H=100-130
    blue_mask = cv2.inRange(hsv,
                            np.array([95, 80, 80]),
                            np.array([135, 255, 255]))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE,
                                 np.ones((7, 7), np.uint8))

    # красный: H=0-10 или 165-180
    red_mask1 = cv2.inRange(hsv, np.array([0,  80, 80]), np.array([10,  255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE,
                                np.ones((7, 7), np.uint8))

    def mask_to_pts(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return np.stack([xs, ys], axis=1).astype(float)

    blue_pts = mask_to_pts(blue_mask)
    red_pts  = mask_to_pts(red_mask)
    return blue_pts, red_pts


def fit_baseline_from_pts(pts):
    """Фитирует прямую по облаку точек (SVD). Возвращает (origin, direction, alpha_deg)."""
    center = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - center)
    d = Vt[0]
    if d[0] < 0:
        d = -d
    alpha_deg = np.degrees(np.arctan2(d[1], d[0]))
    return center, d, alpha_deg


def measure_from_markup(img_bgr, px_per_mm: float):
    """
    Измерение краевых углов из ручной цветной разметки.
    Синяя линия = поверхность зеркала.
    Красная кривая = боковой профиль капли от одного контакта до другого:
      - конец с меньшим t (ближе к поверхности) с меньшим s -> левый контакт
      - конец с меньшим t с бОльшим s                       -> правый контакт
      - точка с максимальным t                               -> вершина
    Краевой угол вычисляется как касательная к кривой в конечных точках.
    """
    blue_pts, red_pts = extract_colored_markup(img_bgr)
    if blue_pts is None or len(blue_pts) < 10:
        raise RuntimeError("Синяя базовая линия не найдена")
    if red_pts is None or len(red_pts) < 10:
        raise RuntimeError("Красный контур капли не найден")

    # базовая линия
    origin, d, alpha_deg = fit_baseline_from_pts(blue_pts)
    n = np.array([-d[1], d[0]])
    # n смотрит в сторону капли: центр красных точек должен давать t > 0
    if (red_pts.mean(axis=0) - origin) @ n < 0:
        n = -n

    # координаты красных точек в системе базовой линии
    rel = red_pts - origin
    s_all = rel @ d
    t_all = rel @ n

    # убираем точки ниже поверхности (шум разметки)
    above = t_all > 2.0
    s_all, t_all = s_all[above], t_all[above]
    red_pts_above = red_pts[above]
    if len(s_all) < 10:
        raise RuntimeError("Недостаточно точек профиля над поверхностью")

    # вершина — точка максимального удаления от поверхности
    i_apex = np.argmax(t_all)
    apex_height = t_all[i_apex]
    apex_px = red_pts_above[i_apex]

    # концы кривой — точки с малым t (у поверхности)
    # сортируем по s: левый контакт (меньший s), правый (больший s)
    t_thresh = min(0.15 * apex_height, 30.0)
    near_surf = t_all < t_thresh
    if near_surf.sum() < 4:
        # запасной вариант: берём крайние точки по s
        near_surf = t_all < (0.25 * apex_height)
    s_near = s_all[near_surf]
    s_left  = s_near.min()
    s_right = s_near.max()
    base_width = s_right - s_left

    cl_px = origin + s_left  * d
    cr_px = origin + s_right * d

    # касательная в точке контакта: локальный полиномиальный фит
    # используем точки в окрестности t < t_band и s близко к контакту
    def _tangent_angle(s_contact, side):
        w_s = max(0.35 * base_width, 20.0)
        t_band = max(0.35 * apex_height, 15.0)
        sel = (t_all > 1.0) & (t_all < t_band) & (np.abs(s_all - s_contact) < w_s)
        if sel.sum() < 5:
            # расширяем окно
            sel = (t_all < t_band) & (np.abs(s_all - s_contact) < w_s * 2)
        if sel.sum() < 4:
            return np.nan
        ss, tt = s_all[sel], t_all[sel]
        # параметрический порядок вдоль кривой: сортируем по t
        order = np.argsort(tt)
        ss, tt = ss[order], tt[order]
        # фитируем s(t): квадратика, ds/dt при t=0 = coeffs[1]
        if len(tt) >= 3:
            coeffs = np.polyfit(tt, ss, 2)
            ds_dt = coeffs[1]
        else:
            ds_dt = (ss[-1] - ss[0]) / max(tt[-1] - tt[0], 1e-6)
        # краевой угол = угол между касательной к профилю и поверхностью
        # касательный вектор (в системе s,t): (ds_dt, 1), нормированный
        # угол с осью s (поверхностью): arctan(1 / |ds_dt|)
        # для левого контакта профиль идёт вправо-вверх (ds_dt > 0 -> угол острый)
        # для правого — влево-вверх (ds_dt < 0 -> угол острый с другой стороны)
        theta = np.degrees(np.arctan2(1.0, abs(ds_dt)))
        return float(theta)

    theta_left  = _tangent_angle(s_left,  "left")
    theta_right = _tangent_angle(s_right, "right")

    m = CVMeasurement(
        alpha_deg=alpha_deg,
        theta_left_deg=theta_left,
        theta_right_deg=theta_right,
        base_width_px=base_width,
        apex_height_px=apex_height,
        contact_left_px=tuple(cl_px),
        contact_right_px=tuple(cr_px),
        apex_px=tuple(apex_px),
    )
    if px_per_mm:
        m.base_width_mm  = base_width  / px_per_mm
        m.apex_height_mm = apex_height / px_per_mm

    m.extras.update(
        origin=origin, d=d, n=n,
        contour=np.stack([s_all, t_all], axis=1),
        s=s_all, t=t_all, px_per_mm=px_per_mm,
        tangent_left=None, tangent_right=None,
        ellipse=None, s_left=s_left, s_right=s_right,
        contour_px=red_pts_above,
    )
    return m


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.join(_dir, "out")
    os.makedirs(_out, exist_ok=True)

    img_path = os.path.join(_dir, "exp_data.jpg")
    if not os.path.exists(img_path):
        print("exp_data.jpg не найден, запускаем на синтетике")
        sys.path.insert(0, _dir)
        from synthetic import render_drop
        img, gt = render_drop(alpha_deg=0.0, volume=45e-9, theta_deg=112.0, px_per_mm=60.0)
        px_per_mm = gt["px_per_mm"]
        m = measure(img, px_per_mm=px_per_mm)
        print(f"GT theta={gt['theta_left']:.1f}, CV L/R={m.theta_left_deg:.1f}/{m.theta_right_deg:.1f}")
        sys.exit(0)

    # загрузка (cv2 не любит кириллицу в пути)
    buf = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Не удалось загрузить exp_data.jpg")
    print(f"Загружен снимок: {img.shape[1]}x{img.shape[0]} px")

    # --- калибровка по линейке ---
    # линейка наклонена на -28 deg (определено ранее из Хафа)
    try:
        px_per_mm, period_px = calibrate_ruler(
            img,
            step_mm=RULER_STEP_MM,
            ruler_angle_deg=-28.0,
            debug_path=os.path.join(_out, "ruler_debug.png"),
        )
        print(f"Калибровка: период={period_px:.1f} px, px/mm={px_per_mm:.2f}")
    except RuntimeError as e:
        print(f"  WARN: Калибровка не удалась: {e}, используем px/mm=60")
        px_per_mm = 60.0

    # --- измерение по цветной разметке ---
    try:
        m = measure_from_markup(img, px_per_mm=px_per_mm)
    except RuntimeError as e:
        print(f"  Разметка не найдена ({e}), пробуем автодетекцию")
        m = measure(img, px_per_mm=px_per_mm)

    # физический угол наклона зеркала: камера вертикальна над зеркалом,
    # синяя линия на экране даёт alpha_screen, реальный наклон = 90 - alpha_screen
    alpha_phys = 90.0 - abs(m.alpha_deg)

    print(f"\nРезультат:")
    print(f"  alpha_screen = {m.alpha_deg:.1f} grad (угол синей линии на экране)")
    print(f"  alpha_phys   = {alpha_phys:.1f} grad (физический наклон зеркала)")
    print(f"  theta_L      = {m.theta_left_deg:.1f} grad")
    print(f"  theta_R      = {m.theta_right_deg:.1f} grad")
    print(f"  основание    = {m.base_width_mm:.2f} мм")
    print(f"  высота       = {m.apex_height_mm:.2f} мм")

    # --- сохраняем числа в машиночитаемый JSON для simulate.py ---
    import json
    result_json = {
        "source":          "exp_data.jpg",
        "px_per_mm":       round(float(px_per_mm), 3),
        "alpha_screen_deg": round(float(m.alpha_deg), 2),
        "alpha_phys_deg":  round(float(alpha_phys), 2),
        "theta_L_deg":     round(float(m.theta_left_deg), 2),
        "theta_R_deg":     round(float(m.theta_right_deg), 2),
        "base_width_mm":   round(float(m.base_width_mm), 3),
        "apex_height_mm":  round(float(m.apex_height_mm), 3),
    }
    json_path = os.path.join(_out, "cv_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"\nЧисла сохранены: {json_path}")

    # визуализация поверх оригинала
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # синяя базовая линия
    blue_pts, red_pts = extract_colored_markup(img)
    if blue_pts is not None:
        ax.scatter(blue_pts[:, 0], blue_pts[:, 1], s=0.5, c="cyan", alpha=0.5,
                   label="базовая линия")
    if red_pts is not None:
        ax.scatter(red_pts[:, 0], red_pts[:, 1], s=0.5, c="red", alpha=0.5,
                   label="профиль капли")

    cl = np.array(m.contact_left_px)
    cr = np.array(m.contact_right_px)
    ax.plot(*cl, "o", color="yellow", ms=10, label="контакт")
    ax.plot(*cr, "o", color="yellow", ms=10)
    ax.plot(*np.array(m.apex_px), "s", color="lime", ms=10, label="вершина")

    info = (f"alpha={m.alpha_deg:.1f} grad\n"
            f"theta_L={m.theta_left_deg:.1f} grad\n"
            f"theta_R={m.theta_right_deg:.1f} grad\n"
            f"base={m.base_width_mm:.2f} mm\n"
            f"h={m.apex_height_mm:.2f} mm\n"
            f"px/mm={px_per_mm:.1f}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top",
            fontsize=11, family="monospace",
            bbox=dict(fc="black", ec="gray", alpha=0.6), color="white")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("exp_data.jpg — измерение краевых углов")
    ax.set_xticks([]); ax.set_yticks([])

    vis_path = os.path.join(_out, "exp_result.png")
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nСохранено: {vis_path}")


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

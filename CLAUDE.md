# CLAUDE.md — контекст для Claude Code

## Что это за проект

Замкнутое физическое исследование равновесной формы капли жидкости (sessile drop).
Три артефакта: (1) LaTeX-статья, (2) симуляционное ядро + 3D-рендер, (3) CV-конвейер для реального снимка.
Статья готова (v2.0). Текущая работа — CV-обработка реального фото, сравнение с теорией.

---

## Язык и стиль

- **Общение с пользователем**: русский, кратко, ведущая мысль вперёд.
- **Комментарии и docstring в коде**: русский.
- **Статья (LaTeX)**: русский текст, английские термины вводятся один раз в скобках.
- Десятичный разделитель в статье: **запятая** (`2{,}73 мм`), не точка.
- Тире в LaTeX: `---`, не дефис.
- Выделение в тексте: `\emph{}`, не `\textbf{}`.

---

## Окружение

- Python: **3.10** на Windows. Вызывать как `py` (py launcher), не `python3`.
- Пакеты: numpy 2.4, scipy 1.17, matplotlib 3.10, opencv-python 4.13 — уже установлены.
- Для установки новых: `py -m pip install PKG`.
- LaTeX: только **xelatex**. `pdflatex` не работает (t2aenc.def не установлен).
- Шрифты: DejaVu Serif/Sans/Mono — есть кириллица, работают в matplotlib и LaTeX.
- **Кириллица в путях**: `cv2.imread/imwrite` не работают с кириллицей на Windows.
  - Чтение: `buf = np.fromfile(path, dtype=np.uint8); img = cv2.imdecode(buf, cv2.IMREAD_COLOR)`
  - Запись: `cv2.imencode(".png", img)[1].tofile(path)`
- **Эмодзи в print**: cp1251 консоль не поддерживает Unicode-эмодзи (`❌`, `⚠️` и др.) — не использовать в print().

---

## Структура

```
sim/
  physics.py      — Fluid, WATER/GLYCERIN/ETHYLENE_GLYCOL/MERCURY, sphere_cap_*
  drop_2d.py      — 2D горизонталь (точное + дуговое решение)
  drop_axisym.py  — 3D осесимметрия (fit_b_for_volume, drop_volume)
  inclined.py     — наклон: max_volume_3d, hysteresis_symmetric, two_arc_profile_2d,
                    contact_angle_azimuthal, two_circle_volume
  figures.py      — генератор 8 рисунков статьи -> figs/
  render3d.py     — 3D-рендер поверхности капли (API + вызов из simulate)

exp/
  exp_data.jpg    — реальный снимок капли (с синей линией + красным профилем)
  dropcv.py       — CV-конвейер: калибровка по линейке, цветная разметка,
                    краевые углы, геометрия, ADSA, visualize_measurement;
                    пишет out/cv_result.json (числа для simulate.py)
  simulate.py     — параметрическая симуляция + plot_sideview + сравнение с теорией.
                    ЗАПУСК БЕЗ АРГУМЕНТОВ = авто-режим: читает cv_result.json,
                    строит два профиля (CV и теория), пишет figs/comparison.pdf,
                    figs/drop_result.pdf, figs/cv_theory_sideview.pdf.
                    run_simulation(theta_A_deg, theta_R_deg) — профиль из ИЗМЕРЕННЫХ
                    углов напрямую, без hysteresis_symmetric.
                    Режимы: (без арг) авто, --cv-auto авто, --cv ручной ввод,
                    --volume/--theta/--alpha параметрический.
                    ВНИМАНИЕ: sim/simulate.py удалён, актуальная версия здесь
  synthetic.py    — синтетические снимки из физ. модели (для валидации CV)
                    theta_deg=0 -> THETA_DEFAULT_DEG=72.8°; асимметричные капли
  out/            — результаты: exp_result.png, cv_result.json

article/
  figures.py      — копия генератора рисунков (запуск из article/)
  figs/           — копии PDF для \includegraphics
  main.tex        — LaTeX-статья
  main.pdf        — скомпилированный PDF

figs/             — рисунки (PDF+PNG): 8 рисунков статьи (figures.py) +
                    comparison.pdf, drop_result.pdf, cv_theory_sideview.pdf, drop_3d.pdf
run_cv.py         — валидация CV на синтетике (GT vs CV таблица)
                    ВНИМАНИЕ: импортирует synthetic/dropcv напрямую — запускать из exp/
literature/       — ключевые статьи
```

---

## Зависимости между файлами

```
physics.py
  └─► drop_2d.py
  └─► drop_axisym.py
  └─► inclined.py
          └─► sim/figures.py      (8 рисунков статьи)
          └─► sim/render3d.py     (3D-рендер)
          └─► exp/synthetic.py    (синтетические снимки)
          └─► exp/dropcv.py       (CV, через sys.path)
          └─► exp/simulate.py     (сравнение с теорией)
```

**Правило при изменении физики**: правка в `physics/drop_2d/drop_axisym/inclined`
→ перезапустить `sim/figures.py` → `cp figs/*.pdf article/figs/` → перекомпилировать статью.

**Все пути вывода** вычисляются через `os.path.dirname(os.path.abspath(__file__))`.
Никаких захардкоженных `/tmp/` или `/home/claude/`.

---

## Ключевые API

### exp/dropcv.py

```python
extract_colored_markup(img_bgr)
# -> (blue_pts, red_pts): пиксели синей линии и красного профиля

fit_baseline_from_pts(pts)
# -> (origin, direction, alpha_deg): прямая по облаку точек (SVD)

calibrate_ruler(img_bgr, step_mm=RULER_STEP_MM, ruler_angle_deg=-28.0)
# -> (px_per_mm, period_px): калибровка по линейке 1/64"

measure_from_markup(img_bgr, px_per_mm)
# -> CVMeasurement: theta_left_deg, theta_right_deg, base_width_mm,
#                   apex_height_mm, alpha_deg

measure(img_bgr, px_per_mm)
# -> CVMeasurement: автодетекция без цветной разметки (Оцу + Хаф)

adsa_fit_surface_tension(m, fluid, px_per_mm)
# -> sigma [Н/м]: только для горизонтальной капли (alpha < 5°)

visualize_measurement(img_bgr, m, save_path, title, gt)
# -> аннотированный PNG
```

### sim/render3d.py

```python
# CLI (интерактивный): спрашивает ТОЛЬКО V и alpha (theta_Y фиксирован 72.8° вода/стекло)
# py render3d.py  ->  запрашивает V, alpha  ->  figs/drop_3d.pdf

surface_axisym(prof)
# -> (X, Y, Z) мм, для горизонтальной капли

surface_inclined(alpha_deg, volume_ul, theta_Y_deg, fluid, width_mm)
# -> (X, Y, Z) мм, для наклонной капли
```

### sim/drop_axisym.py

```python
prof = fit_b_for_volume(volume_m3, fluid.c, theta_rad)
# prof.x           — полуширина (м)
# prof.z           — расстояние от вершины вниз (м)
# prof.apex_height — полная высота (м)
# высота над подложкой: prof.apex_height - prof.z
```

### sim/inclined.py

```python
max_volume_3d(alpha_rad, thA_rad, thR_rad, fluid, width_m)
hysteresis_symmetric(alpha_rad, V_m3, theta_Y_rad, fluid, w_m)  # -> (θA, θR, slid)
two_arc_profile_2d(zeta_m, theta_down_rad, theta_up_rad, n=200)  # -> (x, y, info)
contact_angle_azimuthal(psi_array, theta_max_rad, theta_min_rad)
```

### exp/synthetic.py

```python
render_drop(volume=40e-9, theta_deg=110.0, alpha_deg=0.0,
            fluid=WATER, px_per_mm=60.0, hysteresis_deg=15.0, seed=0)
# theta_deg=0 -> используется THETA_DEFAULT_DEG = 72.8° (вода на стекле)
# -> (image_bgr, ground_truth_dict)
# ground_truth: theta_left, theta_right, base_width_mm, apex_height_mm, ...
# Горизонталь: асимметрия задаётся hysteresis_deg (theta_L != theta_R)
# Наклон: theta_A/theta_R из hysteresis_symmetric
```

---

## Команды сборки

```bash
# Проверка импортов
cd sim && py -c "import physics,drop_2d,drop_axisym,inclined,render3d; print('OK')"

# CV-обработка реального снимка
cd exp && py dropcv.py
# -> exp/out/exp_result.png (визуализация) + exp/out/cv_result.json (числа)

# Авто-режим: читает cv_result.json, строит CV vs теория (без ручного ввода)
cd exp && py simulate.py          # или py simulate.py --cv-auto
# -> figs/comparison.pdf   (таблица CV vs теория + наложенные профили)
# -> figs/drop_result.pdf  (CV-профиль из измеренных углов)
# -> figs/cv_theory_sideview.pdf (то же)

# Ручное сравнение CV с теорией (ввод чисел с клавиатуры)
cd exp && py simulate.py --cv

# Параметрическая симуляция (интерактивный ввод V/theta/alpha)
cd exp && py simulate.py --volume 10 --alpha 30 --theta 72

# 3D-рендер капли воды (вводятся только объём и угол наклона)
cd sim && py render3d.py
# -> figs/drop_3d.pdf

# Валидация CV на синтетике
py run_cv.py

# Рисунки статьи
cd sim && py figures.py && cp ../figs/*.pdf ../article/figs/

# Статья (всегда дважды)
cd article && xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex

# Просмотр PDF как PNG
pdftoppm -png -r 90 -f 1 -l 1 article/main.pdf /tmp/pg
```

---

## Реальный снимок exp/exp_data.jpg

- Камера стоит **вертикально** над наклонным зеркалом с каплей.
- Линейка лежит на зеркале; шаг — **1/64 дюйма** (0.397 мм), точность 1/128".
- Линейка наклонена на **~-28°** в кадре (определено через Хаф).
- Физический угол наклона зеркала = **90° − alpha_px** (alpha_px из синей линии).
- Ручная разметка маркером:
  - **синяя линия** — поверхность зеркала (базовая линия)
  - **красная кривая** — боковой профиль капли (один контакт → вершина → другой контакт)
- Вершина = точка красной кривой с max расстоянием от синей линии.
- **Пока один снимок** — сравнение CV vs теория даёт одну точку данных, не статистику.

---

## Частые ошибки

| Ошибка | Решение |
|--------|---------|
| `cv2.imread` возвращает None | Кириллица в пути — `np.fromfile + cv2.imdecode` |
| `cv2.imwrite` не сохраняет | Кириллица в пути — `cv2.imencode(...)[1].tofile(path)` |
| `UnicodeEncodeError` cp1251 | Убрать эмодзи из print, заменить на ASCII-текст |
| `ModuleNotFoundError` | Запускать из нужной директории или добавить `sys.path.insert` |
| Кириллица не видна в matplotlib | `rcParams["font.family"] = "DejaVu Sans"` |
| Ссылки `??` в PDF | Компилировать статью **дважды** |
| `t2aenc.def not found` | Использовать xelatex, не pdflatex |
| Рисунок не виден в PDF | `cp figs/*.pdf article/figs/` |
| simulate.py не найден в sim/ | Файл находится в `exp/simulate.py` |

---

## Физический справочник

| Величина | Формула |
|----------|---------|
| Капиллярная длина | a = sqrt(σ / Δρg) |
| Число Бонда | Bo = ΔρgL² / σ = (L/a)² |
| Константа c | c = Δρg / σ = 1/a² |
| 2D ОДУ | dφ/ds = b + c·z |
| 3D осесимм. ОДУ | dφ/ds = 2b + c·z − sinφ/x |
| Предельная высота 2D | h* = 2a sin(θ/2) |
| Критерий Фурмиджа | ρgV sinα = kw σ (cosθR − cosθA) |
| Симм. гистерезис | sinδ = ρgV sinα / (2σw sinθY) |
| Кубика угла | cos θ(ψ) = 2D/π³ ψ³ − 3D/π² ψ² + cos θmax |
| ADSA | σ = Δρg / c_fitted |

---

## Провалидированные числа

| Проверка | Точность |
|----------|----------|
| Точное 2D vs дуговое (площадь) | 0.000% |
| Осесимм. 3D vs сферическая шапка (c→0) | 0.04% |
| Метод двух окружностей — вырожд. (θmax=θmin) | 0.000% |
| CV краевой угол (θ < 115°) | < 2° |
| CV ADSA восстановление σ воды | 72.8 мН/м |

---

## Текущая повестка

1. **Эксперимент (готово)**: снимок `exp/exp_data.jpg` обработан CV — theta_L=23.6°,
   theta_R=37.7°, base=6.16 мм, h=0.73 мм, alpha_phys=30.3°, px/mm=83.15.
2. **simulate.py (готово)**: `py simulate.py` (без аргументов) → авто-режим,
   читает `cv_result.json`, строит два профиля:
   - красный (CV): theta_A=37.7°, theta_R=23.6°, base=6.16 мм, h=0.80 мм
   - синий (теория, hysteresis_symmetric): theta_A=37.5°, theta_R=23.8°, base=6.77 мм, h=0.88 мм
   Расхождение: base −9.1%, h −16.9%. Сохраняет `figs/comparison.pdf` и `figs/drop_result.pdf`.
3. **render3d.py (готово)**: CLI спрашивает только V и alpha, theta_Y=72.8° фиксирован.
   3D-капля асимметрична (theta_A/theta_R из hysteresis_symmetric), повёрнута на склон.
4. **Статья**: добавить таблицу свойств жидкостей, расширить список литературы.
5. **Больше снимков**: пайплайн готов — при добавлении новых фото запускать `dropcv.py`
   для каждого (перезаписывает `cv_result.json`) и сравнивать через `py simulate.py`.

---

## Известные ограничения

- Метод двух дуг — приближение; точная форма требует интегрирования Юнга-Лапласа с пиннингом.
- CV-угол занижает при θ > 130°.
- Высота по CV может быть занижена если красная разметка не доходит до вершины.
- Объём CV не измеряет: V ≈ π·h·(3R²+h²)/6 где R = base/2.
- Один снимок — статистики нет.

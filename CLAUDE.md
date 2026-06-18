# CLAUDE.md — контекст для Claude Code

## Что это за проект

Замкнутое физическое исследование равновесной формы капли жидкости (sessile drop).
Три артефакта: (1) LaTeX-статья, (2) симуляционное ядро, (3) CV-конвейер для снимков.
Статья готова (v2.0). Текущая работа — симуляции, 3D-рендер, обработка экспериментальных данных.

---

## Язык и стиль

- **Общение с пользователем**: русский, кратко, ведущая мысль вперёд.
- **Комментарии и docstring в коде**: русский.
- **Статья (LaTeX)**: русский текст, английские термины вводятся один раз в скобках при первом упоминании.
- Десятичный разделитель в статье: **запятая** (`2{,}73 мм`), не точка.
- Тире в LaTeX: `---`, не дефис.
- Выделение в тексте: `\emph{}`, не `\textbf{}`.

---

## Окружение

- Python: **3.11** (системный). Вызывать как `python3` или `python3.11`.
- Пакеты: numpy 2.4, scipy 1.17, matplotlib 3.10, opencv-python 4.13 — уже установлены.
- Для установки новых пакетов: `pip3 install PKG --break-system-packages`.
- LaTeX: только **xelatex**. `pdflatex` не работает (t2aenc.def не установлен).
- Шрифты: DejaVu Serif/Sans/Mono — есть кириллица, работают в matplotlib и LaTeX.
- Сеть в bash: доступны github, pypi, ubuntu. CTAN недоступен (`tlmgr install` не работает).

---

## Структура

```
sim/
  physics.py      — Fluid, WATER/GLYCERIN/ETHYLENE_GLYCOL/MERCURY, sphere_cap_*
  drop_2d.py      — 2D горизонталь (точное + дуговое решение)
  drop_axisym.py  — 3D осесимметрия (fit_b_for_volume, drop_volume)
  inclined.py     — наклон: max_volume_3d, hysteresis_symmetric, two_arc_profile_2d,
                    contact_angle_azimuthal, two_circle_volume
  figures.py      — генератор 8 рисунков статьи → figs/
  simulate.py     — CLI + API run_simulation() + plot_sideview()
  render3d.py     — surface_axisym(), surface_inclined(), render_and_save()
  synthetic.py    — синтетические снимки из физ. модели

exp/
  dropcv.py       — CV-конвейер: контур, baseline, краевые углы, ADSA
run_cv.py         — раннер CV

figs/             — 8 рисунков (PDF+PNG), генерируются figures.py
article/
  main.tex        — LaTeX-статья
  main.pdf        — скомпилированный PDF
  figs/           — копии рисунков для \includegraphics
literature/       — ключевые статьи (ElSherbini 2004, Rotenberg 1983, Lv 2017, ...)
```

---

## Зависимости между файлами

```
physics.py
  └─► drop_2d.py
  └─► drop_axisym.py
  └─► inclined.py
          └─► figures.py
          └─► simulate.py
          └─► render3d.py
          └─► synthetic.py (через exp/dropcv.py sys.path)
```

**Правило при изменении физики**: правка в `physics/drop_2d/drop_axisym/inclined`
→ перезапустить `figures.py` → `cp figs/*.pdf article/figs/` → перекомпилировать статью.

---

## Ключевые API

### simulate.py
```python
from simulate import run_simulation, plot_sideview
result = run_simulation(volume_ul=30, alpha_deg=35, theta_deg=100, fluid=WATER)
# result["geometry"] — словарь (base_width_mm, apex_height_mm, bond_number, ...)
# result["profile_x"], result["profile_z"] — профиль в мм
# result["axisym_profile"] — DropProfile или None (для наклонной)
# result["warnings"] — список строк

plot_sideview(result, save_path="/tmp/fig.pdf", show=False)
```

Исключения: `DropSlidesError`, `VolumeError`, `AngleError` (все подклассы `DropPhysicsError`).

### render3d.py
```python
from render3d import render_and_save, surface_axisym, surface_inclined
render_and_save(result, path="/tmp/drop_3d.pdf", view="iso")
# view: 'iso' | 'side' | 'top' | 'front'
```

### drop_axisym.py
```python
prof = fit_b_for_volume(volume_m3, fluid.c, theta_rad)
# prof.x          — полуширина (м), от вершины вниз
# prof.z          — расстояние от вершины вниз (0 у вершины, растёт к основанию)
# prof.apex_height — полная высота (м)
# высота над подложкой: prof.apex_height - prof.z
```

### inclined.py
```python
max_volume_3d(alpha_rad, thA_rad, thR_rad, fluid, width_m)  # → V_max, м³
hysteresis_symmetric(alpha_rad, V_m3, theta_Y_rad, fluid, w_m)  # → (θA, θR, slid)
two_arc_profile_2d(zeta_m, theta_down_rad, theta_up_rad, n=200)  # → (x, y, info)
contact_angle_azimuthal(psi_array, theta_max_rad, theta_min_rad)  # → θ(ψ), кубика
```

---

## Команды сборки

```bash
# Проверка импортов
cd sim && python3 -c "import physics,drop_2d,drop_axisym,inclined,simulate,render3d; print('OK')"

# Симуляция
cd sim && python3 simulate.py --volume 30 --alpha 35 --theta 100
cd sim && python3 simulate.py --volume 30 --alpha 0  --theta 112 --3d --save /tmp/drop3d.pdf

# Рисунки статьи
cd sim && python3 figures.py && cp ../figs/*.pdf ../article/figs/

# CV-конвейер
python3 run_cv.py

# Статья (всегда дважды)
cd article && xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex

# Просмотр PDF как PNG
pdftoppm -png -r 90 -f 1 -l 1 article/main.pdf /tmp/pg
```

---

## Частые ошибки

| Ошибка | Решение |
|--------|---------|
| `ModuleNotFoundError` при запуске из неправильной директории | Запускать из `sim/` или добавить `sys.path.insert(0, "sim")` |
| Кириллица не видна в matplotlib | Добавить `rcParams["font.family"] = "DejaVu Sans"` |
| `numpy.ndarray` не итерируется при малом V | Проверить диапазон V перед вызовом `fit_b_for_volume` |
| `plot_surface` не работает | Добавить `from mpl_toolkits.mplot3d import Axes3D` |
| Ссылки `??` в PDF | Компилировать статью **дважды** |
| `t2aenc.def not found` | Использовать xelatex, не pdflatex |
| Рисунок не виден в PDF | Скопировать в `article/figs/`: `cp figs/*.pdf article/figs/` |

---

## Физический справочник

| Величина | Формула |
|----------|---------|
| Капиллярная длина | a = √(σ / Δρg) |
| Число Бонда | Bo = ΔρgL² / σ = (L/a)² |
| Константа c | c = Δρg / σ = 1/a² |
| 2D ОДУ | dφ/ds = b + c·z |
| 3D осесимм. ОДУ | dφ/ds = 2b + c·z − sinφ/x |
| Предельная высота 2D | h* = 2a sin(θ/2) |
| Критерий Фурмиджа | ρgV sinα = kw σ (cosθR − cosθA) |
| Симм. гистерезис | sinδ = ρgV sinα / (2σw sinθY); θA = θY+δ, θR = θY−δ |
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

## Текущая повестка (next steps)

1. **Симуляции**: render3d.py готов — тестировать 3D-рендер при разных параметрах.
2. **Эксперимент**: подключить реальные снимки к `exp/dropcv.py`, сравнить GT↔CV.
3. **Статья**: добавить таблицу свойств жидкостей, расширить список литературы.
4. **Тесты**: pytest с регрессионными проверками из "Провалидированные числа".

---

## Известные ограничения

- Метод двух дуг для наклонной капли — приближение; точная форма требует интегрирования Юнга–Лапласа с пиннингом контактной линии.
- CV-угол занижает при θ > 130° (ограничение фита окружностью).
- Синтетические снимки из той же модели — проверяют CV-конвейер, но не валидируют модель против реальных фото.

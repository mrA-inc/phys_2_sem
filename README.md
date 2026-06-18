# Равновесная форма капли жидкости

Исследование равновесной формы сидящей (sessile) капли в зависимости от угла
наклона поверхности, поверхностного натяжения и условия начала соскальзывания.
Три артефакта в порядке приоритета:
1. **Статья** — LaTeX-статья журнального уровня (`article/main.tex`)
2. **Симуляция** — расчёт формы и параметрическое CLI (`sim/`)
3. **CV-анализ** — компьютерное зрение для снимков капли (`exp/`)

---

## Структура репозитория

```
phys_2_sem/
├── sim/
│   ├── physics.py        — физические константы; класс Fluid, жидкости
│   ├── drop_2d.py        — 2D капля на горизонтали (точное + дуговое решение)
│   ├── drop_axisym.py    — 3D осесимметрия (Башфорт–Адамс, объём)
│   ├── inclined.py       — наклон (Фурмидж, гистерезис, метод двух окружностей)
│   ├── simulate.py       — параметрическое CLI + API run_simulation()
│   ├── render3d.py       — 3D-рендер поверхности капли (matplotlib)
│   └── synthetic.py      — генератор синтетических снимков из физ. модели
├── exp/
│   └── dropcv.py         — CV-конвейер: контур, базовая линия, углы, ADSA
├── figs/                 — 8 рисунков статьи (PDF + PNG)
├── article/
│   ├── figures.py        — генерация 8 рисунков статьи → figs/
│   ├── figs/             — копии рисунков для LaTeX \includegraphics
│   ├── main.tex          — исходник статьи (xelatex)
│   └── main.pdf          — скомпилированный PDF (20 стр., 37 уравнений, 11 рис.)
├── literature/           — ключевые статьи (Rotenberg 1983, ElSherbini 2004 и др.)
├── run_cv.py             — раннер CV (обрабатывает горизонтальный и наклонный снимки)
└── figures.py            — alias / копия для генерации рисунков из корня
```

---

## Физическая модель

| Задача | Метод | Файл |
|--------|-------|------|
| 2D горизонталь | Точное решение первого интеграла + дуговой ОДУ | `sim/drop_2d.py` |
| 3D горизонталь | Уравнение Башфорта–Адамса (осесимметрия) | `sim/drop_axisym.py` |
| Критерий соскальзывания | Критерий Фурмиджа: ρgV sinα = kw σ(cosθR − cosθA) | `sim/inclined.py` |
| Форма на наклоне | Метод двух окружностей (ElSherbini 2004) | `sim/inclined.py` |
| Краевой угол по периметру | Кубическая аппроксимация θ(ψ) | `sim/inclined.py` |
| CV-измерение | Локальный фит окружности у каждого контакта | `exp/dropcv.py` |
| Поверхностное натяжение | ADSA (подгонка Юнга–Лапласа к контуру) | `exp/dropcv.py` |

Капиллярная длина **a = √(σ / Δρg)**; число Бонда **Bo = (L/a)²**.
Уравнение Юнга–Лапласа в дуговой параметризации:
- 2D: `dφ/ds = b + c·z`
- 3D (осесимм.): `dφ/ds = 2b + c·z − sinφ/x`

---

## Быстрый старт

```bash
# 1. Проверить импорты sim-модулей
cd sim && python3 -c "import physics,drop_2d,drop_axisym,inclined,simulate; print('OK')"

# 2. Симуляция — горизонтальная капля
python3 simulate.py --volume 30 --alpha 0 --theta 100 --save /tmp/drop.pdf

# 3. Симуляция — наклонная капля с 3D-рендером
python3 simulate.py --volume 30 --alpha 35 --theta 100 --3d --save /tmp/drop3d.pdf

# 4. Проверка ошибки соскальзывания
python3 simulate.py --volume 200 --alpha 60 --theta 100
# → DropSlidesError с критическим углом

# 5. Перегенерировать рисунки статьи
python3 figures.py && cp ../figs/*.pdf ../article/figs/

# 6. Запустить CV-конвейер
cd .. && python3 run_cv.py

# 7. Скомпилировать статью (дважды — для ссылок)
cd article && xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex
```

---

## CLI simulate.py

```
python3 simulate.py --volume V --theta θ [--alpha α] [--fluid ЖИДКОСТЬ] [--width ш] [--3d] [--save файл]

Аргументы:
  --volume   объём капли, мкл (обязательно)
  --theta    равновесный краевой угол, ° (обязательно)
  --alpha    угол наклона поверхности, ° (по умолч. 0)
  --fluid    жидкость: water | glycerin | ethylene_glycol | mercury (по умолч. water)
  --width    ширина контактного пятна поперёк склона, мм (по умолч. авто)
  --3d       добавить 3D-рендер поверхности
  --save     путь к выходному PDF
```

Скрипт выполняет физическую валидацию и выдаёт информативную ошибку:
- `VolumeError`     — объём вне [0.01, 10000] мкл
- `AngleError`      — угол наклона вне [0°, 90°) или краевой угол вне (1°, 179°)
- `DropSlidesError` — капля соскальзывает; сообщается V_max и критический угол α_c

---

## API Python

```python
from sim.simulate import run_simulation
from sim.physics import WATER

result = run_simulation(volume_ul=30, alpha_deg=35, theta_deg=100, fluid=WATER)
print(result["geometry"])   # словарь с base_width_mm, apex_height_mm, Bo и др.

# 3D-рендер
from sim.render3d import render_and_save
render_and_save(result, path="/tmp/drop_3d.pdf", view="iso")
```

---

## Провалидированные числа

| Проверка | Расхождение |
|----------|------------|
| Точное 2D vs дуговое (площадь) | 0.000 % |
| Осесимм. 3D vs сферическая шапка (без гравитации) | 0.04 % |
| Метод двух окружностей — вырожд. случай vs шапка | 0.000 % |
| CV краевой угол ошибка до θ ≈ 115° | < 2° |
| CV восстановление σ воды (ADSA) | 72.8 мН/м (точно) |

---

## Зависимости

Python 3.11+, numpy ≥ 2.4, scipy ≥ 1.17, matplotlib ≥ 3.10, opencv-python ≥ 4.13.

Статья компилируется только через **xelatex** (pdflatex не работает: T2A кодировка отсутствует).
Шрифты: DejaVu Serif/Sans/Mono (есть кириллица — для matplotlib и LaTeX).

---

## Ключевые источники

- Bashforth & Adams (1883) — дуговая параметризация профиля
- Frenkel (1948), Furmidge (1962) — критерий соскальзывания
- Rotenberg, Boruvka & Neumann (1983) — метод ADSA
- ElSherbini & Jacobi (2004) — метод двух окружностей, кубика θ(ψ)
- Lv et al. (2017), arXiv:1705.03548 — точные 2D-решения
- Матюхин & Фроленков (2013) — вариационный вывод, насыщение высоты h*

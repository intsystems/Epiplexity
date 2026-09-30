# Проект: Epiplexity

Описание проекта, архитектуры, публичного API и библиотек.
Документ составлен на основе текущего состояния репозитория и плана в
[`plan/README.md`](plan/README.md).

---

## 1. Название проекта

**Epiplexity** — оценка *эпиплексити* $S_T$ малыми моделями (MLP/CNN/DiT) на малых
датасетах на одной GPU.

- Основной источник: Finzi, Qiu et al., 2026 — [arXiv:2601.03220](https://arxiv.org/abs/2601.03220).
- Код проекта живёт в Python-пакете **`epimeter`** (`src/epimeter/`).
- Репозиторий — общий каркас для команды из четырёх человек; замыкает два курса
  (байесовские методы BMM и R&D «Интеллектуальные системы»), целевой вендор — CPAL 2027.
- Литературный обзор, план, роли и календарь: [`notes/epiplexity.pdf`](notes/epiplexity.pdf),
  [`plan/README.md`](plan/README.md).

### Цель

Реализовать интерфейс, который возвращает разложение кодовой длины на
**(модельная часть, данных часть)** в битах для преквентной, реквентной,
вариационной и лапласовской схем, классических критериев, EDL и reservoir-score,
и измерить эпиплексити для малых MLP/CNN-наблюдателей. Основные исследовательские
вопросы (RQ1–RQ4) зафиксированы в плане.

---

## 2. Архитектура проекта

Код организован как небольшой PyTorch-пакет с **интерфейсами** (abstract base
classes) и **реестрами** для расширяемости. Ядро пакета ничего не знает о конкретных
сетях: наблюдатели и сэмплеры подключаются через декоратор `@register`.

### Слои и классы

```
                    ┌──────────────────────────────────────────────┐
   данные ─────────▶│ train(model, train_loader)                   │
                    │            │                                 │
                    │            ▼                                 │
                    │   BaseModel  (nn.Module, наблюдатель)        │
                    │            │                                 │
                    │            ▼                                 │
                    │   BaseEstimator ── fit(model, data)          │
                    │            │        estimate() ──▶ (S, H)    │
                    │            ▼                                 │
                    │   BaseSampler  (для байесовских оценщиков)   │
                    └──────────────────────────────────────────────┘
                            │  результат в битах
                            ▼
                       evaluate(model, eval_loader, estimator)
```

- **`BaseEstimator`** (`src/epimeter/base.py`) — абстрактный контракт всех оценщиков.
  Разделён на *состоящий* `fit` (может обучать модели, считать свипы, сэмплировать)
  и *независящий от состояния* `estimate`, возвращающий `(model_part, data_part)` в битах.
  Конкретные скелеты: `Prequential`, `Requential`, `Bayesian`, `Proxies`.

- **`BaseModel`** (`src/epimeter/models/base.py`) — «наблюдатель»: любая сеть
  (MLP, CNN, DiT), используемая для оценки. Наследует `torch.nn.Module`, обязан
  реализовать `forward`. Дополнительно даёт `count_parameters`, `save`, `load`.

- **Реестры** (`models/registry.py`, `samplers/registry.py`) — `register(name)` и
  `build(name, **config)`. Точка расширяемости: внешняя сеть регистрируется и
  становится доступной через `epimeter.build("mlp", ...)` без изменений в ядре
  (пример — `experiments/example_mlp.py`).

- **`BaseSampler`** (`src/epimeter/samplers/base.py`) — рисует сэмплы параметров
  модели для байесовских оценщиков (например, SGLD, априорное распределение весов).

- **`train` / `evaluate`** (`training.py`, `eval.py`) — верхнеуровневые функции:
  обучают модель и вычисляют оценку поверх модели и датасета.

### Структура репозитория

```
src/epimeter/
  base.py                BaseEstimator (абстрактный контракт)
  estimators/base.py     Prequential, Requential, Bayesian, Proxies (скелеты)
  models/                BaseModel + реестр наблюдателей
  samplers/              BaseSampler + реестр сэмплеров
  training.py            train(...)
  eval.py                evaluate(...)
  utils/logging.py       configure(...)
experiments/             исполняемые скрипты, подключённые через реестры
tests/                   pytest (структурные + проверка расширяемости)
docs/                    Sphinx (скелет, заполняется на стадии 7)
plan/  research_notes/  notes/  slides/   планирование, литература, презентация
```

### Взаимодействие классов

1. Пользователь регистрирует сеть/сэмплер через `@register("name")` или использует
   уже зарегистрированные.
2. `train(model, loader)` обучает `BaseModel` на данных.
3. Оценщик (наследник `BaseEstimator`) вызывает `fit(model, data)` — строит своё
   внутреннее состояние (свип, сэмплы, пол), затем `estimate()` возвращает
   `(model_part_bits, data_part_bits)`.
4. Для байесовских схем оценщик использует `BaseSampler.sample(...)` для получения
   сэмплов параметров.
5. `evaluate(model, eval_loader, estimator)` агрегирует результат и возвращает его
   потребителю (эксперименту/ноутбуку).

---

## 3. Планируемые публичные функции и методы (с аннотациями)

> Обозначение `⏳` — сигнатура зафиксирована, реализация появится на стадии 4.
> Сейчас эти методы бросают `NotImplementedError`.

### 3.1. Базовый оценщик

`src/epimeter/base.py`

```python
class BaseEstimator(ABC):
    name: str = "estimator"
    units: str = "bits"

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> "BaseEstimator":
        """Обучить/настроить оценщик на данных; возвращает self (для chaining)."""

    @abstractmethod
    def estimate(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Возвращает (model_part_bits, data_part_bits)."""
```

- Реализации возвращают кодовые длины в **битах** (`units == "bits"`).
- `fit` — с состоянием, `estimate` — без изменения состояния.

### 3.2. Конкретные оценщики

`src/epimeter/estimators/base.py` — все наследуют `BaseEstimator`,
переопределяют `fit`/`estimate`.

```python
class Prequential(BaseEstimator):
    """Онлайн (преквентный) код: площадь над полом и его варианты.
    Стадия 4: онлайн-код, полы held-out / EDL / training-loss,
    блочные варианты Bornschein et al. (2022)."""
    name = "prequential"

class Requential(BaseEstimator):
    """Двухчастный / реквентный MDL-код."""
    name = "requential"

class Bayesian(BaseEstimator):
    """Вариационная и лапласовская байесовские длины кода."""
    name = "bayesian"

class Proxies(BaseEstimator):
    """Классические критерии и быстрые прокси: AIC, BIC, HQIC, WAIC, WBIC,
    EDL и reservoir score (стадия 4)."""
    name = "proxies"
```

### 3.3. Модель-наблюдатель

`src/epimeter/models/base.py`

```python
class BaseModel(nn.Module, ABC):
    """Наблюдатель: любая сеть (MLP, CNN, DiT) для оценки эпиплексити."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход сети. Обязан реализовать подкласс."""

    def count_parameters(self) -> int:
        """Число обучаемых параметров сети."""
        # return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        """Сохранить state_dict в ``path``."""

    def load(self, path: str | Path) -> None:
        """Загрузить state_dict из ``path`` на месте."""
```

### 3.4. Сэмплер

`src/epimeter/samplers/base.py`

```python
class BaseSampler(ABC):
    @abstractmethod
    def sample(self, model: Any, num_samples: int, shape: Sequence[int]) -> torch.Tensor:
        """Рисует ``num_samples`` сэмплов параметров формы
        ``(num_samples,) + shape`` (например, из апостериорного распределения)."""
```

### 3.5. Реестры

`src/epimeter/models/registry.py` и `src/epimeter/samplers/registry.py`

```python
def register(name: str) -> Callable[[type[Any]], type[Any]]:
    """Декоратор: регистрирует класс (наблюдатель/сэмплер) под именем ``name``."""

def build(name: str, **config: Any) -> Any:
    """Инстанцирует класс по имени с аргументами ``config``.
    Бросает NotImplementedError, если имя не зарегистрировано."""
```

### 3.6. Обучение и оценка

`src/epimeter/training.py` ⏳

```python
def train(model: Any, train_loader: Any, **kwargs: Any) -> Any:
    """Обучить ``model`` на ``train_loader``; возвращает обученную модель.
    Стадия 4: цикл с Adam, lr, масштабированной как 1/fan-in,
    постоянной скоростью и EMA весов (общий протокол в plan/README.md)."""
```

`src/epimeter/eval.py` ⏳

```python
def evaluate(model: Any, eval_loader: Any, estimator: Any, **kwargs: Any) -> Any:
    """Вычислить ``estimator`` (модельная и данных части в битах) на ``eval_loader``."""
```

### 3.7. Логирование

`src/epimeter/utils/logging.py`

```python
def configure(level: int = logging.INFO) -> None:
    """Настроить логгер epimeter (однократно, при первом вызове)."""
```

### 3.8. Публичный интерфейс пакета

`src/epimeter/__init__.py` экспортирует:

```
__version__, BaseEstimator, BaseModel, BaseSampler,
Prequential, Requential, Bayesian, Proxies, build, register
```

---

## 4. Используемые и планируемые библиотеки

### Обязательные зависимости (`pyproject.toml` → `dependencies`)

| Библиотека | Версия | Назначение |
|---|---|---|
| **torch** | `>=2.0` | тензоры, `nn.Module` для моделей-наблюдателей, обучение, сам счёт |
| **numpy** | `>=1.24` | численные операции, массивы результатов, свипы |

> Пакет установлен через `src`-layout с помощью `setuptools`; управление средой — **uv**.

### Опциональные (extras)

| Extra | Библиотека | Назначение |
|---|---|---|
| `bayesian` | `pyro-ppl>=1.9` | вариационные коды для `Bayesian`-оценщика (подключается на поздних стадиях, чтобы держать ядро лёгким) |

### Dev-зависимости (extra `dev`)

| Библиотека | Назначение |
|---|---|
| `pytest>=7.4`, `pytest-cov>=4.1` | тесты и покрытие (цель — >90%) |
| `ruff>=0.4` | линтер/форматтер |
| `matplotlib>=3.7` | графики (кривые $S_T$, variance study) |
| `jupyter>=1.0` | ноутбуки RQ1–RQ4 и Colab-демо |

### Инструменты и интеграции

- **Sphinx** (`docs/`) — автодокументация по API (структура на месте, заполняется на стадии 7).
- **Colab GPU** (T4) — среда прогонов; каждый ран сохраняет чекпоинт и умеет резюмироваться.
- **Git + GitLab** — репозиторий, MR, потенциальный CI для тестов и линта.
- **Данные/тестбеды** — синтетические генераторы на GPU (ECA, label noise, permuted pixels,
  planted teacher–student), которые создаются в рамках проекта (не внешняя библиотека).

---

## 5. Текущее состояние репозитория

**Реализовано (стадии 1–3): интерфейсный скелет.**

-  Импорт пакета без side-effect; публичный интерфейс (`__init__.py`) собран.
-  `BaseEstimator` (абстрактный), скелеты `Prequential`, `Requential`, `Bayesian`, `Proxies`.
-  `BaseModel` (`nn.Module`) с `count_parameters`, `save`, `load`.
-  `BaseSampler` с методом `sample`.
-  **Работающие реестры** `models` и `samplers`: `register`/`build` уже функциональны —
  зарегистрированную сеть можно получить через `epimeter.build("mlp", ...)` (см.
  `experiments/example_mlp.py`, `experiments/example_sampler.py`).
-  `utils.logging.configure`.
-  8 структурных тестов в `tests/` зелёные на скелете; настроены `pytest`, `ruff`.
-  Протокол эксперимента зафиксирован в `plan/README.md`: биты, пол (held-out EMA-модели),
  ≥5 сидов, контроли (random/constant labels, shuffled targets), JSON-записи результатов,
  RQ1–RQ4, роли четырёх участников, календарь до CPAL 2027.

**Запланировано (стадии 4+):**

- ⏳ Реализация `Prequential` (онлайн-код, полы EDL/held-out, блочные варианты).
- ⏳ Реализация `Requential` (двухчастный MDL-код).
- ⏳ Реализация `Bayesian` (variational/Laplace) с использованием `pyro-ppl` и `BaseSampler`.
- ⏳ Реализация `Proxies` (AIC, BIC, HQIC, WAIC, WBIC, EDL, reservoir score).
- ⏳ Реализация `train`/`evaluate`, регистрация реальных MLP/CNN/DiT, ноутбуки RQ1–RQ4,
  Sphinx-документация, Docker, статья.

Лицензия: MIT.

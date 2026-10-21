"""Запуск прогона обучения по файлу конфигурации.

Файл JSON с разделами `data` (откуда ряды), `spec` (форма задачи, `WindowSpec`), `model`
(конфигурация Moirai, `MoiraiConfig`) и `train` (параметры прогона, `TrainConfig`). Раздел `data`
задаёт генератор из реестра с генерацией на лету, либо корпус этого генератора (каталоги `corpus`),
либо равную смесь нескольких таких источников (`mixture`). Поле `context_table` — путь к JSON с
полями `lengths` и `probs`, которые подставляются в `spec` как распределение длин контекста.
Любое поле переопределяется из командной строки через `--set раздел.поле=значение` (значение
читается как JSON, иначе как строка). Разрешённая конфигурация пишется в каталог прогона как
`run.json`; существующий `run.json` не перезаписывается, а прогон с `last.pt` возобновляется.

    python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth --set train.lr=3e-4
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .corpus import Corpus
from .generators import GENERATORS, Generator, Mixture
from .model.moirai import MoiraiConfig, small
from .training import TrainConfig, Trainer, build_model
from .windows import WindowSpec


class DataConfig(BaseModel):
    """Источник рядов.

    Attributes:
        generator: Имя генератора в реестре; у смеси не используется, у корпуса служит проверкой.
        corpus: Каталоги корпуса этого генератора; пусто — генерация на лету.
        generator_args: Аргументы конструктора генератора при генерации на лету.
        mixture: Компоненты равной смеси; непусто — остальные поля не используются.
    """

    generator: str = ""
    corpus: list[str] = []
    generator_args: dict = {}
    mixture: list["DataConfig"] = []


class RunConfig(BaseModel):
    """Полная конфигурация прогона.

    Attributes:
        data: Источник рядов.
        spec: Форма задачи.
        model: Конфигурация модели; по умолчанию Moirai-Small с патчем 32 и тремя непрерывными компонентами.
        train: Параметры прогона.
        context_table: Путь к JSON с `lengths` и `probs`; задаёт распределение длин контекста в `spec`.
    """

    data: DataConfig
    spec: WindowSpec = WindowSpec()
    model: MoiraiConfig = Field(default_factory=lambda: small(patch_sizes=(32,), components=("student_t", "normal_fixed", "log_normal")))
    train: TrainConfig
    context_table: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        if self.spec.patch_size not in self.model.patch_sizes:
            raise ValueError(f"размер патча {self.spec.patch_size} не входит в {self.model.patch_sizes}")
        if self.spec.n_tokens > self.model.max_seq_len:
            raise ValueError("окно длиннее max_seq_len модели")
        return self


def resolve_context_table(raw: dict) -> dict:
    """Подставляет таблицу длин контекста из файла `context_table` в раздел `spec` сырой конфигурации."""
    path = raw.get("context_table")
    if path:
        table = json.loads(Path(path).read_text())
        raw.setdefault("spec", {}).update(context_lengths=list(table["lengths"]), context_probs=list(table["probs"]))
    return raw


def build_generator(data: DataConfig, length: int) -> Generator:
    """Источник рядов по разделу `data`: генератор, корпус или смесь.

    Args:
        data: Раздел конфигурации.
        length: Длина ряда из формы задачи.

    Returns:
        Generator: Источник рядов.
    """
    if data.mixture:
        return Mixture([build_generator(d, length) for d in data.mixture])
    if data.corpus:
        corpus = Corpus(*data.corpus)
        if data.generator and corpus.name != data.generator:
            raise ValueError(f"корпус генератора {corpus.name}, а в конфигурации {data.generator}")
        if corpus.length != length:
            raise ValueError(f"длина ряда корпуса {corpus.length}, а в форме задачи {length}")
        return corpus
    return GENERATORS[data.generator](length=length, **data.generator_args)


def seeds_available(generator: Generator, first_seed: int, n: int) -> bool:
    """Покрывает ли источник рядов сиды `[first_seed, first_seed + n)`: корпус — своими диапазонами, смесь — компонентами, генератор — всегда."""
    if isinstance(generator, Mixture):
        k = len(generator.components)
        lo, hi = first_seed // k, (first_seed + n - 1) // k + 1
        return all(seeds_available(g, lo, hi - lo) for g in generator.components)
    if isinstance(generator, Corpus):
        return any(first_seed >= a and first_seed + n <= b for a, b in generator.seed_ranges)
    return True


def check_seed_ranges(generator: Generator, train: TrainConfig) -> None:
    """Проверяет до запуска, что обучающие и отложенные сиды есть в источнике рядов; иначе `ValueError`."""
    for name, first, n in (("обучающие", train.first_seed, train.n_windows), ("отложенные", train.heldout_first_seed, train.heldout_windows)):
        if not seeds_available(generator, first, n):
            raise ValueError(f"{name} сиды [{first}, {first + n}) не покрыты источником рядов {generator.name}")


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Подставляет `раздел.поле=значение` в сырой словарь конфигурации; значение читается как JSON, иначе строка."""
    for item in overrides:
        key, _, value = item.partition("=")
        node = raw
        *path, last = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        try:
            node[last] = json.loads(value)
        except json.JSONDecodeError:
            node[last] = value
    return raw


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Конфигурация из файла с переопределениями, с подставленной таблицей длин контекста."""
    raw = resolve_context_table(apply_overrides(json.loads(Path(path).read_text()), overrides or []))
    return RunConfig.model_validate(raw)


def main() -> None:
    """Прогон из командной строки."""
    parser = argparse.ArgumentParser(description="Запустить или возобновить прогон обучения по файлу конфигурации.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="раздел.поле=значение")
    args = parser.parse_args()
    config = load_run_config(args.config, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "run.json").exists():
        (out / "run.json").write_text(config.model_dump_json(indent=1))
    generator = build_generator(config.data, config.spec.series_length)
    check_seed_ranges(generator, config.train)
    model = build_model(config.model, config.train.init_seed)
    Trainer(model, generator, config.spec, config.train, out).run()


if __name__ == "__main__":
    main()

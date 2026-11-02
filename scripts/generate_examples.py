"""Прогоняет товары из data/ через Groq и записывает примеры в examples/."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from app.domain import CardError, parse_product, render_notes_markdown
from app.generator import CardGenerator
from app.llm import GroqClient
from app.main import _generate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXAMPLES_DIR = ROOT / "examples"

BROKEN_INPUTS = (
    ("Пустой запрос", ""),
    ("Текст вместо JSON", "не json"),
    (
        "JSON обрезан при копировании",
        '{\n  "name": "Манометр показывающий DM 02",\n  "specs": {\n    "диапазон измерений": "0–1,6 МПа",\n',
    ),
    (
        "Пропущена запятая",
        '{\n  "name": "Манометр показывающий DM 02"\n  "specs": {}\n}',
    ),
    (
        "Лишняя запятая в конце",
        '{\n  "name": "Манометр показывающий DM 02",\n  "specs": {},\n}',
    ),
    ("Все поля пустые", '{\n  "name": "",\n  "manufacturer": "",\n  "description": "",\n  "specs": {}\n}'),
    ("Массив вместо объекта", "[]"),
    ("Характеристики списком", '{"name": "Манометр показывающий DM 02", "specs": ["класс точности"]}'),
)


def main() -> None:
    """Собирает карточки для всех JSON в data и сохраняет ответ и Markdown."""
    load_dotenv(ROOT / ".env")
    generator = CardGenerator(GroqClient.from_env())
    EXAMPLES_DIR.mkdir(exist_ok=True)
    paths = sorted(DATA_DIR.glob("*.json"))
    if not paths:
        raise SystemExit("В каталоге data нет JSON.")
    for path in paths:
        product = parse_product(json.loads(path.read_text(encoding="utf-8")))
        card = generator.generate(product)
        payload = card.model_dump()
        (EXAMPLES_DIR / f"{path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        notes = render_notes_markdown(card)
        (EXAMPLES_DIR / f"{path.stem}.md").write_text(
            f"{card.markdown}\n---\n\n*Служебные пометки для контент-менеджера, в карточку не входят:*\n\n{notes}",
            encoding="utf-8",
        )
        print(path.stem)
    write_invalid_inputs(generator)


def write_invalid_inputs(generator: CardGenerator) -> None:
    """Записывает ответы сервиса на типичные ошибки ввода. Модель при этом не вызывается.

    Args:
        generator: Сборщик карточки. До него битый вход не доходит.
    """
    parts = [
        "# Некорректный вход\n",
        "Ответы `POST /api/generate` на типичные ошибки. Во всех случаях HTTP 400, "
        "карточка не собирается, модель не вызывается. Поле `line` — строка, "
        "которую форма подсвечивает в JSON.\n",
    ]
    for title, body in BROKEN_INPUTS:
        try:
            _generate(body.encode("utf-8"), generator)
        except CardError as exc:
            status = exc.status_code
            response: dict[str, object] = {"error": exc.message}
            if exc.line is not None:
                response["line"] = exc.line
        else:
            raise SystemExit(f"Вход «{title}» неожиданно прошёл проверку.")
        shown = body if body else "(пустое тело)"
        parts.append(
            f"## {title}\n\nВход:\n\n```text\n{shown}\n```\n\n"
            f"Ответ `{status}`:\n\n```json\n"
            f"{json.dumps(response, ensure_ascii=False, indent=2)}\n```\n"
        )
    (EXAMPLES_DIR / "invalid_json.md").write_text("\n".join(parts), encoding="utf-8")
    print("invalid_json")


if __name__ == "__main__":
    main()

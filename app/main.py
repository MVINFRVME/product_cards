"""HTTP-слой: форма для контент-менеджера и JSON API."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.domain import CardError, ProductCard, parse_product
from app.generator import CardGenerator
from app.llm import GroqClient, LLMClient

load_dotenv()

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_LABELS = {
    "manometer": "Манометр DM 02",
    "pressure_sensor": "Датчик давления ДД-420",
    "manometer_incomplete": "Манометр DM 02 (с пропусками)",
}


def create_app(client: LLMClient | None = None) -> FastAPI:
    """Собирает приложение. В тестах сюда передают подменный клиент модели.

    Args:
        client: Реализация LLM. По умолчанию Groq из переменных окружения.

    Returns:
        Приложение с формой и POST /api/generate.
    """
    application = FastAPI(title="Генерация карточки товара")
    application.state.generator = CardGenerator(client or GroqClient.from_env())

    @application.exception_handler(CardError)
    async def handle_card_error(_request: Request, exc: CardError) -> JSONResponse:
        """Отдаёт понятную ошибку вместо падения сервера."""
        content: dict[str, object] = {"error": exc.message}
        if exc.line is not None:
            content["line"] = exc.line
        return JSONResponse(status_code=exc.status_code, content=content)

    @application.get("/")
    def index(request: Request):
        """Отдаёт форму с примерами из data/."""
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "samples_json": json.dumps(_load_samples(), ensure_ascii=False).replace("<", "\\u003c"),
            },
        )

    @application.post("/api/generate", response_model=ProductCard)
    async def generate(request: Request) -> ProductCard:
        """Принимает JSON товара и возвращает черновик карточки."""
        raw = await request.body()
        return await run_in_threadpool(_generate, raw, request.app.state.generator)

    return application


def _generate(raw: bytes, generator: CardGenerator) -> ProductCard:
    """Разбирает тело запроса и собирает карточку.

    Args:
        raw: Сырые байты тела, ещё не разобранные FastAPI.
        generator: Сборщик карточки для этого приложения.

    Returns:
        Черновик карточки.

    Raises:
        CardError: Пустое тело, не-JSON или товар не проходит проверку.
    """
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise CardError("Пустой запрос: вставьте JSON товара или выберите пример.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _json_error(text, exc) from exc
    product = parse_product(payload)
    has_text = any(value.strip() for value in (product.name, product.manufacturer, product.description))
    has_spec = any(value and value.strip() for value in product.specs.values())
    if not has_text and not has_spec:
        raise CardError("Нечего собирать: все поля пустые. Заполните хотя бы название прибора или выберите пример.")
    return generator.generate(product)


_JSON_REASONS = {
    "Expecting value": "на месте значения пусто или стоит лишняя запятая",
    "Expecting property name enclosed in double quotes": (
        "название поля должно быть в двойных кавычках, или после последнего поля стоит лишняя запятая"
    ),
    "Expecting ',' delimiter": "не хватает запятой между полями",
    "Expecting ':' delimiter": "после названия поля нужно двоеточие",
    "Unterminated string starting at": "не закрыта кавычка",
    "Extra data": "после закрывающей скобки «}» есть лишний текст",
    "Invalid control character at": "внутри значения есть перенос строки или табуляция",
    "Invalid \\escape": "внутри значения неверный обратный слэш «\\»",
}


def _json_error(text: str, exc: json.JSONDecodeError) -> CardError:
    """Превращает ошибку разбора в подсказку, где и что поправить.

    Args:
        text: Тело запроса строкой.
        exc: Ошибка стандартного json.

    Returns:
        CardError с текстом для контент-менеджера и номером строки для подсветки.
    """
    lines = text.splitlines() or [""]
    if exc.pos == len(text) - len(text.lstrip()):
        return CardError(
            "Это не JSON. Данные товара должны начинаться с «{». "
            "Нажмите любой пример, чтобы увидеть правильный формат.",
            line=1,
        )
    if exc.pos >= len(text.rstrip()):
        return CardError(
            "JSON обрывается на середине: не хватает закрывающей скобки «}» или кавычки. "
            "Проверьте, что скопировали его целиком.",
            line=len(lines),
        )
    reason = next(
        (value for key, value in _JSON_REASONS.items() if exc.msg.startswith(key)),
        "формат нарушен",
    )
    line = exc.lineno
    current = lines[line - 1] if line <= len(lines) else ""
    at_line_start = exc.colno <= len(current) - len(current.lstrip()) + 1
    if exc.msg.startswith("Expecting ',' delimiter") and at_line_start:
        previous = next((index for index in range(line - 1, 0, -1) if lines[index - 1].strip()), None)
        if previous is not None:
            line = previous
            reason = "в конце строки не хватает запятой"
    snippet = lines[line - 1].strip() if line <= len(lines) else ""
    if len(snippet) > 60:
        snippet = snippet[:59].rstrip() + "…"
    where = f" Строка {line}: «{snippet}»." if snippet else f" Строка {line}."
    return CardError(f"Не получается прочитать JSON: {reason}.{where}", line=line)


def _load_samples() -> list[dict[str, object]]:
    """Читает примеры из data/: сначала в порядке SAMPLE_LABELS, остальные по имени файла.

    Returns:
        Список с id файла, подписью кнопки и исходным JSON.
    """
    samples: list[dict[str, object]] = []
    if not DATA_DIR.exists():
        return samples
    order = list(SAMPLE_LABELS)
    paths = sorted(
        DATA_DIR.glob("*.json"),
        key=lambda path: (order.index(path.stem) if path.stem in order else len(order), path.stem),
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = payload.get("name") if isinstance(payload, dict) else None
        label = SAMPLE_LABELS.get(path.stem) or name or path.stem
        samples.append({"id": path.stem, "label": label, "payload": payload})
    return samples


app = create_app()

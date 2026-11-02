"""Входные данные товара, карточка и ошибки, которые видит пользователь."""

from __future__ import annotations

from pydantic import BaseModel, Field

MISSING = "данных нет"
REMOVED_INTRO = (
    "Нейросеть дописала эти фразы сама, в данных о приборе их нет. "
    "Если утверждение верное, верните его в текст вручную."
)


class CardError(Exception):
    """Ошибка входа или модели, которую можно отдать клиенту без падения сервера."""

    def __init__(self, message: str, status_code: int = 400, line: int | None = None) -> None:
        """Сохраняет текст для клиента и HTTP-статус.

        Args:
            message: Что показать пользователю. Без стека и без ключа.
            status_code: Код ответа. Для кривого JSON это 400.
            line: Номер строки входного JSON, где разбор споткнулся. Форма подсвечивает её.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.line = line


class ProductInput(BaseModel):
    """Данные прибора в формате из тестового задания."""

    name: str = ""
    manufacturer: str = ""
    description: str = ""
    specs: dict[str, str | None] = Field(default_factory=dict)


class SpecItem(BaseModel):
    """Одна характеристика: значение из входа или пометка «данных нет»."""

    name: str
    value: str


class Gap(BaseModel):
    """Пометка для контент-менеджера.

    kind="missing" — поле входа пустое, его нужно уточнить перед публикацией.
    kind="removed" — фраза модели вырезана, потому что её нет во входных данных.
    """

    field: str
    note: str
    kind: str = "missing"


class ProductCard(BaseModel):
    """Черновик карточки и тот же текст в Markdown для копирования."""

    seo_title: str
    seo_description: str
    specs: list[SpecItem]
    search_queries: list[str]
    commercial_block: str
    gaps: list[Gap]
    markdown: str


def parse_product(payload: object) -> ProductInput:
    """Проверяет JSON товара и приводит характеристики к строкам.

    Args:
        payload: Уже разобранный JSON. Это не строка и не байты.

    Returns:
        Товар с пустыми полями, если их не передали. Пустота не считается
        битым JSON: её позже видно в блоке «Что уточнить».

    Raises:
        CardError: Тело не объект, нет specs или тип поля не подходит.
    """
    if not isinstance(payload, dict):
        raise CardError(
            "Ожидался JSON-объект с полями name, manufacturer, specs и description."
        )
    if "specs" not in payload:
        raise CardError("В объекте нет поля specs. Нужен объект характеристик.")
    raw_specs = payload["specs"]
    if not isinstance(raw_specs, dict):
        raise CardError(
            "Поле specs должно быть объектом «название: значение», а не списком или строкой."
        )

    specs: dict[str, str | None] = {}
    for key, value in raw_specs.items():
        if not isinstance(key, str) or not key.strip():
            raise CardError("Название характеристики должно быть непустой строкой.")
        specs[key.strip()] = _spec_value(key, value)

    return ProductInput(
        name=_text_field(payload, "name"),
        manufacturer=_text_field(payload, "manufacturer"),
        description=_text_field(payload, "description"),
        specs=specs,
    )


def render_markdown(card: ProductCard) -> str:
    """Собирает текст карточки, который контент-менеджер копирует на сайт.

    Args:
        card: Уже проверенная карточка. Поле markdown при сборке ещё пустое.

    Returns:
        Markdown с заголовком, описанием, характеристиками, запросами и
        блоком для менеджера. Служебные пометки сюда не входят, чтобы их не
        вставили на сайт вместе с карточкой: см. render_notes_markdown.
    """
    if card.specs:
        spec_lines = "\n".join(f"- **{item.name}:** {item.value}" for item in card.specs)
    else:
        spec_lines = f"- {MISSING}"
    if card.search_queries:
        query_lines = "\n".join(
            f"{index}. {query}" for index, query in enumerate(card.search_queries, start=1)
        )
    else:
        query_lines = f"- {MISSING}"
    return (
        f"## SEO-заголовок\n{card.seo_title}\n\n"
        f"## SEO-описание\n{card.seo_description}\n\n"
        f"## Характеристики\n{spec_lines}\n\n"
        f"## Поисковые запросы\n{query_lines}\n\n"
        f"## Блок для менеджера\n{card.commercial_block}\n"
    )


def render_notes_markdown(card: ProductCard) -> str:
    """Собирает служебные пометки для контент-менеджера отдельно от карточки.

    Args:
        card: Готовая карточка.

    Returns:
        Markdown: «Что уточнить» и, если код что-то вырезал, «Удалено из текста».
        На странице это отдельные блоки, в примерах — раздел после карточки.
    """
    missing = _gap_lines(
        [gap for gap in card.gaps if gap.kind != "removed"],
        "Все поля заполнены.",
    )
    notes = f"## Что уточнить перед публикацией\n{missing}\n"
    removed = [gap for gap in card.gaps if gap.kind == "removed"]
    if removed:
        lines = "\n".join("- " + gap.note.replace("\n", "  \n  ") for gap in removed)
        notes += f"\n## Удалено из текста — этого нет в ваших данных\n{REMOVED_INTRO}\n\n{lines}\n"
    return notes


def _gap_lines(gaps: list[Gap], empty: str) -> str:
    """Собирает маркированный список одного вида пометок.

    Args:
        gaps: Пропуски входа или снятые фразы модели.
        empty: Строка, если список пуст.

    Returns:
        Markdown-список.
    """
    if not gaps:
        return f"- {empty}"
    return "\n".join(f"- **{gap.field}:** {gap.note}" for gap in gaps)


def _text_field(payload: dict[object, object], field: str) -> str:
    """Возвращает строковое поле или пустую строку, если его нет.

    Args:
        payload: Объект товара.
        field: Имя поля name, manufacturer или description.

    Returns:
        Текст поля. None и отсутствие ключа становятся пустой строкой.

    Raises:
        CardError: Поле передано не строкой.
    """
    if field not in payload or payload[field] is None:
        return ""
    value = payload[field]
    if not isinstance(value, str):
        raise CardError(f"Поле {field} должно быть строкой.")
    return value


def _spec_value(key: str, value: object) -> str | None:
    """Приводит значение характеристики к строке.

    Args:
        key: Название характеристики, попадает в текст ошибки.
        value: Строка, число или null из JSON.

    Returns:
        None для null, иначе строка. Пробелы здесь не считаются пустотой.

    Raises:
        CardError: Значение — список, объект или bool.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise CardError(
            f"Характеристика «{key}» должна быть строкой, числом или null."
        )
    if isinstance(value, str):
        return value
    return str(value)

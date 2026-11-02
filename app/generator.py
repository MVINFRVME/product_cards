"""Сборка карточки: факты из кода, текст модели, одна правка, жёсткая сверка."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.domain import MISSING, CardError, Gap, ProductCard, ProductInput, SpecItem, render_markdown
from app.facts import FactSet, build_facts, remove_fragments
from app.llm import LLMClient
from app.prompts import initial_messages, repair_messages

TITLE_MAX = 70
DESCRIPTION_MIN = 800
DESCRIPTION_MAX = 1200

_FILLER = "Карточка будет дополняться по мере поступления сведений от поставщика."
_DANGLING_TITLE = re.compile(
    r"(?:,\s*|\s+)(?:производитель|диапазон|класс|присоединение)\s*$",
    re.IGNORECASE,
)
_FLUFF_PREFIXES = (
    "описан",
    "подтвержд",
    "позволя",
    "явля",
    "единствен",
    "параметр",
    "разрешен",
    "техническ",
    "указан",
    "факт",
    "данн",
    "назван",
    "компани",
    "произв",
    "составля",
    "отраж",
    "определ",
    "габарит",
    "характеристик",
    "указ",
    "для",
    "что",
    "это",
)


def _is_fluff(token: str) -> bool:
    """Отличает связки и слова оформления от слов самой характеристики."""
    return token in _FLUFF or any(token.startswith(prefix) for prefix in _FLUFF_PREFIXES)


_FLUFF = frozenset(
    word.replace("ё", "е")
    for word in """
    где также прибор прибора приборе составляет позволяет использовать
    системы системе контроль контроля обеспечивает удобство установки установка
    имеет указанные характеристика характеристики описание указывает разработан
    специально является может можно данный данном применения применение условий
    условие компактность компактности надежность надежности
    """.split()
)


@dataclass(frozen=True)
class Draft:
    """Текстовые поля, которые разрешено писать модели."""

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial_block: str


class CardGenerator:
    """Оркестратор черновика. Не знает, Groq это или подмена в тесте."""

    def __init__(self, client: LLMClient) -> None:
        """Принимает клиент модели. Groq подставляется только на границе HTTP.

        Args:
            client: Объект с методом complete.
        """
        self._client = client

    def generate(self, product: ProductInput) -> ProductCard:
        """Собирает карточку по товару.

        Args:
            product: Проверенный вход. Характеристики уже не содержат чужих типов.

        Returns:
            Карточка, в которой технические значения либо из входа, либо «данных нет».

        Raises:
            CardError: Модель оба раза не вернула JSON или клиент LLM отказал.
        """
        facts = build_facts(product)
        raw = self._client.complete(initial_messages(facts))
        draft = parse_draft(raw)
        if draft is not None and draft_problems(draft, facts):
            repaired = parse_draft(self._repair(facts, raw, draft_problems(draft, facts)))
            if repaired is not None:
                draft = repaired
        if draft is None:
            draft = parse_draft(
                self._repair(
                    facts,
                    raw,
                    [
                        "Ответ не является JSON-объектом с полями seo_title, "
                        "seo_description, search_queries и commercial_block."
                    ],
                )
            )
        if draft is None:
            raise CardError(
                "Модель не вернула JSON, из которого можно собрать карточку. Черновик не отдан.",
                502,
            )
        return assemble(draft, facts)

    def _repair(self, facts: FactSet, previous: str, problems: list[str]) -> str:
        """Просит один новый JSON. Пустой ответ модели не отменяет первый черновик.

        Args:
            facts: Разрешённые факты.
            previous: Сырой предыдущий ответ.
            problems: Что исправить.

        Returns:
            Текст повторного ответа или пустая строка, если Groq ничего не прислал.
        """
        try:
            return self._client.complete(repair_messages(facts, previous, problems))
        except CardError:
            return ""


def parse_draft(raw: str) -> Draft | None:
    """Достаёт текстовые поля из ответа модели.

    Args:
        raw: Сырой текст, возможно внутри markdown-ограждения.

    Returns:
        Черновик или None, если объекта JSON нет.
    """
    payload = _load_object(raw)
    if payload is None:
        return None
    return Draft(
        seo_title=_as_text(payload.get("seo_title")),
        seo_description=_as_text(payload.get("seo_description")),
        search_queries=_as_queries(payload.get("search_queries")),
        commercial_block=_as_text(payload.get("commercial_block")),
    )


def draft_problems(draft: Draft, facts: FactSet) -> list[str]:
    """Собирает замечания для одного повторного запроса.

    Args:
        draft: Разобранный ответ модели.
        facts: Разрешённые факты.

    Returns:
        Пустой список, если длину, запросы и значения можно отдавать дальше.
        Оставшиеся замечания после повтора всё равно закрывает assemble.
    """
    problems: list[str] = []
    fields = {
        "seo_title": draft.seo_title,
        "seo_description": draft.seo_description,
        "commercial_block": draft.commercial_block,
    }
    for name, text in fields.items():
        invented = facts.invented_fragments(text)
        if invented:
            shown = ", ".join(f"«{item}»" for item in invented)
            problems.append(
                f"В поле {name} есть значения не из входных данных: {shown}. "
                "Удалите их и не заменяйте другими характеристиками."
            )
    for query in draft.search_queries:
        invented = facts.invented_fragments(query)
        if invented:
            problems.append(
                f"Поисковый запрос «{query}» содержит значение не из входа. "
                "Уберите это значение, не подставляя другое."
            )
    if not draft.seo_title.strip():
        problems.append("seo_title пустой.")
    elif len(draft.seo_title) > TITLE_MAX:
        problems.append(f"seo_title сейчас {len(draft.seo_title)} символов, нужно не больше {TITLE_MAX}.")
    length = len(draft.seo_description)
    if length < DESCRIPTION_MIN or length > DESCRIPTION_MAX:
        problems.append(
            f"seo_description сейчас {length} символов, нужно от {DESCRIPTION_MIN} до {DESCRIPTION_MAX}. "
            f"Расширяйте только разрешёнными фактами и пометками «{MISSING}»."
        )
    if not 3 <= len(draft.search_queries) <= 5:
        problems.append(f"search_queries сейчас {len(draft.search_queries)}, нужно от 3 до 5.")
    return problems


def assemble(draft: Draft, facts: FactSet) -> ProductCard:
    """Подставляет характеристики из входа и вырезает чужие значения.

    Args:
        draft: Текст модели после не более чем одного повтора.
        facts: Характеристики и пропуски, собранные кодом.

    Returns:
        Карточка для API и для копирования.
    """
    removed: list[str] = []
    title = clip(" ".join(_clean_prose(draft.seo_title, facts, removed).split()), TITLE_MAX)
    name = facts.product.name.strip()
    title_lost_fact = bool(removed) and name and name.casefold() not in title.casefold()
    if not title or title_lost_fact or "данных нет" in title.casefold():
        title = _fallback_title(facts)
    title = title.rstrip(" ,;:—") or _fallback_title(facts)
    title = _DANGLING_TITLE.sub("", title).rstrip(" ,;:—") or _fallback_title(facts)

    model_text = _clean_prose(draft.seo_description, facts, removed)
    leftovers = facts.invented_fragments(model_text)
    if leftovers:
        removed.extend(leftovers)
        model_text = remove_fragments(model_text, leftovers)
    description = fit_description(model_text, facts)
    commercial = _clean_prose(draft.commercial_block, facts, removed)
    if not _commercial_usable(commercial) or _echoes_description(commercial, facts):
        commercial = _fallback_commercial(facts)
    if commercial and commercial[-1] not in ".!?…":
        commercial += "."

    queries = _queries(draft.search_queries, facts, removed)
    gaps = _unique_gaps([*facts.gaps, *_removed_gaps(removed, facts)])
    if not 3 <= len(queries) <= 5:
        gaps = _unique_gaps(
            [
                *gaps,
                Gap(field="Поисковые запросы", note=MISSING),
            ]
        )
    card = ProductCard(
        seo_title=title,
        seo_description=description,
        specs=facts.specs,
        search_queries=queries,
        commercial_block=commercial,
        gaps=gaps,
        markdown="",
    )
    return card.model_copy(update={"markdown": render_markdown(card)})


def _unique_sentences(text: str) -> str:
    """Убирает одинаковые и близкие предложения, которыми модель добирает объём.

    Args:
        text: Описание после вырезания чужих значений.

    Returns:
        Текст без повторного пересказа одной и той же характеристики.
    """
    seen: list[str] = []
    exact: set[str] = set()
    for sentence in _split_sentences(text):
        normalized = sentence.casefold().rstrip(".!?… ")
        if normalized in exact or any(_same_claim(sentence, kept) for kept in seen):
            continue
        exact.add(normalized)
        seen.append(sentence)
    return " ".join(seen)


def _split_sentences(text: str) -> list[str]:
    """Делит текст на предложения и схлопывает пробелы внутри них."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [" ".join(part.split()) for part in parts if part.strip()]


def _same_claim(left: str, right: str) -> bool:
    """Считает два предложения одним утверждением.

    Args:
        left: Кандидат, который ещё не принят в текст.
        right: Предложение, которое уже оставлено.

    Returns:
        True, если кандидат не добавляет нового факта или лишь разворачивает те же числа.
    """
    candidate_key = set(_claim_key(left))
    existing_key = set(_claim_key(right))
    numbers = _numbers(left)
    existing_numbers = _numbers(right)
    if numbers and numbers == existing_numbers and len(existing_key) >= 3 and existing_key <= candidate_key:
        return True
    if not set(numbers) <= set(existing_numbers):
        return False
    return len(candidate_key) >= 3 and candidate_key <= existing_key


def _numbers(sentence: str) -> tuple[str, ...]:
    """Возвращает числа предложения в одном порядке для сравнения."""
    folded = sentence.casefold().replace("ё", "е")
    found = re.findall(r"\d+(?:[.,]\d+)?", folded)
    if re.search(r"\bнул[яю]\b|\bнол[яю]\b", folded):
        found.append("0")
    return tuple(sorted(found))


def _claim_key(sentence: str) -> tuple[str, ...]:
    """Оставляет числа и слова, которые отличают одну характеристику от другой."""
    folded = sentence.casefold().replace("ё", "е")
    numbers = re.findall(r"\d+(?:[.,]\d+)?", folded)
    folded = folded.replace("–", " ").replace("—", " ").replace("-", " ").replace("…", " ")
    folded = re.sub(r"[^\w\s]", " ", folded, flags=re.UNICODE)
    words: list[str] = []
    for token in folded.split():
        if token in {"нуля", "нулю", "ноля", "нолю"}:
            numbers.append("0")
            continue
        token = _word_stem(token)
        if _is_fluff(token):
            continue
        short = len(token) <= 2
        has_digit = any(char.isdigit() for char in token)
        if short and not has_digit and token not in {"мм", "см"}:
            continue
        if token.isdigit():
            continue
        words.append(token)
    return tuple(sorted({*words, *numbers}))


def _word_stem(token: str) -> str:
    """Снимает окончание, чтобы «измерений» и «измерения» совпали."""
    synonyms = {
        "мегапаскал": "мпа",
        "мегапаскаль": "мпа",
        "килопаскал": "кпа",
        "килопаскаль": "кпа",
        "миллиметр": "мм",
        "сантиметр": "см",
    }
    token = synonyms.get(token, token)
    for suffix in (
        "ения",
        "ение",
        "ений",
        "ются",
        "ится",
        "ает",
        "яет",
        "иями",
        "ами",
        "ями",
        "ого",
        "ему",
        "ыми",
        "ими",
        "иях",
        "ах",
        "ях",
        "ов",
        "ев",
        "ий",
        "ый",
        "ой",
        "ая",
        "ое",
        "ые",
        "ие",
        "ам",
        "ям",
        "ом",
        "ем",
        "ую",
        "юю",
        "ей",
        "ия",
        "а",
        "я",
        "ы",
        "и",
        "е",
        "у",
        "ю",
        "о",
        "ен",
    ):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if token.startswith("пред") and len(token) - 4 >= 4:
        token = token[4:]
    return synonyms.get(token, token)


def _commercial_usable(text: str) -> bool:
    """Отличает нормальный блок для менеджера от короткой заглушки модели.

    Args:
        text: Commercial-блок после вырезания чужих значений.

    Returns:
        False, если текста нет или в нём осталась только пометка без смысла.
    """
    folded = text.casefold().strip(" .")
    return bool(folded) and folded not in {"по назначению данных нет", "данных нет"}


def fit_description(text: str, facts: FactSet) -> str:
    """Доводит описание до 800–1200 символов только готовыми фразами.

    Args:
        text: Описание модели после вырезания чужих значений.
        facts: Входные факты, из которых собираются добавочные фразы.

    Returns:
        Текст в нужном диапазоне. Если модель не упомянула известное значение,
        оно добавляется фразой из входа. Добавочные фразы идут одним абзацем
        после текста модели. Новых характеристик здесь нет.
    """
    base = clip(_drop_restatements(_unique_sentences(text.strip()), facts), DESCRIPTION_MAX, on_sentence=True)
    added: list[str] = []

    def current() -> str:
        extra = " ".join(added)
        return f"{base}\n\n{extra}" if base and extra else base or extra

    def add(sentence: str, strict: bool = True) -> None:
        before = current()
        after = _append_sentence(before, sentence, strict=strict, separator=" ")
        if after != before:
            added.append(after[len(before):].strip())

    compact_base = re.sub(r"\s+", "", base.casefold())
    for spec in facts.specs:
        if spec.value != MISSING and re.sub(r"\s+", "", spec.value.casefold()) not in compact_base:
            add(_spec_sentence(spec))
    def values_already_said(sentence: str) -> bool:
        text = current().casefold()
        compact_text = re.sub(r"\s+", "", text)
        values = [spec.value for spec in facts.specs if spec.value != MISSING and spec.value in sentence]
        if values:
            return all(re.sub(r"\s+", "", value.casefold()) in compact_text for value in values)
        if MISSING in sentence:
            names = [spec.name for spec in facts.specs if spec.value == MISSING and spec.name in sentence]
            return bool(names) and all(name.casefold()[:6] in text for name in names)
        return False

    for strict in (True, False):
        for sentence in _factual_sentences(facts):
            if len(current()) >= DESCRIPTION_MIN:
                break
            if strict and values_already_said(sentence):
                continue
            add(sentence, strict=strict)
    if len(current()) < DESCRIPTION_MIN:
        add(_FILLER)
    return clip(current(), DESCRIPTION_MAX, on_sentence=True)


_RESTATE_FILLER = frozenset(
    "данных нет и а с в по для имеет имеют составляет указан указана указано указаны не также".split()
)


def _drop_restatements(text: str, facts: FactSet) -> str:
    """Убирает предложения, которые только повторяют уже названные пропуски и значения.

    Args:
        text: Описание модели без точных повторов.
        facts: Входные факты: название, характеристики и пропуски.

    Returns:
        Текст, где «производитель — данных нет» и подобное сказано один раз.
        Предложение остаётся, если в нём есть хоть одно новое слово или факт.
    """
    vocabulary = set(_words(facts.product.name))
    for spec in facts.specs:
        vocabulary.update(_words(spec.name))
        if spec.value != MISSING:
            vocabulary.update(_words(spec.value))
    for gap in facts.gaps:
        vocabulary.update(_words(gap.field))
    vocabulary |= _RESTATE_FILLER
    said: set[str] = set()
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        kept: list[str] = []
        for sentence in _split_sentences(paragraph):
            words = _words(sentence)
            facts_here = {word for word in words if word not in _RESTATE_FILLER}
            only_known = bool(words) and all(word in vocabulary for word in words)
            if only_known and facts_here <= said:
                continue
            said |= facts_here
            kept.append(sentence)
        if kept:
            paragraphs.append(" ".join(kept))
    return "\n\n".join(paragraphs)


def _words(text: str) -> list[str]:
    """Основы слов и числа в нижнем регистре, без знаков препинания.

    Длинные слова обрезаются до шести букв, чтобы «производителе» и
    «производитель» считались одним словом.
    """
    raw = re.findall(r"[\w,]+", text.casefold().replace("ё", "е").replace(", ", " "))
    return [word[:6] if len(word) > 6 and word.isalpha() else word for word in raw]


def _spec_sentence(spec: SpecItem) -> str:
    """Фраза покупателю с одной характеристикой ровно в том виде, как во входе."""
    return f"{spec.name[:1].upper()}{spec.name[1:]}: {spec.value}."


def clip(text: str, limit: int, on_sentence: bool = False) -> str:
    """Обрезает текст по границе слова, не добавляя новых слов.

    Args:
        text: Исходная строка.
        limit: Максимум символов.
        on_sentence: Для описания резать по последней точке, а не посреди фразы.

    Returns:
        Строка не длиннее limit.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    if on_sentence:
        ends = [cut.rfind(mark) for mark in ".!?…"]
        end = max(ends)
        if end >= int(limit * 0.6):
            return cut[: end + 1].rstrip()
    space = cut.rfind(" ")
    if space >= int(limit * 0.6):
        cut = cut[:space]
    return cut.rstrip()


def _clean_prose(text: str, facts: FactSet, removed: list[str]) -> str:
    """Вырезает чужие значения и запоминает их для блока «Удалено из текста».

    Args:
        text: Поле ответа модели.
        facts: Разрешённые факты.
        removed: Общий список отброшенных фрагментов, пополняется на месте.

    Returns:
        Текст без этих фрагментов.
    """
    fragments = facts.invented_fragments(text)
    removed.extend(fragments)
    cleaned = remove_fragments(text, fragments)
    if facts.product.description.strip():
        cleaned = _FALSE_NO_PURPOSE.sub("", cleaned).strip()
    return cleaned


_FALSE_NO_PURPOSE = re.compile(r"\s*По назначению данных нет\.?", re.IGNORECASE)


def _queries(raw_queries: list[str], facts: FactSet, removed: list[str]) -> list[str]:
    """Оставляет до пяти запросов без чужих значений и добирает их из входа.

    Args:
        raw_queries: Запросы модели.
        facts: Разрешённые факты.
        removed: Сюда попадают отброшенные фрагменты.

    Returns:
        От нуля до пяти строк. Меньше трёх — повод для пометки «данных нет».
    """
    result: list[str] = []
    for query in [*raw_queries, *_fallback_queries(facts)]:
        normalized = " ".join(query.split())
        if not normalized or normalized in result:
            continue
        invented = facts.invented_fragments(normalized)
        if invented or "данных нет" in normalized.casefold():
            removed.extend(invented)
            continue
        result.append(normalized)
        if len(result) == 5:
            break
    return result


def _fallback_queries(facts: FactSet) -> list[str]:
    """Собирает запасные запросы только из названия, производителя и известных характеристик.

    Args:
        facts: Входные факты.

    Returns:
        Запросы без новых технических значений.
    """
    product = facts.product
    name = product.name.strip()
    manufacturer = product.manufacturer.strip()
    queries: list[str] = []
    if name and manufacturer:
        queries.append(f"{name} {manufacturer}")
    if name:
        queries.append(name)
        queries.append(f"купить {name}")
    for spec in facts.specs:
        if not name or spec.value == MISSING:
            continue
        queries.append(f"{name} {spec.name} {spec.value}")
    if name:
        queries.extend([f"{name} цена", f"{name} характеристики", f"{name} заказать"])
    return queries


def _append_sentence(text: str, sentence: str, strict: bool = True, separator: str = "\n\n") -> str:
    """Добавляет фразу, если она новая и описание не выйдет за 1200 символов.

    Args:
        text: Уже набранное описание.
        sentence: Фраза только из входных фактов, иногда из двух предложений.
        strict: Если начало фразы уже сказано, не добавлять и её хвост, чтобы
            в тексте не оставалось предложений без контекста.
        separator: Чем отделить фразу: новым абзацем или пробелом в том же абзаце.

    Returns:
        Прежний текст, если фраза не помещается или уже есть. Уже сказанные
        предложения внутри фразы не повторяются.
    """
    sentence = " ".join(sentence.split())
    if not sentence or sentence in text:
        return text
    kept = _split_sentences(text)
    known = {item.casefold().rstrip(".!?… ") for item in kept}

    def is_new(part: str) -> bool:
        return part.casefold().rstrip(".!?… ") not in known and not any(
            _same_claim(part, item) for item in kept
        )

    parts = _split_sentences(sentence)
    if not parts or (strict and not is_new(parts[0])):
        return text
    fresh = [part for part in parts if is_new(part)]
    if not fresh:
        return text
    sentence = " ".join(fresh)
    addition = f"{separator}{sentence}" if text else sentence
    if len(text) + len(addition) > DESCRIPTION_MAX:
        return text
    return text + addition


def _factual_sentences(facts: FactSet) -> list[str]:
    """Готовит фразы для добора длины. Они пишутся для покупателя, а не о системе.

    Args:
        facts: Характеристики и пропуски из входа.

    Returns:
        Фразы в порядке, в котором их можно дописывать к описанию. Технические
        значения берутся только из входа, остальное — общие фразы B2B-магазина
        про заказ и консультацию, без характеристик прибора.
    """
    product = facts.product
    name = product.name.strip()
    manufacturer = product.manufacturer.strip()
    description = product.description.strip()
    sentences: list[str] = []
    if name and manufacturer:
        sentences.append(f"{name} выпускает компания {manufacturer}.")
    elif name:
        sentences.append(f"{name}. Производитель: {MISSING}, уточняйте у менеджера.")
    elif manufacturer:
        sentences.append(f"Производитель прибора — {manufacturer}.")

    if description:
        if description[-1] not in ".!?…":
            description += "."
        sentences.append(f"Назначение: {description}")
    else:
        sentences.append(f"Назначение прибора: {MISSING}, уточняйте у менеджера.")

    known = [item for item in facts.specs if item.value != MISSING]
    missing = [item for item in facts.specs if item.value == MISSING]
    if known:
        joined = "; ".join(f"{item.name} — {item.value}" for item in known)
        sentences.append(f"Основные характеристики: {joined}.")
    sentences.extend(_spec_sentence(item) for item in known)
    if missing:
        names = ", ".join(f"«{item.name}»" for item in missing)
        sentences.append(f"По характеристикам {names} {MISSING}, их можно уточнить у менеджера перед заказом.")
    for item in missing:
        sentences.append(f"Значение «{item.name}» в карточке пока не указано: {MISSING}.")
    sentences.extend(
        [
            "Чтобы подобрать прибор под вашу задачу, обратитесь к менеджеру: "
            "он подскажет по наличию, срокам поставки и комплектации.",
            "Для оформления заказа укажите нужное количество и реквизиты организации.",
            "Если в карточке не хватает сведений для выбора, менеджер запросит их у поставщика.",
            "Перед заказом сверьте характеристики прибора с требованиями вашего проекта "
            "или технического задания.",
            "Стоимость и условия поставки менеджер подскажет по запросу, "
            "вместе с коммерческим предложением можно запросить документацию на прибор.",
            "Если нужен прибор с другими параметрами, менеджер предложит подходящие варианты из каталога.",
        ]
    )
    return sentences


def _fallback_title(facts: FactSet) -> str:
    """Собирает заголовок из названия и первой известной характеристики.

    Args:
        facts: Входные факты.

    Returns:
        Строка не длиннее 70 символов без «данных нет».
    """
    name = facts.product.name.strip() or "Карточка товара"
    manufacturer = facts.product.manufacturer.strip()
    base = name
    if manufacturer:
        with_maker = f"{name} {manufacturer}"
        if len(with_maker) <= TITLE_MAX:
            base = with_maker
    for spec in facts.specs:
        if spec.value == MISSING:
            continue
        for candidate in (
            f"{base}, {spec.name} {spec.value}",
            f"{base}, {spec.value}",
            f"{base} {spec.value}",
        ):
            if len(candidate) <= TITLE_MAX:
                return candidate
    return clip(base, TITLE_MAX)


def _fallback_commercial(facts: FactSet) -> str:
    """Собирает commercial-блок, если модель его не дала, он весь вырезан или повторяет описание.

    Args:
        facts: Входные факты.

    Returns:
        «Подходит для задач …» по назначению из описания и характеристики, важные
        при подборе; если назначения нет — прямая пометка «данных нет».
    """
    known = [spec for spec in facts.specs if spec.value != MISSING]
    selection = ""
    if known:
        listed = ", ".join(f"{spec.name} {spec.value}" for spec in known[:4])
        selection = f" При подборе важны: {listed}."
    purpose = _purpose(facts.product.description)
    if purpose:
        return f"Подходит для задач {purpose}.{selection}"
    name = facts.product.name.strip() or "прибор"
    return (
        f"Кому подходит {name} и какие задачи он решает: {MISSING}. "
        f"Назначение не указано — уточните его у поставщика, прежде чем предлагать прибор клиенту.{selection}"
    )


def _purpose(description: str) -> str:
    """Достаёт из описания назначение: «измерения избыточного давления».

    Args:
        description: Описание или фрагмент паспорта из входа.

    Returns:
        Назначение в родительном падеже, без многоточия и точки, или пустая
        строка, если в описании нет оборота «предназначен для».
    """
    match = re.search(r"предназначен\w*\s+для\s+([^.!?…]+)", description, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip(" .,")


def _echoes_description(text: str, facts: FactSet) -> bool:
    """Проверяет, не повторила ли модель описание из входа вместо блока для менеджера."""
    description = facts.product.description
    if not description.strip():
        return False

    def compact(value: str) -> str:
        return re.sub(r"[\W_]+", "", value.casefold())

    return compact(text) in compact(description) or compact(description) in compact(text)


def _load_object(raw: str) -> dict[str, object] | None:
    """Достаёт JSON-объект из ответа, даже если он обёрнут в markdown.

    Args:
        raw: Сырой текст модели.

    Returns:
        Словарь или None, если объекта нет.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _as_text(value: object) -> str:
    """Оставляет строку и отбрасывает null и числа. Пустоту потом заполнит код."""
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_queries(value: object) -> list[str]:
    """Оставляет только непустые строки. Одиночная строка считается одним запросом."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    queries: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            queries.append(item.strip())
    return queries


_INVENTED_MISSING = re.compile(
    r"^[\s\-•\d.)]*([^:]{2,60}?):\s*данных нет\.?\s*$",
    re.IGNORECASE,
)


def _removed_gaps(fragments: list[str], facts: FactSet | None = None) -> list[Gap]:
    """Собирает пометки об отброшенном тексте модели.

    Args:
        fragments: Куски, которые вырезали из ответа.
        facts: Входные факты. Нужны, чтобы назвать слово, из-за которого снято предложение.

    Returns:
        Список «Текст модели». Однотипные «Параметр: данных нет» схлопываются
        в одну строку, чтобы не засорять экран после удаления поля во входе.
    """
    invented_names: list[str] = []
    other: list[str] = []
    for fragment in _unique(fragments):
        label = _invented_missing_label(fragment)
        if label is not None:
            invented_names.append(label)
            continue
        other.append(fragment)

    gaps: list[Gap] = []
    names = _unique(invented_names)
    if names:
        gaps.append(
            Gap(
                field="Текст модели",
                note=(
                    f"Убрано: «{', '.join(names)}: данных нет».\n"
                    "Почему: модель сама перечислила эти характеристики, хотя во входных данных "
                    "их нет вообще. Это не пропуски ваших данных, а лишний текст."
                ),
                kind="removed",
            )
        )
    for fragment in other:
        topic = facts.trigger_topic(fragment) if facts else None
        if topic:
            note = (
                f"Убрано: «{_short_fragment(fragment)}»\n"
                f"Почему: здесь утверждение про {topic}, а во входных данных об этом ничего нет."
            )
        else:
            note = (
                f"Убрано: «{_short_fragment(fragment)}»\n"
                "Почему: такого значения нет во входных данных — модель его придумала "
                "или взяла из общих знаний о похожих приборах."
            )
        gaps.append(Gap(field="Текст модели", note=note, kind="removed"))
    return gaps


def _invented_missing_label(fragment: str) -> str | None:
    """Если фрагмент — строка «чужая характеристика: данных нет», возвращает имя.

    Args:
        fragment: Вырезанный кусок ответа модели.

    Returns:
        Название характеристики или None, если это не такой шаблон.
    """
    match = _INVENTED_MISSING.match(" ".join(fragment.split()))
    if match is None:
        return None
    return match.group(1).strip(" -•")


def _short_fragment(fragment: str) -> str:
    """Оставляет в пометке вырезанное предложение, но не абзац ответа модели.

    Args:
        fragment: Кусок, который вырезали из текста.

    Returns:
        Фрагмент не длиннее 300 символов: обычно предложение целиком, чтобы
        контент-менеджер мог решить, вернуть ли его вручную.
    """
    compact = " ".join(fragment.split())
    if len(compact) <= 300:
        return compact
    return compact[:299].rstrip() + "…"


def _unique(items: list[str]) -> list[str]:
    """Убирает повторы строк, сохраняя порядок."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _unique_gaps(gaps: list[Gap]) -> list[Gap]:
    """Убирает одинаковые пометки: пропуски входа и снятые фразы модели."""
    seen: set[tuple[str, str]] = set()
    result: list[Gap] = []
    for gap in gaps:
        key = (gap.field, gap.note)
        if key in seen:
            continue
        seen.add(key)
        result.append(gap)
    return result

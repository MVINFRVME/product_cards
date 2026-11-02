"""Список характеристик из входа и поиск значений, которых во входе не было."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain import MISSING, Gap, ProductInput, SpecItem

_UNIT = r"мпа|кпа|мкм|мм|см|ма|кгц|гц|°c|℃|°с|бар|па|м|в|а|%"
_NUMBER = r"-?\d+(?:[.,]\d+)?"
_RANGE = re.compile(
    rf"(?<![\w.,])({_NUMBER})\s*[-–—…]{{1,3}}\s*({_NUMBER})\s*({_UNIT})(?![\w])",
    re.IGNORECASE,
)
_SINGLE = re.compile(
    rf"(?<![\w.,])({_NUMBER})\s*({_UNIT})(?![\w])",
    re.IGNORECASE,
)
_PHRASE = re.compile(
    r"g\s*\d+\s*/\s*\d+|ip\s*\d{2}|гост(?:\s*р)?\s*\d+(?:[.\-]\d+)*",
    re.IGNORECASE,
)
_CLASS = re.compile(
    r"класс(?:\s+точности|\s+допуска)?\s+(-?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_STEMS = (
    "взрывозащищ",
    "взрывозащит",
    "пищев",
    "нержав",
    "пылевлаго",
    "гост",
    "безопасн",
    "нефте",
    "гидравл",
    "монтаж",
    "технологическ",
    "предназначен",
    "газ",
    "температур",
    "питан",
    "защит",
    "материал",
    "интерфейс",
    "погрешн",
    "отраслев",
    "промышлен",
    "норматив",
)
_TOPICS = {
    "взрывозащищ": "взрывозащиту",
    "взрывозащит": "взрывозащиту",
    "пищев": "пищевое исполнение",
    "нержав": "нержавеющую сталь",
    "пылевлаго": "пылевлагозащиту",
    "гост": "соответствие ГОСТ",
    "безопасн": "безопасность",
    "нефте": "нефтегазовую отрасль",
    "гидравл": "гидравлику",
    "монтаж": "монтаж",
    "технологическ": "технологические процессы",
    "предназначен": "назначение прибора",
    "газ": "газ",
    "температур": "температуру",
    "питан": "питание",
    "защит": "защиту",
    "материал": "материал",
    "интерфейс": "интерфейс",
    "погрешн": "погрешность",
    "отраслев": "отрасль применения",
    "промышлен": "промышленное применение",
    "норматив": "нормативы",
}
_THREAD_WORDS = (
    ("четвертьдюйм", "g1/4"),
    ("полудюйм", "g1/2"),
    ("трехчетверт", "g3/4"),
    ("трёхчетверт", "g3/4"),
)
_SHORT_UNITS = frozenset({"м", "в", "а"})
_NEGATION = re.compile(
    r"без указания|не указан\w*|отсутству\w*|не привед\w*|нет сведений",
    re.IGNORECASE,
)
_OTHERS = re.compile(r"остальн|друг(?:ие|их|им)|кроме|помимо", re.IGNORECASE)
_NUMBER_ONLY = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class FactSet:
    """Факты, которые разрешено пересказывать, и пропуски во входе."""

    product: ProductInput
    specs: list[SpecItem]
    gaps: list[Gap]
    source_text: str

    def trigger_topic(self, fragment: str) -> str | None:
        """Называет тему, из-за которой предложение сняли целиком.

        Args:
            fragment: Вырезанное предложение.

        Returns:
            Тема вроде «монтаж» или «промышленное применение», о которой во
            входе ничего нет, или None, если предложение снято не из-за темы.
        """
        source = self.source_text.lower()
        for stem in _STEMS:
            if _has_word_start(source, stem):
                continue
            if re.search(rf"(?<!\w){re.escape(stem)}", fragment, re.IGNORECASE):
                return _TOPICS.get(stem, stem)
        return None

    def invented_fragments(self, text: str) -> list[str]:
        """Возвращает фрагменты текста, которых нет во входных данных.

        Args:
            text: Кусок ответа модели: заголовок, описание, запрос или commercial-блок.

        Returns:
            Фрагменты в порядке появления, без повторов. Пустой список значит,
            что новых чисел, резьб, классов и запрещённых формулировок не видно.
        """
        return find_invented_fragments(text, self.source_text, self.specs)


def build_facts(product: ProductInput) -> FactSet:
    """Собирает характеристики и пропуски строго из полей товара.

    Args:
        product: Проверенный вход. Пустые строки здесь уже допустимы.

    Returns:
        Характеристики со значением или пометкой «данных нет», пропуски и
        исходный текст, с которым потом сверяется ответ модели.
    """
    gaps: list[Gap] = []
    for label, value in (
        ("Название", product.name),
        ("Производитель", product.manufacturer),
        ("Описание", product.description),
    ):
        if not value.strip():
            gaps.append(Gap(field=label, note=MISSING))

    specs: list[SpecItem] = []
    for key, raw in product.specs.items():
        if raw is None or not raw.strip():
            specs.append(SpecItem(name=key, value=MISSING))
            gaps.append(Gap(field=key, note=MISSING))
        else:
            specs.append(SpecItem(name=key, value=raw.strip()))
    if not product.specs:
        gaps.append(Gap(field="Характеристики", note=MISSING))

    lines = [
        product.name.strip(),
        product.manufacturer.strip(),
        product.description.strip(),
    ]
    # Имена всех характеристик из входа разрешены в тексте, даже если значения нет.
    # Иначе фраза «питание: данных нет» сама себе запрещает основу «питан».
    lines.extend(item.name for item in specs)
    lines.extend(
        f"{item.name}: {item.value}" for item in specs if item.value != MISSING
    )
    return FactSet(
        product=product,
        specs=specs,
        gaps=gaps,
        source_text="\n".join(line for line in lines if line),
    )


def find_invented_fragments(
    text: str,
    source_text: str,
    specs: list[SpecItem],
) -> list[str]:
    """Ищет в тексте технические значения, которых нет в источнике.

    Args:
        text: Проверяемый текст.
        source_text: Название, производитель, описание и непустые характеристики.
        specs: Уже нормализованный список характеристик.

    Returns:
        Фрагменты, которые нельзя оставлять в карточке.
    """
    if not text:
        return []
    allowed_pairs = _measurement_pairs(source_text)
    allowed_phrases = {_phrase(match.group(0)) for match in _PHRASE.finditer(source_text)}
    allowed_classes = _class_numbers(source_text, specs)
    source_lower = source_text.lower()
    found: list[str] = []

    for match in _RANGE.finditer(text):
        if _lowercase_short_unit(match.group(3)):
            continue
        unit = _unit(match.group(3))
        if not _pair_allowed(match.group(1), unit, allowed_pairs) or not _pair_allowed(
            match.group(2), unit, allowed_pairs
        ):
            found.append(match.group(0))
    for match in _SINGLE.finditer(text):
        if _lowercase_short_unit(match.group(2)):
            continue
        if not _pair_allowed(match.group(1), _unit(match.group(2)), allowed_pairs):
            found.append(match.group(0))
    for match in _PHRASE.finditer(text):
        if _phrase(match.group(0)) not in allowed_phrases:
            found.append(match.group(0))
    for match in _CLASS.finditer(text):
        if _number(match.group(1)) not in allowed_classes:
            found.append(match.group(0))
    for stem in _STEMS:
        if _has_word_start(text.lower(), stem) and not _has_word_start(source_lower, stem):
            found.extend(_sentences_with(text, stem))
    found.extend(_false_missing(text, specs))
    found.extend(_thread_paraphrases(text, source_text))
    return _unique(found)


def remove_fragments(text: str, fragments: list[str]) -> str:
    """Вырезает запрещённые фрагменты и подчищает пробелы.

    Args:
        text: Исходный кусок ответа.
        fragments: Фрагменты из find_invented_fragments.

    Returns:
        Текст без этих фрагментов. Замена на новое значение не подставляется.
        Если от предложения остался огрызок вроде «Диаметр корпуса.», оно
        убирается целиком.
    """
    ordered = sorted((item for item in _unique(fragments) if item), key=len, reverse=True)
    paragraphs = []
    for paragraph in text.split("\n\n"):
        sentences = []
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
            cut = sentence
            for fragment in ordered:
                cut = re.sub(_loose_pattern(fragment), "", cut, flags=re.IGNORECASE)
            if cut != sentence and _is_stub(cut):
                continue
            sentences.append(cut)
        paragraphs.append(" ".join(item for item in sentences if item.strip()))
    cleaned = "\n\n".join(item for item in paragraphs if item.strip())
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(
        r",\s*а\s+в\s+\S+(?:\s+\S+){0,4}\s+также\s+указано\s*\.",
        ".",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _loose_pattern(fragment: str) -> str:
    """Шаблон фрагмента, где любой пробел совпадает с любым другим, включая неразрывный."""
    return r"\s+".join(re.escape(part) for part in fragment.split())


def _is_stub(sentence: str) -> bool:
    """Отличает огрызок после вырезания значения от нормального предложения.

    Args:
        sentence: Предложение, из которого уже вырезали чужое значение.

    Returns:
        True, если осталось меньше четырёх слов или фраза обрывается на тире,
        двоеточии или предлоге.
    """
    words = re.findall(r"[\w/–-]+", sentence)
    if len(words) < 4:
        return True
    tail = sentence.strip().rstrip(".!?…").rstrip()
    return tail.endswith(("–", "—", "-", ":")) or tail.split()[-1].casefold() in {
        "в", "до", "от", "по", "на", "с", "из", "при", "составляет", "равен", "равна",
    }


def _lowercase_short_unit(raw_unit: str) -> bool:
    """Отличает предлог «в» от единицы «В». Строчная м, в или а единицей не считается.

    Args:
        raw_unit: Единица так, как она написана в тексте.

    Returns:
        True, если это одна строчная буква и её нельзя принимать за вольт, ампер или метр.
    """
    unit = _unit(raw_unit)
    return unit in _SHORT_UNITS and raw_unit[:1].islower()


def _thread_paraphrases(text: str, source_text: str) -> list[str]:
    """Ищет размер резьбы словами, если такой G-резьбы во входе нет.

    Args:
        text: Проверяемый текст.
        source_text: Название, описание и непустые характеристики.

    Returns:
        Предложения с чужим словесным размером. «Полудюймовая» при G1/2 во входе остаётся.
    """
    allowed = {_phrase(match.group(0)) for match in _PHRASE.finditer(source_text)}
    folded = text.casefold()
    found: list[str] = []
    for stem, canonical in _THREAD_WORDS:
        if stem in folded and canonical not in allowed:
            found.extend(_sentences_with(text, stem))
    named = any(stem in folded for stem, _canonical in _THREAD_WORDS)
    if "дюйм" in folded and not named and not any(item.startswith("g") for item in allowed):
        found.extend(_sentences_with(text, "дюйм"))
    return found


def _sentences_with(text: str, stem: str) -> list[str]:
    """Возвращает предложения, в которых есть запрещённая основа.

    Args:
        text: Проверяемый текст.
        stem: Основа слова, которой не было во входе.

    Returns:
        Целые предложения. Вырезать одно слово и оставлять «– данных нет» нельзя.
    """
    return [sentence for sentence in _sentences(text) if _has_word_start(sentence.casefold(), stem)]


def _has_word_start(text: str, stem: str) -> bool:
    """Ищет основу только с начала слова, чтобы «газ» не находился в «магазине»."""
    return re.search(rf"(?<!\w){re.escape(stem)}", text) is not None


def _false_missing(text: str, specs: list[SpecItem]) -> list[str]:
    """Ищет фразы, где модель пишет, что заполненной характеристики нет.

    Args:
        text: Проверяемый текст.
        specs: Характеристики со значениями из входа.

    Returns:
        Предложения, которые противоречат уже переданному значению.
    """
    found: list[str] = []
    for sentence in _sentences(text):
        folded = sentence.casefold()
        denies = "данных нет" in folded or _NEGATION.search(sentence) is not None
        if not denies:
            continue
        others = _OTHERS.search(folded)
        compact_sentence = _compact(sentence)
        for spec in specs:
            if spec.value == MISSING or len(spec.name.strip()) < 5:
                continue
            name_at = folded.find(spec.name.casefold()[:8])
            if name_at == -1 or _compact(spec.value) in compact_sentence:
                continue
            if others is not None and name_at < others.start():
                before = folded[: others.start()]
                own_denial = "данных нет" in before or _NEGATION.search(before) is not None
                if not own_denial:
                    continue
            found.append(sentence)
            break
    return found


def _compact(text: str) -> str:
    """Убирает пробелы, чтобы «-50…200°C» совпал с «-50…200 °C»."""
    return re.sub(r"\s+", "", text.casefold())


def _sentences(text: str) -> list[str]:
    """Делит текст на предложения вместе с завершающим знаком."""
    return [" ".join(part.split()) for part in re.split(r"(?<=[.!?…])\s+|\n+", text) if part.strip()]


def _measurement_pairs(source_text: str) -> set[tuple[str, str]]:
    """Собирает пары «число, единица» из текста входа.

    Args:
        source_text: Название, описание и непустые характеристики.

    Returns:
        Канонические пары. Оба конца диапазона получают одну и ту же единицу.
    """
    pairs: set[tuple[str, str]] = set()
    for match in _RANGE.finditer(source_text):
        if _lowercase_short_unit(match.group(3)):
            continue
        unit = _unit(match.group(3))
        pairs.add((_number(match.group(1)), unit))
        pairs.add((_number(match.group(2)), unit))
    for match in _SINGLE.finditer(source_text):
        if _lowercase_short_unit(match.group(2)):
            continue
        pairs.add((_number(match.group(1)), _unit(match.group(2))))
    return pairs


def _class_numbers(source_text: str, specs: list[SpecItem]) -> set[str]:
    """Собирает числа, которыми разрешено называть класс точности или допуска.

    Args:
        source_text: Текст входа, где класс может быть назван словами.
        specs: Характеристики. Чисто числовое значение тоже считается разрешённым.

    Returns:
        Канонические числа с точкой вместо запятой.
    """
    numbers = {_number(match.group(1)) for match in _CLASS.finditer(source_text)}
    for spec in specs:
        if spec.value == MISSING:
            continue
        if _NUMBER_ONLY.fullmatch(spec.value.strip()):
            numbers.add(_number(spec.value))
    return numbers


def _pair_allowed(number: str, unit: str, allowed: set[tuple[str, str]]) -> bool:
    """Проверяет, что такая пара числа и единицы была во входе."""
    return (_number(number), unit) in allowed


def _number(raw: str) -> str:
    """Приводит число к сравнению: запятая становится точкой."""
    return raw.replace(",", ".").replace("−", "-")


def _unit(raw: str) -> str:
    """Приводит единицу к одному написанию, включая °C и °С."""
    return raw.lower().replace("ё", "е").replace("℃", "°c").replace("°с", "°c")


def _phrase(raw: str) -> str:
    """Убирает пробелы в резьбе, IP и ГОСТ, чтобы G1/2 совпал с G 1/2."""
    return re.sub(r"\s+", "", raw.lower().replace("ё", "е"))


def _unique(items: list[str]) -> list[str]:
    """Убирает повторы, сохраняя порядок первого появления."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result

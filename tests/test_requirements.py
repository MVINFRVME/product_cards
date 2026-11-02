"""Проверки требований ТЗ без сети: пустые поля, битый JSON, чужие характеристики, объём и запросы."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app


class StubClient:
    """Возвращает один и тот же ответ, не ходя в сеть."""

    def __init__(self, reply: str) -> None:
        """Запоминает ответ модели, который вернёт complete.

        Args:
            reply: Сырой JSON, как его прислал бы Groq.
        """
        self.reply = reply

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Возвращает заранее заданный ответ и не обращается к Groq."""
        return self.reply


def _client(reply: str) -> TestClient:
    """Собирает тестовый клиент приложения с подменной моделью.

    Args:
        reply: Ответ, который вернёт подменная модель.

    Returns:
        Клиент FastAPI без обращения к сети.
    """
    return TestClient(create_app(StubClient(reply)))


def test_invented_missing_catalog_is_collapsed() -> None:
    """Список чужих «параметр: данных нет» не превращается в девять одинаковых пометок."""
    laundry = " ".join(
        [
            "ГОСТ: данных нет.",
            "Питание: данных нет.",
            "Степень защиты: данных нет.",
            "Взрывозащита: данных нет.",
            "Материал корпуса: данных нет.",
            "Интерфейс подключения: данных нет.",
            "Погрешность измерения: данных нет.",
            "Отраслевое применение: данных нет.",
            "Класс точности: данных нет.",
            "Диапазон измерений составляет 0–1,6 МПа.",
        ]
    )
    reply = json.dumps(
        {
            "seo_title": "Манометр показывающий DM 02 Манотомь",
            "seo_description": laundry + (" Диапазон измерений 0–1,6 МПа." * 8),
            "search_queries": ["манометр DM 02", "манометр Манотомь", "манометр"],
            "commercial_block": "Для измерения избыточного давления.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Манометр показывающий DM 02",
                "manufacturer": "Манотомь",
                "description": "Прибор предназначен для измерения избыточного давления...",
                "specs": {
                    "диапазон измерений": "0–1,6 МПа",
                    "присоединение": "G1/2",
                    "диаметр корпуса": "100 мм",
                },
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    folded = card["seo_description"].casefold()
    assert "гост: данных нет" not in folded
    assert "питание: данных нет" not in folded
    assert "1,6" in folded
    removed = [gap for gap in card["gaps"] if gap["kind"] == "removed"]
    assert len(removed) == 1
    assert "модель сама перечислила эти характеристики" in removed[0]["note"]
    assert "Питание" in removed[0]["note"]
    assert "ГОСТ" in removed[0]["note"]


def test_thin_input_still_has_three_queries() -> None:
    """Только название: запросов всё равно 3–5, и слово «магазин» не путается с «газ»."""
    reply = json.dumps(
        {
            "seo_title": "Расходомер РМ-5",
            "seo_description": "Расходомер РМ-5 для интернет-магазина.",
            "search_queries": ["Расходомер РМ-5"],
            "commercial_block": "данных нет",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {"name": "Расходомер РМ-5", "manufacturer": "", "description": "", "specs": {"питание": ""}},
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert 3 <= len(card["search_queries"]) <= 5
    assert not any(gap["kind"] == "removed" for gap in card["gaps"])
    assert "магазина" in card["seo_description"]


def test_units_are_not_guessed_and_no_stubs_left() -> None:
    """Без единицы во входе модель не может её дописать, а от вырезанного не остаётся огрызков."""
    reply = json.dumps(
        {
            "seo_title": "Манометр DM 02, 0–1,6 МПа",
            "seo_description": (
                "Диапазон измерений 0–1,6 МПа. Присоединение G1/2. "
                "Прибор показывает избыточное давление на круглой шкале."
            ),
            "search_queries": ["манометр DM 02", "манометр 0–1,6 МПа", "манометр Манотомь"],
            "commercial_block": "Для измерения избыточного давления.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Манометр показывающий DM 02",
                "manufacturer": "Манотомь",
                "description": "Прибор предназначен для измерения избыточного давления...",
                "specs": {"диапазон измерений": "0–1,6", "присоединение": "LG"},
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert [item["value"] for item in card["specs"]] == ["0–1,6", "LG"]
    assert "МПа" not in card["seo_title"] + card["seo_description"]
    assert "G1/2" not in card["seo_description"]
    assert "Диапазон измерений." not in card["seo_description"]
    assert "Присоединение." not in card["seo_description"]
    assert "круглой шкале" in card["seo_description"]


def test_whitespace_spec_is_marked_missing() -> None:
    """Строка из пробелов в характеристике даёт «данных нет», как и null."""
    reply = json.dumps(
        {
            "seo_title": "Термопреобразователь ТП-2088",
            "seo_description": "Диапазон измерений: -50…200 °C. " * 20,
            "search_queries": ["ТП-2088", "термопреобразователь", "купить ТП-2088"],
            "commercial_block": "данных нет",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Термопреобразователь ТП-2088",
                "manufacturer": "",
                "description": "",
                "specs": {
                    "диапазон измерений": "-50…200 °C",
                    "класс допуска": " ",
                },
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert card["specs"][1] == {"name": "класс допуска", "value": "данных нет"}
    assert any(gap["field"] == "класс допуска" for gap in card["gaps"])


def test_empty_fields_are_marked_missing() -> None:
    """Пустые поля остаются в карточке со значением «данных нет»."""
    reply = json.dumps(
        {
            "seo_title": "Датчик давления ДД-1",
            "seo_description": "Короткое описание без характеристик.",
            "search_queries": ["датчик давления", "датчик ДД-1", "купить датчик"],
            "commercial_block": "Назначение в ответе модели не раскрыто.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Датчик давления ДД-1",
                "manufacturer": "",
                "description": "",
                "specs": {"питание": None, "класс точности": "  "},
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert [item["value"] for item in card["specs"]] == ["данных нет", "данных нет"]
    gap_fields = {gap["field"] for gap in card["gaps"]}
    assert {"Производитель", "Описание", "питание", "класс точности"} <= gap_fields
    assert all(gap["note"] == "данных нет" for gap in card["gaps"] if gap["field"] != "Текст модели")
    assert "данных нет" in card["markdown"]
    assert len(card["seo_title"]) <= 70
    assert 800 <= len(card["seo_description"]) <= 1200


def test_invalid_json_returns_400() -> None:
    """Не-JSON, массив и specs-список возвращают 400, карточки нет."""
    client = _client("{}")
    broken = client.post(
        "/api/generate",
        content="не json".encode(),
        headers={"Content-Type": "application/json"},
    )
    assert broken.status_code == 400
    assert "JSON" in broken.json()["error"]

    cut = client.post(
        "/api/generate",
        content='{\n  "name": "Манометр",\n  "specs": {\n'.encode(),
    )
    assert cut.status_code == 400
    assert "обрывается" in cut.json()["error"]
    assert cut.json()["line"] >= 3

    no_comma = client.post(
        "/api/generate",
        content='{\n  "name": "Манометр"\n  "specs": {}\n}'.encode(),
    )
    assert no_comma.status_code == 400
    assert no_comma.json()["line"] == 2
    assert "запятой" in no_comma.json()["error"]

    not_object = client.post("/api/generate", content=b"[]")
    assert not_object.status_code == 400
    assert "объект" in not_object.json()["error"]

    nothing = client.post(
        "/api/generate",
        content=b'{"name": "", "manufacturer": "", "description": "", "specs": {"a": ""}}',
    )
    assert nothing.status_code == 400
    assert "Нечего собирать" in nothing.json()["error"]

    bad_specs = client.post(
        "/api/generate",
        content='{"name":"Прибор","specs":["питание"]}'.encode(),
    )
    assert bad_specs.status_code == 400
    assert "specs" in bad_specs.json()["error"]


def test_filled_spec_is_not_called_missing() -> None:
    """Модель не может написать, что заполненный класс точности не указан."""
    reply = json.dumps(
        {
            "seo_title": "Манометр показывающий DM 02",
            "seo_description": "Класс точности – данных нет. Диапазон 0–1,6 МПа.",
            "search_queries": ["манометр DM 02", "манометр Манотомь", "манометр"],
            "commercial_block": "Для измерения избыточного давления. По назначению данных нет.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Манометр показывающий DM 02",
                "manufacturer": "Манотомь",
                "description": "Прибор предназначен для измерения избыточного давления...",
                "specs": {
                    "диапазон измерений": "0–1,6 МПа",
                    "класс точности": "1,5",
                },
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert "по назначению данных нет" not in card["commercial_block"].casefold()
    assert "класс точности – данных нет" not in card["seo_description"].casefold()
    assert "1,5" in card["seo_description"]
    assert card["specs"][1]["value"] == "1,5"


def test_invented_pressure_is_removed() -> None:
    """Давление, которого не было во входе, вырезается, а диапазон из входа остаётся."""
    reply = json.dumps(
        {
            "seo_title": "Манометр 2,5 МПа",
            "seo_description": "Прибор держит 0–1,6 МПа, а в паспорте также указано 2,5 МПа.",
            "search_queries": ["манометр 2,5 МПа", "манометр DM 02", "манометр Манотомь"],
            "commercial_block": "Подходит для контура 2,5 МПа.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Манометр показывающий DM 02",
                "manufacturer": "Манотомь",
                "specs": {
                    "диапазон измерений": "0–1,6 МПа",
                    "класс точности": "1,5",
                    "присоединение": "G1/2",
                    "диаметр корпуса": "100 мм",
                },
                "description": "Прибор предназначен для измерения избыточного давления...",
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    prose = " ".join(
        [card["seo_title"], card["seo_description"], card["commercial_block"], *card["search_queries"]]
    )
    assert "2,5" not in prose
    assert "1,6" in card["seo_description"]
    assert any("2,5 МПа" in gap["note"] for gap in card["gaps"])
    assert [item["value"] for item in card["specs"]] == [
        "0–1,6 МПа",
        "1,5",
        "G1/2",
        "100 мм",
    ]


def test_other_gaps_do_not_drop_a_known_range() -> None:
    """Фраза про остальные пропуски не считается отказом от уже указанного диапазона."""
    known = "Доступна только информация о диапазоне измерений, остальные характеристики отсутствуют."
    reply = json.dumps(
        {
            "seo_title": "Термопреобразователь ТП-2088, производитель данных нет, диапазон -50…200°C",
            "seo_description": f"{known} " + ("Диапазон измерений: -50…200 °C. " * 12),
            "search_queries": ["ТП-2088", "термопреобразователь", "купить ТП-2088"],
            "commercial_block": "данных нет",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Термопреобразователь ТП-2088",
                "manufacturer": "",
                "description": "",
                "specs": {
                    "диапазон измерений": "-50…200 °C",
                    "выходной сигнал": None,
                },
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert known in card["seo_description"]
    assert card["seo_title"].startswith("Термопреобразователь ТП-2088")
    assert "-50…200" in card["seo_title"]
    assert len(card["seo_title"]) <= 70
    assert not any("остальные" in gap["note"] for gap in card["gaps"])
    assert card["specs"][0]["value"] == "-50…200 °C"
    assert card["specs"][1]["value"] == "данных нет"


def test_repeated_range_is_collapsed() -> None:
    """Один и тот же диапазон разными словами не занимает описание несколько раз."""
    repeated = " ".join(
        [
            "Диапазон измерений составляет 0–1,6 МПа.",
            "Диапазон измерений: 0–1,6 МПа.",
            "Диапазон измерений 0–1,6 МПа позволяет контролировать давление.",
            "Прибор предназначен для измерения избыточного давления.",
            "Прибор предназначен для измерения избыточного давления и имеет указанные характеристики.",
        ]
        * 3
    )
    reply = json.dumps(
        {
            "seo_title": "Манометр показывающий DM 02, производитель",
            "seo_description": repeated,
            "search_queries": ["манометр DM 02", "манометр Манотомь", "манометр"],
            "commercial_block": "Для измерения избыточного давления.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post("/api/generate", content=_manometer().encode())
    assert response.status_code == 200
    card = response.json()
    assert card["seo_description"].count("0–1,6") <= 3
    assert card["seo_title"] == "Манометр показывающий DM 02"
    assert 800 <= len(card["seo_description"]) <= 1200


def test_thread_paraphrase_and_industry_claim_are_split_from_gaps() -> None:
    """Чужая резьба словами и «промышленные условия» уходят в отдельный список."""
    reply = json.dumps(
        {
            "seo_title": "Манометр показывающий DM 02",
            "seo_description": (
                "Присоединение называют четвертьдюймовой резьбой. "
                "Присоединение G1/2 называют полудюймовым. "
                "Прибор рассчитан на промышленные условия и нормативные значения. "
                "Диапазон измерений составляет 0–1,6 МПа."
            ),
            "search_queries": ["манометр DM 02", "манометр Манотомь", "манометр"],
            "commercial_block": "Для измерения избыточного давления.",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post("/api/generate", content=_manometer().encode())
    assert response.status_code == 200
    card = response.json()
    folded = card["seo_description"].casefold()
    assert "четвертьдюйм" not in folded
    assert "промышлен" not in folded
    assert "норматив" not in folded
    assert "полудюйм" in folded
    assert "1,6" in folded
    removed = [gap for gap in card["gaps"] if gap["kind"] == "removed"]
    missing = [gap for gap in card["gaps"] if gap["kind"] != "removed"]
    assert missing == []
    assert any("четвертьдюйм" in gap["note"].casefold() for gap in removed)
    assert any("утверждение про промышленное применение" in gap["note"] for gap in removed)
    assert all(gap["note"].startswith("Убрано: «") and "\nПочему: " in gap["note"] for gap in removed)
    assert "Что уточнить" not in card["markdown"]
    assert "Удалено из текста" not in card["markdown"]
    from app.domain import ProductCard, render_notes_markdown

    notes = render_notes_markdown(ProductCard(**card))
    assert "Все поля заполнены." in notes
    assert "## Удалено из текста — этого нет в ваших данных" in notes


def test_repeated_missing_lines_collapse_and_empty_removed_is_hidden() -> None:
    """«Производитель: данных нет.» десять раз подряд остаётся один раз, а пустого блока «Удалено» нет."""
    laundry = "Производитель: данных нет. Описание: данных нет. " * 10
    reply = json.dumps(
        {
            "seo_title": "Манометр показывающий DM 02",
            "seo_description": "Манометр показывающий DM 02, диапазон 0–1,6 МПа. " + laundry,
            "search_queries": ["манометр DM 02", "манометр 0–1,6 МПа", "купить манометр DM 02"],
            "commercial_block": "данных нет",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Манометр показывающий DM 02",
                "manufacturer": "",
                "description": "",
                "specs": {"диапазон измерений": "0–1,6 МПа"},
            },
            ensure_ascii=False,
        ).encode(),
    )
    assert response.status_code == 200
    card = response.json()
    assert card["seo_description"].count("Производитель: данных нет.") == 1
    assert "Удалено из текста" not in card["markdown"]


def test_restated_gaps_and_nbsp_fragments() -> None:
    """Пропуски, пересказанные разными словами, остаются один раз; фраза с неразрывными пробелами вырезается."""
    reply = json.dumps(
        {
            "seo_title": "Манометр DM 02",
            "seo_description": (
                "Манометр показывающий DM 02. Диапазон измерений 0–1,6 МПа. Производитель: данных нет. "
                "Производитель данных нет. Манометр показывающий DM 02, диапазон 0–1,6 МПа, производитель данных нет. "
                "Прибор\u00a0работает в\u00a0технологических системах. Шкала прибора круглая."
            ),
            "search_queries": ["манометр DM 02", "манометр 0–1,6 МПа", "купить манометр DM 02"],
            "commercial_block": "данных нет",
        },
        ensure_ascii=False,
    )
    response = _client(reply).post(
        "/api/generate",
        content=json.dumps(
            {
                "name": "Манометр показывающий DM 02",
                "manufacturer": "",
                "description": "",
                "specs": {"диапазон измерений": "0–1,6 МПа"},
            },
            ensure_ascii=False,
        ).encode(),
    )
    card = response.json()
    text = card["seo_description"].casefold()
    assert text.count("производитель") == 1
    assert "технологических" not in text
    assert "шкала прибора круглая" in text
    assert any("технологических" in gap["note"] for gap in card["gaps"] if gap["kind"] == "removed")


def test_commercial_block_is_not_a_copy_of_description() -> None:
    """Если модель повторила описание, блок для менеджера собирается из назначения и характеристик."""
    reply = json.dumps(
        {
            "seo_title": "Манометр показывающий DM 02",
            "seo_description": "Манометр показывающий DM 02, диапазон 0–1,6 МПа. " * 20,
            "search_queries": ["манометр DM 02", "манометр 0–1,6 МПа", "купить манометр DM 02"],
            "commercial_block": "Прибор предназначен для измерения избыточного давления...",
        },
        ensure_ascii=False,
    )
    card = _client(reply).post("/api/generate", content=_manometer().encode()).json()
    assert card["commercial_block"].startswith("Подходит для задач измерения избыточного давления.")
    assert "класс точности 1,5" in card["commercial_block"]


def test_rate_limit_message_says_what_and_when() -> None:
    """Суточный лимит Groq объясняется словами и со временем ожидания, а не «ошибка 429»."""
    from app.llm import _rate_limit_message

    daily = _rate_limit_message("Limit reached on tokens per day (TPD). Please try again in 4m29.5s.")
    assert daily.startswith("Закончился суточный лимит токенов Groq")
    assert "через 5 мин" in daily
    assert "Карточка не собрана" in daily


def _manometer() -> str:
    """Вход манометра из тестового задания."""
    return json.dumps(
        {
            "name": "Манометр показывающий DM 02",
            "manufacturer": "Манотомь",
            "specs": {
                "диапазон измерений": "0–1,6 МПа",
                "класс точности": "1,5",
                "присоединение": "G1/2",
                "диаметр корпуса": "100 мм",
            },
            "description": "Прибор предназначен для измерения избыточного давления...",
        },
        ensure_ascii=False,
    )

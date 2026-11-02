"""Клиент LLM. В прототипе одна реализация: Groq."""

from __future__ import annotations

import os
import re
from typing import Protocol

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.domain import CardError

DEFAULT_MODEL = "openai/gpt-oss-120b"
_RETRY_IN = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)(?:\.\d+)?s)?", re.IGNORECASE)


def _rate_limit_message(detail: str) -> str:
    """Объясняет лимит Groq словами контент-менеджера и называет время ожидания.

    Args:
        detail: Текст ошибки Groq, в нём бывает тип лимита и «try again in 4m29s».

    Returns:
        Сообщение: какой лимит кончился и через сколько повторить, если Groq это сообщил.
    """
    daily = "per day" in detail.casefold()
    reason = "Закончился суточный лимит токенов Groq" if daily else "Groq временно ограничил частоту запросов"
    wait = ""
    match = _RETRY_IN.search(detail)
    if match and any(match.groups()):
        hours, minutes, seconds = (int(value or 0) for value in match.groups())
        total = hours * 60 + minutes + (1 if seconds else 0)
        wait = f" Повторите примерно через {max(total, 1)} мин."
    return f"{reason}.{wait} Карточка не собрана."


class LLMClient(Protocol):
    """Узкий контракт генератора: текст на вход, текст на выход."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Возвращает сырой ответ модели.

        Args:
            messages: Сообщения chat completions.

        Returns:
            Текст ответа. Это ещё не карточка.

        Raises:
            CardError: Клиент не смог получить текст ответа.
        """


class GroqClient:
    """Вызов Groq Chat Completions."""

    def __init__(self, api_key: str, model: str) -> None:
        """Запоминает ключ и модель. Окружение здесь больше не читается.

        Args:
            api_key: Ключ Groq. Пустая строка допустима до первого запроса.
            model: Идентификатор модели. Пустая строка заменяется на модель по умолчанию.
        """
        self._api_key = api_key.strip()
        self._model = model.strip() or DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> GroqClient:
        """Собирает клиент из GROQ_API_KEY и GROQ_MODEL.

        Returns:
            Клиент. Пустой ключ не рвёт импорт: ошибка появится на запросе.
        """
        return cls(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=os.environ.get("GROQ_MODEL", ""),
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Запрашивает JSON-ответ у Groq.

        Args:
            messages: System, user и при необходимости черновик на исправление.

        Returns:
            Непустой текст ответа.

        Raises:
            CardError: Нет ключа, таймаут, сеть, 401, 429, другой статус или пустой ответ.
        """
        if not self._api_key:
            raise CardError(
                "Не задан GROQ_API_KEY. Добавьте ключ в файл .env и перезапустите сервер. Карточка не собрана.",
                503,
            )
        client = Groq(api_key=self._api_key, timeout=60.0)
        try:
            response = self._create(client, messages, json_mode=True)
        except APIStatusError as exc:
            if exc.status_code != 400:
                self._raise_status(exc)
            try:
                response = self._create(client, messages, json_mode=False)
            except APIStatusError as retry_exc:
                self._raise_status(retry_exc)
            except APITimeoutError as retry_exc:
                raise CardError("Groq не ответил вовремя. Карточка не собрана.", 504) from retry_exc
            except APIConnectionError as retry_exc:
                raise CardError("Нет соединения с Groq. Карточка не собрана.", 503) from retry_exc
        except APITimeoutError as exc:
            raise CardError("Groq не ответил вовремя. Карточка не собрана.", 504) from exc
        except APIConnectionError as exc:
            raise CardError("Нет соединения с Groq. Карточка не собрана.", 503) from exc
        return self._read(response)

    def _create(self, client: Groq, messages: list[dict[str, str]], json_mode: bool) -> object:
        """Отправляет chat completion, по желанию в режиме JSON.

        Args:
            client: Уже созданный SDK-клиент.
            messages: Диалог с моделью.
            json_mode: Просить объект JSON через response_format.

        Returns:
            Сырой ответ SDK.
        """
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4000,
            "reasoning_effort": "low",
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    def _read(self, response: object) -> str:
        """Достаёт текст первого варианта ответа.

        Args:
            response: Ответ chat.completions.create.

        Returns:
            Непустая строка.

        Raises:
            CardError: Вариантов нет или текст пустой.
        """
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise CardError("Groq вернул пустой ответ. Карточка не собрана.", 502)
        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise CardError("Groq вернул пустой ответ. Карточка не собрана.", 502)
        return content

    def _raise_status(self, exc: APIStatusError) -> None:
        """Переводит статус Groq в CardError и всегда выбрасывает его.

        Args:
            exc: Ошибка SDK со статус-кодом.

        Raises:
            CardError: 401, 429 или другой код. Черновик при этом не собирается.
        """
        if isinstance(exc, RateLimitError) or exc.status_code == 429:
            raise CardError(_rate_limit_message(str(exc)), 429) from exc
        if isinstance(exc, AuthenticationError) or exc.status_code == 401:
            raise CardError(
                "Groq отклонил ключ. Проверьте GROQ_API_KEY. Карточка не собрана.",
                401,
            ) from exc
        raise CardError(
            f"Groq вернул ошибку {exc.status_code}. Карточка не собрана.",
            502,
        ) from exc

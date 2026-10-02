"""Offline stand-ins shared by the integration tests.

``FakeOpenAIClient`` replaces the HTTP client behind the production pipelines, so
everything up to the HTTP call (prompt building, request shaping, response parsing,
document writing) runs for real. ``FakePineconeIndex`` does the same for the real
``PineconeDocumentStore``. ``telegram_message`` and ``FakeClock`` build the inputs that
the real TeleBot dispatch and the handlers need.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING

from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.create_embedding_response import Usage
from telebot.types import Chat, Message, User

# Haystack is imported inside the functions that need it, never here: conftest imports
# this module before it opts Haystack out of telemetry, and Haystack reads that setting
# when it is first imported.
if TYPE_CHECKING:
    from haystack import Document
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# Shaped like a real Telegram bot token: "<bot id>:<secret>".
BOT_TOKEN = "123456789:AAH-s3cretTokenValue_0123456789abcdefghi"
BOT_USERNAME = "ThisBot"
GROUP_CHAT_ID = -1001234567890
SENT_AT = 1_705_320_600  # 2024-01-15T12:30:00Z

DEFAULT_REPLY = "Тема\nИтог"
DEFAULT_VECTOR = (0.1, 0.2, 0.3)


class FakeOpenAIClient:
    """Answers embedding and chat requests and records them.

    Set ``embedding_error`` or ``chat_error`` to make the matching endpoint raise.
    """

    def __init__(self, *, reply: str = DEFAULT_REPLY, vector: Sequence[float] = DEFAULT_VECTOR) -> None:
        self.reply = reply
        self.vector = list(vector)
        self.embedding_error: Exception | None = None
        self.chat_error: Exception | None = None
        self.embedding_inputs: list[list[str]] = []
        self.chat_requests: list[list[dict[str, object]]] = []
        self.embeddings = SimpleNamespace(create=self._create_embeddings)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat))

    def _create_embeddings(self, *, input: list[str], **_: object) -> CreateEmbeddingResponse:
        if self.embedding_error is not None:
            raise self.embedding_error
        self.embedding_inputs.append(list(input))
        return CreateEmbeddingResponse(
            data=[
                Embedding(embedding=list(self.vector), index=index, object="embedding")
                for index in range(len(input))
            ],
            model="test-embedding-model",
            object="list",
            usage=Usage(prompt_tokens=1, total_tokens=1),
        )

    def _create_chat(self, *, messages: list[dict[str, object]], **_: object) -> ChatCompletion:
        if self.chat_error is not None:
            raise self.chat_error
        self.chat_requests.append(messages)
        return ChatCompletion(
            id="chatcmpl-test",
            created=0,
            model="test-chat-model",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content=self.reply),
                )
            ],
        )

    def prompt_text(self) -> str:
        """The text of the only chat request made so far, all roles joined."""
        assert len(self.chat_requests) == 1, f"expected one chat request, got {len(self.chat_requests)}"
        return "\n".join(str(message["content"]) for message in self.chat_requests[0])


class FakePineconeIndex:
    """The Pinecone SDK's index client, answering from memory.

    Set ``upsert_error`` or ``query_error`` to make the matching call raise. Documents
    given to ``serve`` come back from every query, shaped like Pinecone's matches.
    """

    def __init__(self) -> None:
        self.upsert_error: Exception | None = None
        self.query_error: Exception | None = None
        self.upserted: list[object] = []
        self._served: list[Document] = []

    def serve(self, documents: Sequence[Document]) -> None:
        self._served = list(documents)

    def upsert(self, *, vectors: Sequence[object], **_: object) -> SimpleNamespace:
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserted.extend(vectors)
        return SimpleNamespace(upserted_count=len(vectors))

    def query(self, **_: object) -> SimpleNamespace:
        if self.query_error is not None:
            raise self.query_error
        # PineconeDocumentStore mutates each match's metadata, so build them per query.
        return SimpleNamespace(
            matches=[
                {
                    "id": document.id,
                    "values": list(DEFAULT_VECTOR),
                    "score": 0.5,
                    "metadata": {**document.meta, "content": document.content},
                }
                for document in self._served
            ]
        )

    def describe_index_stats(self) -> dict[str, object]:
        return {}


def pinecone_document_store(index: FakePineconeIndex) -> PineconeDocumentStore:
    """The production store class with ``index`` as its remote client.

    Setting ``_index`` is the seam: the store only connects to Pinecone, listing and
    possibly creating indexes, while that attribute is unset.
    """
    from haystack.utils import Secret
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

    store = PineconeDocumentStore(
        api_key=Secret.from_token("test-pinecone-key"),
        index="test-index",
        namespace="test-namespace",
        dimension=len(DEFAULT_VECTOR),
        show_progress=False,
    )
    store._index = index  # type: ignore[assignment]
    return store


def telegram_message(
    text: str,
    *,
    message_id: int = 1,
    chat_id: int = GROUP_CHAT_ID,
    chat_type: str = "supergroup",
    user_id: int = 7,
    is_bot: bool = False,
    first_name: str = "Alice",
    last_name: str | None = None,
    username: str | None = "alice",
    sender_chat: Chat | None = None,
    date: object = SENT_AT,
) -> Message:
    """A Telegram text message as TeleBot delivers it to handlers."""
    options: dict[str, object] = {"text": text}
    if sender_chat is not None:
        options["sender_chat"] = sender_chat
    return Message(
        message_id=message_id,
        from_user=User(
            id=user_id,
            is_bot=is_bot,
            first_name=first_name,
            last_name=last_name,
            username=username,
        ),
        date=date,
        chat=Chat(id=chat_id, type=chat_type, title="Team Chat"),
        content_type="text",
        options=options,
        json_string="{}",
    )


def anonymous_admin_message(text: str, *, message_id: int = 1) -> Message:
    """The payload Telegram sends for a group admin posting anonymously.

    ``from`` is the fake GroupAnonymousBot user and ``sender_chat`` is the group itself.
    """
    return telegram_message(
        text,
        message_id=message_id,
        user_id=1087968824,
        is_bot=True,
        first_name="Group",
        username="GroupAnonymousBot",
        sender_chat=Chat(id=GROUP_CHAT_ID, type="supergroup", title="Team Chat"),
    )


class FakeClock:
    """A monotonic clock that only moves when the test advances ``now``."""

    def __init__(self, now: float = 5_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def ticking_clock(step: float = 1.0) -> Callable[[], float]:
    """A monotonic clock that advances ``step`` on every reading.

    Polling helpers compare it with a deadline, so a poll with ``timeout_seconds=3``
    really runs a couple of times and then gives up. (A frozen clock with a zero
    timeout would skip the loop entirely and raise the timeout without polling.)
    """
    now = -step

    def monotonic() -> float:
        nonlocal now
        now += step
        return now

    return monotonic

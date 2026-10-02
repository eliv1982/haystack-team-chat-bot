# Architecture

Companion to the [README](../README.md), which shows the capture and summary paths and the reliability decisions. This page holds the detail behind them: the module map, the session lifecycle and what the user sees in each failure mode.

## Pipelines

| Pipeline | Components | Used by |
| -------- | ---------- | ------- |
| Indexing | `OpenAIDocumentEmbedder` → `DocumentWriter` (`OVERWRITE`) → `PineconeDocumentStore` | bot runtime, indexing and summarization smokes |
| Summarization | `ChatPromptBuilder` → `OpenAIChatGenerator` | bot runtime, summarization smoke |
| Retrieval (probe) | `OpenAITextEmbedder` → `PineconeEmbeddingRetriever` | retrieval smoke only |

`runtime.py` builds the first two. The retrieval pipeline is a retained capability: it answers "which messages of this session are most similar to a query", which is a different question from "what was said in this session", so it is not the source of a summary. If no semantic-search feature is planned it is a candidate for removal (see the [roadmap](project_roadmap.md)).

Shared between the retrieval probe and the summary path: `retrieval_filters.build_session_filter` (the `chat_id AND session_id` filter) and `retrieval_service.validate_scoped_document` (the per-document scope check).

## Module map

| Module | Responsibility |
| ------ | -------------- |
| `config.py` | `Settings` loaded and validated from the environment; credentials are excluded from `repr` |
| `models.py` | domain types: `ChatMessage`, `SummarizationRequest` / `SummarizationResult`, `RetrievalRequest` |
| `documents.py` | `ChatMessage` → Haystack `Document`: deterministic ID, content format, metadata |
| `document_store.py` | `PineconeDocumentStore` factory |
| `pinecone_preflight.py` | startup check of an existing index (exists, ready, dimension, metric) |
| `indexing_service.py` | runs the indexing pipeline and validates its output |
| `session_documents.py` | whole-session load: hard filter, scope re-validation, completeness gate, chronological order |
| `summarization_prompt.py` | system and user prompt template |
| `summarization_service.py` | session load → output validation → summarization pipeline → `SummarizationResult` |
| `retrieval_filters.py`, `retrieval_service.py` | filter builder and scope validation (shared); the retrieval probe |
| `pipelines.py` | factories for the three pipelines |
| `session_store.py` | thread-safe in-memory registry: one active and one latest completed session per chat |
| `telegram_adapter.py` | pure Telegram `Message` → `ChatMessage`, sender and command checks, summary phrases |
| `telegram_application.py` | listening lifecycle, per-chat locking, recording |
| `telegram_summary_application.py` | session resolution, completeness retry, summary orchestration |
| `telegram_handlers.py` | handler registration, user-facing replies, failure policy, reply splitting |
| `telegram_commands.py` | command menu for group chats |
| `telegram_bot.py` | `TeleBot` factory |
| `runtime.py` | assembly of pipelines, services and bot; polling boundary |
| `bot.py` | entry point |
| `error_reporting.py` | exception descriptions for logs that never include the exception message |
| `scripts/` | three live smoke tests |
| `tests/` | offline suite (see the README) |

Nothing is built at import time: no bot, pipeline or store exists as a module-level global, and importing the application reads no `.env` file. A test enforces this.

## Session lifecycle

```text
none ──/start_listening──▶ active ──/stop_listening──▶ latest completed
                              │                              │
                              └─ record: message_count + 1   └─ replaced by the next completed session
```

- `/start_listening` while a session is active is rejected.
- `/stop_listening` moves the active session to "latest completed", replacing the previous one.
- `/summary` resolves the active session first, otherwise the latest completed one.
- The registry is in memory, keyed by chat ID. After a restart it is empty; documents already in Pinecone are not reachable through the bot.

## Concurrency

TeleBot runs handlers on worker threads. The session store is guarded by a lock, and recording and stopping take a per-chat lock so that a message cannot be counted into a session that is being stopped. Summaries do not take the chat lock: the completeness check re-reads the session's count on every attempt, so messages recorded during the retry window are not mistaken for extra documents.

## Failure modes: what the user sees

| Situation | Behaviour | Reply to the chat |
| --------- | --------- | ----------------- |
| Any message in a private chat or channel | ignored entirely (handlers are registered for groups and supergroups only) | none |
| `/summary`, no active or completed session | nothing to summarize | "no recorded discussion yet" |
| Session with zero recorded messages | no model call | "not enough saved messages" |
| Fewer documents visible than counted | re-checked up to 3 times, 1.5 s apart, then refused; no model call | "summary not prepared: repeat in a few seconds; no partial summary is produced" |
| More documents than counted | refused at once, never retried; logged as an error | generic internal-error reply |
| Registered count ≥ 1,000 | refused without querying the store | explains the limit; suggests a new discussion |
| OpenAI, Pinecone or invalid pipeline output during a summary | no application-level retry or fallback | generic internal-error reply |
| Indexing fails while recording | the message is not counted | warning that some messages may be missing, at most once per 10 minutes per chat |
| Telegram or network error while replying | logged and absorbed in the handler | none (the reply could not be sent) |

The application's own log lines carry counts and exception types, never message text, prompts, summaries or credentials. Some warning and error lines include the Telegram chat ID.

# Haystack Team Chat Bot

A Telegram bot that records an explicitly started group discussion and, on request, summarizes the **whole** session. Messages are captured with deterministic identities, embedded with OpenAI and stored in Pinecone with hard chat/session isolation. A summary is built from the complete, chronologically ordered session record, and the bot **refuses instead of guessing** when that record is incomplete or inconsistent. [Haystack](https://haystack.deepset.ai/) provides the pipeline composition.

This is a portfolio engineering project: a working, tested, single-process MVP. It is not a production service, and it makes no claim of production-scale readiness (see [Known limitations](#known-limitations)). The bot's replies and its summarization prompt are in Russian; this documentation is in English.

## Why this project

A chat summarizer that silently drops messages is worse than no summarizer: the reader cannot tell what is missing. The interesting engineering problem is not "call an LLM on some messages" but making the input set **provably the right one** and failing loudly otherwise:

- which messages belong to a discussion, and no others;
- whether every one of them reached the summarizer;
- what happens when the vector store, the model or Telegram misbehave.

The project is built around those questions, with offline deterministic tests for the failure paths and separate live smoke tooling for the real services.

## Core workflow

| Command            | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `/start_listening` | Start recording; the bot announces in the group what it will save |
| `/summary`         | Summarize the active session, or the latest completed one         |
| `/stop_listening`  | Stop recording                                                    |
| `/status`          | Show recording state and the number of recorded messages         |
| `/help`            | Show usage                                                        |

The exact phrases «Подведи итог», «Подведи итог обсуждения» and «Что думаешь?» are aliases for `/summary` (case- and whitespace-insensitive).

```text
/start_listening → participants write → /status → /summary → /stop_listening → /summary
```

- Works in groups and supergroups only; text messages only.
- Messages are recorded only while a listening session is active, and only after they were indexed successfully.
- Commands, summary requests, photos, files, voice, video, stickers, polls and service messages are not recorded.
- Per chat there is one active session and one latest completed session. `/summary` after `/stop_listening` summarizes the latest completed one.
- A summary lists positions, decisions, next actions with owners and deadlines, and unresolved questions. It ends with a short AI recommendation that is labeled as such and kept separate from what the participants decided.

## Architecture

### Capture path

```text
Telegram update (text message in an active session)
→ Telegram adapter / application service
→ deterministic message document
→ Haystack indexing pipeline
→ OpenAI document embedding
→ Pinecone
→ session message_count + 1   (only after a successful write)
```

- The document ID is a SHA-256 of `telegram-message:v1:{chat_id}:{message_id}`, and the write policy is `OVERWRITE`, so re-indexing a message is idempotent.
- The embedder runs with `raise_on_failure=True`. By default Haystack would log a failed embedding request and let the store write a dummy vector that is still reported as written.

### Summary path

```text
Telegram /summary
→ resolve active, else latest completed, session
→ fetch documents with a hard chat_id + session_id metadata filter
→ re-validate every document's scope and invariants
→ exact document-count completeness gate
→ chronological sort (sent_at, message_id, document id)
→ Haystack summarization pipeline
→ OpenAI chat generation
→ Telegram-safe split reply
```

**Semantic top-k retrieval is not used as the source set for a summary.** A similarity search returns "the most relevant" messages, which is the wrong contract for "everything that was said". The summary is built from an exhaustive, filter-only load of the session.

### The retrieval pipeline's remaining role

The repository still contains a separate retrieval pipeline (`OpenAITextEmbedder → PineconeEmbeddingRetriever`, filtered by chat and session) and its live smoke test. It is a **retrieval capability / probe**, not part of the bot's runtime: `runtime.py` assembles only the indexing and summarization pipelines. Two of its pieces are shared with the summary path: the chat/session filter builder and the scope validator. `RETRIEVAL_TOP_K` is used only by the retrieval probe.

### Haystack's role

Haystack composes the two production pipelines (indexing: `OpenAIDocumentEmbedder → DocumentWriter → PineconeDocumentStore`; summarization: `ChatPromptBuilder → OpenAIChatGenerator`) and supplies the document and store abstractions. Telegram handling, session lifecycle, the completeness gate and failure policy are application code around it.

Module map, session lifecycle and the failure-mode table: [docs/architecture.md](docs/architecture.md).

## Reliability and data-boundary decisions

- **Hard isolation.** All chats share one Pinecone namespace; isolation is a `chat_id AND session_id` metadata filter on every read, and each loaded document is re-validated against that scope (including `source`) before use. A mismatch fails closed.
- **Exact completeness gate.** The session registry counts messages that were indexed successfully. A summary is generated only if exactly that many valid, unique documents are loaded:
  - fewer visible → the vector index may simply be behind; the check is repeated up to 3 times, 1.5 s apart, then the summary is refused with a "try again shortly" reply;
  - more than counted → a data-consistency error; refused immediately and never retried;
  - registered count ≥ 1,000 → refused without querying (see below).
- **No partial summaries.** Every refusal above is deliberate. The model is not called, and no summary built from a subset is returned.
- **What the gate does and does not prove.** It compares the store with the bot's *own* count of successful writes. It does not detect messages the bot never captured (see [Known limitations](#known-limitations)).
- **The 1,000-document boundary is explicit.** Pinecone answers metadata-filter queries with a similarity query capped at 1,000 results and gives no truncation signal. A result of exactly 1,000 cannot be told from a truncated one, so sessions registered with 1,000 or more messages are refused rather than risk a silently truncated summary.
- **Failure handling.** Telegram/network errors while replying are absorbed in the handler instead of restarting polling; other handler errors produce a generic reply. Third-party exception text can contain credentials (a `requests` error embeds the bot token in its URL), so only the exception type and source location are logged.
- **Concurrency.** Recording and stopping are serialized per chat; the in-memory session store is lock-protected.
- **Prompt.** Participant messages are passed as data, not instructions; the prompt carries grounding, anti-injection, proposal-versus-decision and completeness rules. Prompt-level mitigation is best-effort, not a guarantee.

## Setup

Python 3.10 or newer (CI runs 3.10 and 3.12; development used 3.12).

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt          # runtime
python -m pip install -r requirements-dev.txt      # development: runtime + pytest + ruff
cp .env.example .env                               # Windows: Copy-Item .env.example .env
```

Fill in `.env` locally. It is ignored by Git; `.env.example` contains placeholders only.

## Configuration

| Variable | Required | Purpose |
| -------- | -------- | ------- |
| `TELEGRAM_BOT_TOKEN` | yes | Telegram Bot API token |
| `OPENAI_API_KEY` | yes | OpenAI API key (embeddings and chat) |
| `OPENAI_BASE_URL` | no | OpenAI-compatible endpoint; omit for the OpenAI API. Message text is sent to this endpoint instead |
| `OPENAI_MODEL` | yes | chat model used for summaries |
| `EMBEDDING_MODEL` | yes | embedding model used for indexing |
| `PINECONE_API_KEY` | yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | yes | name of an **existing** Pinecone index |
| `PINECONE_NAMESPACE` | no | namespace; built-in default `haystack-team-chat-homework` (a legacy name kept so existing data stays addressable, so set your own explicitly) |
| `PINECONE_DIMENSION` | no | expected index dimension; default `1536` |
| `PINECONE_METRIC` | no | expected index metric (`cosine`, `dotproduct`, `euclidean`); default `cosine` |
| `RETRIEVAL_TOP_K` | no | `top_k` of the retrieval probe only; default `50`. Still validated at startup, but unused by the bot's runtime |
| `HAYSTACK_TELEMETRY_ENABLED` | no | see [Telemetry](#telemetry) |

**Pinecone index.** The bot never creates an index. At startup a preflight checks that the index exists, is ready, and that its dimension and metric equal `PINECONE_DIMENSION` / `PINECONE_METRIC`; otherwise it refuses to start. The embedding model must return vectors of exactly that dimension (1536 matches models such as `text-embedding-3-small`). Summaries load documents by metadata filter, not by similarity, so the metric matters only for the retrieval probe.

**Telegram.** Create the bot with BotFather and add it to a group or supergroup. It must receive ordinary group messages, not just commands: make it a group administrator (the setup used in live acceptance), or disable privacy mode in BotFather (`/setprivacy`) and re-add the bot to the group (not exercised in live acceptance). The command menu is registered by the bot at startup. Anyone in the chat can use the commands; there is no access control.

### Telemetry

Haystack sends anonymous usage statistics (to PostHog) **by default**. The offline test suite and CI disable it; the bot does not. To disable it when running the bot or the smoke scripts, set `HAYSTACK_TELEMETRY_ENABLED=False` in the **shell environment** that launches the process. Setting it in `.env` is too late: Haystack reads it when it is imported, before the bot loads `.env`.

## Running the bot

```bash
python bot.py
```

Startup: settings → Pinecone preflight → document store → indexing and summarization pipelines → services → handlers → command menu → long polling. Notes:

- Stop with `Ctrl+C`.
- Only one polling process per bot token may run. Telegram error `409 Conflict` means a second one is running.
- Polling starts with `skip_pending=True` and `allowed_updates=["message"]`: updates that arrived while the bot was down are discarded, and edited messages are never received.

## Tests and CI

```bash
python -m pytest -q
python -m ruff check .
python -m pip check
```

A bare `pytest -q` also works (`pyproject.toml` sets the path). The suite has **720 tests** at the last documented baseline; this is the one place the number is stated.

The suite is fully offline and deterministic:

- real Haystack components and pipelines, with a fake OpenAI client behind them and Haystack's `InMemoryDocumentStore` standing in for Pinecone;
- a real `TeleBot` dispatching handlers with the Bot API mocked, so routing, command addressing and handler failures are exercised;
- `tests/conftest.py` fails any test that connects to a non-loopback host, and disables Haystack telemetry.

GitHub Actions (`.github/workflows/ci.yml`) runs ruff, pytest and `pip check` on Python 3.10 and 3.12 for every push and pull request. It needs no secrets and calls no live service. Code coverage is not measured.

Pinecone-specific behaviour (the 1,000-result cap, index lag) is simulated offline; it is exercised against the real service only by the live smoke tests.

## Live smoke tests

These call real OpenAI and Pinecone with your credentials, may incur cost, and are **not** part of CI or normal verification. Run them from the repository root with a configured `.env`:

| Command | What it checks |
| ------- | -------------- |
| `python scripts/smoke_test_indexing.py` | Pinecone preflight; indexes one synthetic message; verifies the stored document, metadata and embedding dimension |
| `python scripts/smoke_test_retrieval.py` | the retrieval probe: filtered semantic search isolates one session and one chat (target session, another session of the same chat, another chat) |
| `python scripts/smoke_test_summarization.py` | the **production summary path** on a seven-message synthetic corpus: whole-session load and completeness gate, then a summary checked for exact source set, structure, grounded facts, absence of foreign-session facts and absence of architecture terms |

Each script writes synthetic documents under reserved fake chat IDs into your configured index and namespace, deletes them in a `finally` block and confirms the deletion; if cleanup cannot be confirmed it prints the IDs for manual removal. The repository does not record when they were last run.

Manual Telegram acceptance (a reproducible checklist, plus the limits of the one historical run): [docs/live_telegram_acceptance.md](docs/live_telegram_acceptance.md).

## Data handling and privacy

Recording is explicit: nothing is stored until someone sends `/start_listening`, and the bot's reply states that it saves participants' text messages until `/stop_listening`. There is no per-participant consent or opt-out; while a session is active every text message in the chat is recorded. Whoever runs the bot is responsible for informing participants and for complying with the rules that apply to them.

**Third-party services involved**

| Service | Receives |
| ------- | -------- |
| Telegram (Bot API) | every group update the bot is allowed to see; the bot's replies |
| OpenAI (or the endpoint in `OPENAI_BASE_URL`) | each recorded message, for embedding; at `/summary`, all messages of the session plus the fixed instruction, for chat generation |
| Pinecone | the message documents (below), in your index and namespace |
| PostHog (via Haystack telemetry) | anonymous usage statistics, unless disabled; see [Telemetry](#telemetry) |

**What is stored in Pinecone, per recorded message:** the embedding vector; the document content, which is `[UTC timestamp] display name (@username): text`; and metadata `source` (`telegram`), `schema_version`, `chat_id`, `message_id`, `user_id`, `session_id`, `author_name`, `sent_at` (UTC, ISO 8601) and `username` (only if the account has one). Text, names, usernames and numeric Telegram user IDs identify individual people.

**Session state** (chat ID, session ID, start time, who started it, message count) lives **in process memory only**. A restart loses the active and latest-completed session records: the bot can no longer summarize those sessions, but their documents **remain in Pinecone**.

**Retention.**

- Stored documents have no TTL and no automatic or bot-driven deletion.
- Telegram edits and deletions are not synchronized: an edited or deleted message stays in Pinecone as originally recorded and still appears in summaries.
- The only deletion code in the repository is the smoke tests' cleanup of their own synthetic documents.

**Logs.** The application's own log lines contain no message text, prompts, summaries or credentials (bot-token leakage on a startup failure is covered by a test). Some warning and error lines include the Telegram chat ID. The log output of third-party libraries has not been audited.

**Manual purge.** With the same `.env`, using the project's own store factory and filter builder:

```python
from config import load_settings
from document_store import create_pinecone_document_store
from retrieval_filters import build_session_filter

store = create_pinecone_document_store(load_settings())

# one session (session_id is stored in every document's metadata)
store.delete_by_filter(build_session_filter(chat_id=-1001234567890, session_id="telegram-session-<id>"))

# everything stored for one chat
store.delete_by_filter({"field": "meta.chat_id", "operator": "==", "value": "-1001234567890"})
```

`delete_by_filter` loads the matches through the same filter query, which returns at most 1,000 documents, so repeat it until it returns `0`. `store.delete_all_documents()` empties the entire configured namespace. You can also delete by ID or namespace in the Pinecone console. The filters are checked against the store's filter validation offline; this snippet has not been run against a live index.

## Known limitations

**Summaries**

- **Sessions with 1,000 or more recorded messages are refused** for summarization (recording continues). The cap comes from the vector store's filter-query limit; see above.
- **A summary requires exact document-count consistency.** This is a deliberate refusal, not a partial result. *Temporary:* Pinecone is eventually consistent, so right after new messages a summary can be unavailable for a few seconds; the bot retries 3 times, then asks the user to repeat `/summary`. *Persistent:* more documents than counted are never retried and surface as a generic internal-error reply, with the details in the log.
- **The completeness gate checks the store against the bot's own count**, not against Telegram. A message that failed to index is not counted and so is absent from the summary without any refusal (the chat is warned, at most once per 10 minutes, that some messages may be missing).
- **Prompt and context-window limits are not calculated.** Up to 999 messages go into a single prompt with no token counting or truncation; an oversized prompt fails as a generic error.
- **Summary quality is not evaluated automatically.** Offline tests check the plumbing against a fake model; the live smoke test checks one small scenario.

**Capture**

- **Not recorded:** messages sent as an anonymous group admin or on behalf of a chat (`sender_chat`, including linked-channel posts), and messages from bots. Messages sent while the bot is down are lost.
- **Forwarded messages** are recorded under the forwarding user's name; the original author is not read. **Replies** are recorded as plain text without a link to the replied-to message.
- **Forum topics are not modeled.** A session is per chat, spans all topics, and no topic ID is stored or used when replying.
- **Edits and deletions are not reconciled** (see Data handling).

**State and operations**

- **The session registry is in memory** and is lost on restart; there is no persistent session database and no way to select an older session.
- **No history UI or export**, and **no retention or deletion policy** beyond the manual purge above.
- **No access control:** any member of the chat can start, stop and summarize.
- **Single process, long polling:** no webhook, no horizontal scaling.

## Development approach

The repository owner is the product owner and the final reviewer: they set the requirements, make the architecture decisions, and decide what is accepted. AI tools were used, openly, as assistants for implementation, testing, review and an independent audit of the codebase. They do not make product decisions and are not credited as authors. The audit's findings drove the reliability work above: the whole-session completeness gate, failure handling, test consolidation and offline CI.

Work follows a human-led loop:

```text
Specification → Implementation → Independent Review → Human Decision
```

## Roadmap

The README describes what exists. Planned work, ordered by priority, is in [docs/project_roadmap.md](docs/project_roadmap.md): a persistent session registry, a data retention and deletion lifecycle, long-session summarization, structured and evaluated summary output, and access controls.

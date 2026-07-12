# Project Roadmap

Technical plan for evolving the homework-stage Telegram bot into a team AI assistant. Product direction is described in the README section **«Дальнейшее масштабирование»**. Nothing in this document is implemented in the current MVP unless explicitly stated elsewhere.

## Current baseline

- `InMemorySessionStore` — one active and one latest completed session per chat
- Session registry is lost on process restart; Pinecone documents remain
- No arbitrary historical session selection
- Text-only ingestion; long polling; no external integrations

## Development stages

```text
Stage 1 — persistent session registry
Stage 2 — /sessions and historical session selection
Stage 3 — structured decisions and action items
Stage 4 — cross-session search and Q&A
Stage 5 — exports and external integrations
Stage 6 — rich content and voice
Stage 7 — privacy controls and production deployment
```

## 1. Persistent session registry

**Goal:** replace in-memory registry with durable storage and restart recovery.

| Option | Use case |
|--------|----------|
| SQLite | local MVP, single-process deployment |
| PostgreSQL | production, multi-worker, webhook mode |

**Planned schema fields:**

- `chat_id`, `session_id`, `status`
- `title`, `started_at`, `stopped_at`, `started_by`
- `message_count`, `created_at`, `updated_at`

**Technical tasks:**

- repository layer abstracting storage backend
- restore active/completed sessions on startup
- transactional consistency between registry writes and Pinecone document metadata
- migration path from current in-memory store

## 2. Session history and selection

**Goal:** summary for any past discussion, not only active or latest completed.

**Planned UX:**

- `/sessions` — paginated list of recent discussions
- inline Telegram buttons for quick selection
- filter by date range and title
- `/summary <session>` — re-fetch summary for a chosen session
- pin important meetings; archive old sessions

**Technical tasks:**

- session list API backed by persistent registry
- callback handlers for inline selection
- summary resolution by explicit `session_id`

## 3. Session naming and meeting management

**Planned capabilities:**

- `/start_listening Meeting title`
- auto-title from first N messages (LLM or heuristic)
- rename session post-hoc
- tags/categories; link to project, team, or client
- multiple parallel Telegram topics within one group (topic-scoped sessions)

## 4. Structured meeting intelligence

**Goal:** move from free-text summary to structured, machine-readable output.

**Planned fields:**

- topic, participants, positions
- decisions, action items, owners, deadlines
- unresolved questions, risks, blockers, dependencies
- fallback scenarios, links, contact channels

**Technical approach:**

- Pydantic models / structured LLM output
- store structured result separately from raw messages
- decision log, action-item registry
- overdue-task tracking
- diff between initial agreements and later changes

## 5. Search and Q&A

**Planned retrieval modes:**

- Q&A scoped to one selected session
- search across all discussions in a chat
- cross-session RAG
- example queries: when was a decision made, who owned a task, deadline changes, which meetings discussed a risk
- compare decisions across sessions; detect contradictions
- timeline of decisions and assignments

**Technical tasks:**

- query routing (session-scoped vs chat-wide vs cross-chat)
- metadata filters + semantic retrieval
- dedicated Q&A Haystack pipeline (see §11)

## 6. Export and integrations

**Planned formats:** Markdown, DOCX, PDF

**Planned delivery channels (not implemented):**

- email, Slack, Google Docs, Notion
- Jira, Trello
- corporate knowledge bases

**Planned automation:**

- send protocol after meeting stop (opt-in)
- create tasks from action items
- sync deadlines to calendar
- notify assignees

## 7. Rich content ingestion

Each content type requires a separate ingestion and validation pipeline:

- voice messages → transcription
- meeting audio recordings
- documents, photos with captions, links
- edited/deleted Telegram messages
- replies, threads, Telegram topics
- attachments bound to a specific session

Current MVP accepts **text only**.

## 8. Privacy and data lifecycle

**Planned controls:**

- retention policy for messages and embeddings
- `/delete_session`, `/delete_history`, per-chat purge
- role-based access; summary restricted to admins or participants
- audit log, encryption at rest, data minimization
- participant consent for recording
- sensitive-message exclusion
- enterprise data-residency requirements

## 9. Product UX

**Planned features:**

- auto-summary on `/stop_listening` (opt-in)
- scheduled interim summaries
- action-item reminders
- summary styles: brief, full protocol, decisions-only, action-items-only, risks/unresolved
- multilingual summaries; tone and detail settings
- feedback/rating; regenerate with refined instruction
- inline buttons reducing reliance on command memorization

## 10. Production runtime

**Goal:** move from homework long polling to operable production deployment.

| Area | Planned upgrade |
|------|-----------------|
| Ingress | webhook mode instead of long polling |
| Packaging | Docker; VPS or cloud deployment |
| Data | PostgreSQL backing store |
| Async | task queues, background workers |
| Resilience | rate limits, retries, dead-letter queue |
| Observability | health endpoint, metrics, structured logs, tracing, alerting |
| Ops | backups, horizontal scaling |
| Architecture | separate Telegram ingestion from AI processing |
| Cost | OpenAI/Pinecone spend controls; cache repeated summaries |
| Quality | load testing |

## 11. Haystack architecture evolution

**Planned specialized pipelines:**

- summary (current, to be extended)
- action-item extraction
- decision extraction
- Q&A / retrieval-augmented answers
- document ingestion (per content type)

**Planned platform capabilities:**

- pipeline routing by intent
- evaluation datasets and retrieval quality metrics
- prompt/version management
- Haystack component observability
- A/B testing of prompts and retrieval settings

## Alignment with README

| README (product) | This document (technical) |
|----------------|---------------------------|
| Current limitations | Current baseline |
| §1–§11 scaling directions | Matching sections with implementation tasks |
| §12 development stages | Development stages |
| User-facing scenarios | Commands, APIs, storage, pipelines |

All items above are **planned**. The shipped MVP implements indexing, retrieval, summarization, in-memory sessions, and Telegram command menu only.

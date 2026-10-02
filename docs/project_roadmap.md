# Roadmap

What comes next for the Telegram discussion summarizer. Nothing here is implemented; the [README](../README.md) describes what exists, including the [known limitations](../README.md#known-limitations) these items address.

## Current baseline

- Capture of text messages during an explicit listening session; deterministic document IDs; Pinecone storage with hard chat/session isolation.
- Whole-session, chronologically ordered summarization behind an exact completeness gate; refusal instead of partial results.
- In-memory session registry: one active and one latest completed session per chat, lost on restart.
- Offline suite and CI; live smoke tooling; one recorded live Telegram acceptance run on this build (2026-10-02, see the [record](live_telegram_acceptance.md)).

## P1: meaningful product evolution

### 1. Persistent session registry and restart recovery

Replace `InMemorySessionStore` (it already sits behind the `SessionStore` protocol) with durable storage, for example SQLite for a single process.

- Restore the active and latest completed sessions after a restart, so documents already in Pinecone stay reachable.
- Make recording and the stored count consistent with each other. Today a write that reaches Pinecone but is not counted makes a session permanently inconsistent.
- Open question: whether a restart should resume or close an interrupted active session.

### 2. Data retention and deletion lifecycle

Today stored documents live until someone purges them by hand.

- A retention period and automatic expiry of stored messages.
- A bot-driven deletion path for a session or a chat, replacing the manual purge snippet.
- Decide how Telegram edits and deletions reach the store, since they are not synchronized now.

### 3. Long-session summarization

Lift the 999-message limit without giving up completeness.

- Loading beyond the store's 1,000-result filter cap requires a different enumeration mechanism, for example the registry keeping the document IDs of a session and fetching by ID.
- Hierarchical (map-reduce) summarization with token-aware chunking, with the same fail-closed contract: if any chunk fails, there is no summary.
- Proactive prompt-size accounting, so an oversized request is refused with a clear message instead of failing in the model call.

### 4. Structured, evaluated summary output

- A typed schema for decisions, action items (owner, deadline), unresolved questions and the recommendation, instead of free text with a prompt-enforced layout.
- A small labeled evaluation set and rubric, so prompt changes can be compared rather than judged by eye.

### 5. Access controls

If the bot is used outside a single trusted group: restrict who can start, stop and summarize (for example group administrators), and who may add the bot to a chat.

## P2

- **Message semantics:** edits, replies, forwarded-author attribution and forum topics (a session per topic).
- **Anonymous and on-behalf-of senders:** decide whether and how to record `sender_chat` messages.
- **Exports and session history**, only if real users need them: list or select past sessions, export a summary.
- **Deployment hardening**, only if the bot is run for real users: webhook mode, containerization, structured logs and metrics.

## Maintenance

- Decide the fate of the retrieval pipeline and `RETRIEVAL_TOP_K`: keep them if a semantic-search feature is planned, otherwise remove them together with their smoke test.

## Deliberately not planned

Cross-session search and Q&A, third-party integrations (Slack, Jira, Notion and similar), voice transcription, scheduled summaries and reminders were in earlier drafts of this roadmap. They are removed because nothing in the project calls for them yet.

# Live Telegram Acceptance

## Run metadata

| Field | Value |
|-------|-------|
| Date | 2026-07-12 |
| Offline baseline commit | `45890f2` |
| Python | 3.12.10 |
| pyTelegramBotAPI | 4.34.0 |
| haystack-ai | 2.31.0 |
| OpenAI model | configured via environment |
| Embedding model | configured via environment |
| Pinecone index | configured via environment |
| Pinecone namespace | configured via environment |
| Dimension | 1536 |
| Metric | cosine |

## History

1. **Initial launch attempt** — startup PASS, но updates не доходили до процесса из-за operational/configuration confusion (в том числе опечатка в команде запуска на раннем этапе).
2. **Operational issue: 409 Conflict** — диагностирован второй одновременно работающий polling process на том же bot token. Это operational issue, не production-code defect.
3. **First successful functional run** — start/duplicate start, четыре indexed messages, grounded summary, stop count = 4. Summary после stop был недоступен (active-only resolution). Action item из четвертого сообщения не попал в итог.
4. **Targeted repair** — latest completed session registry, `/summary`, phrase aliases, `/status`, `/help`, command menu, prompt completeness rules.
5. **Final rerun** — полный acceptance PASS на repaired build.

## Prerequisites

| Check | Result |
|-------|--------|
| Test supergroup | PASS |
| Bot administrator | PASS |
| Command menu visible | PASS |
| Single polling process | PASS |

## Startup result

**PASS**

- Settings loaded
- Pinecone preflight completed
- Runtime assembled
- Command menu configured before polling
- Telegram polling started
- No startup ERROR/traceback

## Acceptance steps

| Step | Description | Result |
|------|-------------|--------|
| Command menu | Menu visible in group | **PASS** |
| Start UX | `/start_listening` expanded response | **PASS** |
| Duplicate start | Second start rejected | **PASS** |
| Active status | `/status` during active session | **PASS** |
| Active summary | `/summary` during active session | **PASS** |
| Action item completeness | Owner/deadline/contact channel in summary | **PASS** |
| Stop count | `/stop_listening` count = 4 | **PASS** |
| Completed summary | `/summary` after stop uses latest completed session | **PASS** |
| Completed status | `/status` after stop shows latest completed | **PASS** |
| Phrase alias | `Подведи итог` | **PASS** |
| Help | `/help` static instruction | **PASS** |

## Final message count

**4** (acceptance scenario)

## Summary grounding checks

| Check | Result |
|-------|--------|
| Positions and alternatives reflected | PASS |
| Final decision distinguished from proposals | PASS |
| Action item owner/deadline/channel | PASS |
| Unresolved questions or explicit absence | PASS |
| AI recommendation present and labeled | PASS |
| No internal IDs/architecture terms | PASS |

## Safe observability

Expected log format after successful summary:

```text
Summary completed: message_count=N source_count=M session_state=active|completed
```

Logs do not include chat ID, session ID, document IDs, summary text, prompts, or secrets.

## Shutdown result

**PASS** — process stopped without ERROR traceback after acceptance.

## Defects and fixes

| Issue | Classification | Resolution |
|-------|----------------|------------|
| Updates not received on first attempt | Operational/configuration | Verified admin + single process |
| Telegram 409 Conflict | Operational (duplicate polling) | Stop extra process, single bot instance |
| Summary unavailable after stop | Production gap | Latest completed session registry |
| Missing action item in summary | Prompt completeness gap | Strengthened summarization prompt |
| Limited UX | Product gap | Command menu, `/status`, `/help`, aliases |

## Overall verdict

**PASS**

Live Telegram acceptance confirms the production flow: command menu, listening lifecycle, count = 4, active and latest-completed summary, grounded action items, and safe observability.

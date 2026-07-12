"""Telegram bot production entry point."""

from __future__ import annotations

import logging

from runtime import build_runtime, configure_logging, run_polling

logger = logging.getLogger(__name__)


def main() -> None:
    """Load runtime dependencies and start Telegram polling."""
    configure_logging()
    try:
        runtime = build_runtime()
        run_polling(runtime.bot)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    main()

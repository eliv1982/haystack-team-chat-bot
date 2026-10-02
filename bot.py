"""Telegram bot production entry point."""

from __future__ import annotations

import logging

from config import ConfigurationError
from error_reporting import describe_exception
from pinecone_preflight import PineconePreflightError
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
    except (ConfigurationError, PineconePreflightError) as exc:
        # These messages are written by this project and name settings, never values.
        logger.error("Startup failed: %s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from None
    except Exception as exc:
        # Other exception messages may embed credentials (a Telegram network error
        # contains the bot token in its URL). Neither the log record nor the final
        # traceback may show them, so log only a safe description and exit without
        # re-raising the original exception.
        logger.error("Startup failed: %s", describe_exception(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

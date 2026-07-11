"""Telegram bot entry point. Polling and handlers are not wired yet."""

from __future__ import annotations

from config import Settings


def create_bot(settings: Settings):
    """Create a Telegram bot instance. Handler wiring is deferred to a later stage."""
    raise NotImplementedError("Telegram bot setup is not implemented yet.")


def main() -> None:
    """Application entry point. Polling is not started until a later stage."""
    raise NotImplementedError("Bot polling is not implemented yet.")


if __name__ == "__main__":
    main()

"""Telegram bot entry point (Stage 1 placeholder)."""

from __future__ import annotations

from config import Settings


def create_bot(settings: Settings):
    """Create a Telegram bot instance. Implementation deferred to a later stage."""
    raise NotImplementedError("Telegram bot setup is not implemented yet.")


def main() -> None:
    """Application entry point. Polling is not started in Stage 1."""
    raise NotImplementedError("Bot polling is not implemented yet.")


if __name__ == "__main__":
    main()

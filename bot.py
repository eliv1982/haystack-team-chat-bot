"""Telegram bot entry point. Handlers exist; runtime wiring and polling are not connected."""

from __future__ import annotations

from config import Settings


def create_bot(settings: Settings):
    """Create a configured Telegram bot. Runtime dependency wiring is deferred."""
    raise NotImplementedError("Telegram bot runtime wiring is not implemented yet.")


def main() -> None:
    """Application entry point. Polling is not started until runtime wiring is added."""
    raise NotImplementedError("Bot polling is not implemented yet.")


if __name__ == "__main__":
    main()

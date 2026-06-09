"""One-time Google connection: ``python -m albert_agent.google_auth``.

Run this once to grant the assistant READ-ONLY access to your Gmail and Google
Calendar. It opens a browser for consent and caches a refresh token in
``token.json`` so you never have to do it again; afterwards the app and the
tools refresh the token silently. ``--status`` just reports whether a usable
token already exists, and ``--logout`` deletes it.
"""

from __future__ import annotations

import argparse

from . import config
from .google_client import GoogleError, authenticate, is_connected


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect Google (read-only).")
    parser.add_argument("--status", action="store_true",
                        help="only report whether a token already exists")
    parser.add_argument("--logout", action="store_true",
                        help="delete the cached token.json")
    args = parser.parse_args()

    if args.logout:
        config.GOOGLE_TOKEN_FILE.unlink(missing_ok=True)
        print("✅ Disconnected — token.json removed.")
        return

    if args.status:
        print("✅ Connected." if is_connected()
              else "⚠️  Not connected. Run `python -m albert_agent.google_auth`.")
        return

    try:
        authenticate()
    except GoogleError as exc:
        print(f"⚠️  {exc}")
        raise SystemExit(1)
    print("✅ Connected. Read-only Gmail + Calendar access is now available.")


if __name__ == "__main__":
    main()

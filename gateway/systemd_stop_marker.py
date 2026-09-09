"""Write the planned-stop marker used by systemd before SIGTERM.

This helper is intentionally tiny and profile-local: systemd invokes it with
the current gateway MAINPID, then sends the unit's configured KillSignal.
The gateway consumes the marker and exits cleanly, while unexpected signals
retain the existing crash-recovery behavior.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    try:
        pid = int(args[0])
    except (TypeError, ValueError):
        return 2
    if pid <= 0:
        return 2
    try:
        from gateway.status import write_planned_stop_marker
        return 0 if write_planned_stop_marker(pid) else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

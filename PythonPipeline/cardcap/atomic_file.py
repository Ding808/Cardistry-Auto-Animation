"""Bounded atomic replacement for a single writer and concurrent Windows readers."""
from __future__ import annotations

import os
from pathlib import Path
import time

# Engineering backoff, not a data/model threshold: at most 1 s of sleep per write.
WINDOWS_REPLACE_RETRIES = 20
WINDOWS_REPLACE_DELAY_SECONDS = 0.05
_WINDOWS_SHARING_ERRORS = frozenset((5, 32, 33))  # access, sharing, lock violation


def replace_with_windows_sharing_retry(temporary: Path, destination: Path) -> None:
    """Replace atomically; never delete the old destination to make replacement work.

    ERROR_ACCESS_DENIED also covers permanent access failures, so it is retried
    only on Windows and within the same fixed bound. The final original error is
    raised. The unpublished temporary is removed on failure where possible.
    """
    try:
        for attempt in range(WINDOWS_REPLACE_RETRIES + 1):
            try:
                os.replace(temporary, destination)
                return
            except OSError as error:
                if (os.name != "nt" or getattr(error, "winerror", None) not in _WINDOWS_SHARING_ERRORS
                        or attempt == WINDOWS_REPLACE_RETRIES):
                    raise
                time.sleep(WINDOWS_REPLACE_DELAY_SECONDS)
    except BaseException:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            # Cleanup failure must not mask the actual publication failure.
            pass
        raise

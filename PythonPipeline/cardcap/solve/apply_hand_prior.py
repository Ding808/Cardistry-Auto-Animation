"""Retired population-prior exporter; existing research artifacts remain intact.

No legacy scaling path remains. Use the normal exporter with explicit subject
measurement provenance or unknown metric scale. Pre-retirement source is archived
in Saved/Validation/P0_Contract12/scale_cards/before.
"""
from __future__ import annotations
import sys

RETIREMENT_REASON = (
    "The population hand-prior exporter is retired. No files were read or changed. "
    "Use the normal exporter with an explicit subject measurement and provenance, "
    "or preserve relative model geometry with metric scale unknown. Historical assets remain unchanged."
)

def apply_hand_prior(*args, **kwargs):
    raise RuntimeError(RETIREMENT_REASON)

def transform_hand_capture(*args, **kwargs):
    raise RuntimeError(RETIREMENT_REASON)

def main(argv=None):
    print(RETIREMENT_REASON, file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

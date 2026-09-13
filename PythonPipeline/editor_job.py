"""Hidden editor worker. See cardcap.editor_job for the request/status contract."""
from cardcap.editor_job import main

if __name__ == "__main__":
    raise SystemExit(main())

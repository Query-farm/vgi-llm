"""Pytest bootstrap: put the repo root on ``sys.path``.

Its mere presence lets the tests ``import tests.fake_provider`` and
``import vgi_llm`` without depending on an installed-package layout.

It also forces the framework's per-group storage to a process-local in-memory
SQLite (``VGI_WORKER_SHARED_STORAGE=memory``). The default ``sqlite`` backend is
a shared file, and ``pytest -n auto`` runs each test in a separate xdist worker
process — they would otherwise contend on that one file and raise
``sqlite3.OperationalError: database is locked`` nondeterministically.
"""

import os

os.environ.setdefault("VGI_WORKER_SHARED_STORAGE", "memory")

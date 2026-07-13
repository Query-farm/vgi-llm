"""Pytest bootstrap: put the repo root on ``sys.path``.

Its mere presence lets the tests ``import tests.fake_provider`` and
``import vgi_llm`` without depending on an installed-package layout.
"""

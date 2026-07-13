"""Pytest bootstrap: put the repo root on ``sys.path``.

Its mere presence lets the tests ``import tests.fake_provider`` and
``import vgi_aisql`` without depending on an installed-package layout.
"""

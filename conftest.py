"""
pytest configuration for pdb_dpi test suite.

Registers the ``network`` marker used by ``TestTable3EndToEnd`` and wires up the
``--run-network`` command-line flag that opts those tests in.  Without the flag,
all ``@pytest.mark.network`` tests are automatically skipped so that the core
formula tests can run offline (e.g. in CI).

Usage
-----
Run the full suite including PDB downloads::

    pytest --run-network

Run only the network tests::

    pytest --run-network -m network

Run without network tests (default)::

    pytest
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that require network access (PDB downloads from RCSB)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: marks tests that download files from RCSB or other remote servers",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(
            reason="Network tests are opt-in; re-run with --run-network to enable"
        )
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)

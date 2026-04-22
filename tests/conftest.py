# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

import os

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires model loading)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless activated — tests opt out of the blanket
    # skip when their own body fixture (e.g. ``_skip_if_not_configured``) can
    # decide whether to run based on env vars, by not declaring ``--server-url``
    # as their activation mechanism. In practice, we skip by default and let
    # env-var-gated tests (``QWEN3_CONCURRENT_TEST_URL``, etc.) or an explicit
    # ``--server-url`` opt in.
    server_url_provided = config.getoption("--server-url") != "http://localhost:8000"
    env_integration_activated = any(
        os.environ.get(var)
        for var in (
            "QWEN3_CONCURRENT_TEST_URL",
            "THINKING_BUDGET_TEST_MODEL",
            "VLLM_MLX_INTEGRATION_MODEL",
            "VLLM_MLX_INTEGRATION",
        )
    )
    if not (server_url_provided or env_integration_activated):
        skip_integration = pytest.mark.skip(
            reason=(
                "Integration tests require --server-url or an env-var activation "
                "(e.g. QWEN3_CONCURRENT_TEST_URL, THINKING_BUDGET_TEST_MODEL)"
            )
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def server_url(request):
    """Get server URL from command line."""
    return request.config.getoption("--server-url")

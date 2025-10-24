import pytest
import asyncio
import sys

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-web", action="store_true", default=False, help="run web tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "web: mark test as web test")
    config.addinivalue_line("markers", "asyncio: mark test as async")

def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_web = pytest.mark.skip(reason="need --run-web option to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "web" in item.keywords and not config.getoption("--run-web"):
            item.add_marker(skip_web)

# Add asyncio event loop support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

pytest_plugins = ("pytest_asyncio",)

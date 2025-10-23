import pytest

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-web", action="store_true", default=False, help="run web tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "web: mark test as web test")

def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_web = pytest.mark.skip(reason="need --run-web option to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "web" in item.keywords and not config.getoption("--run-web"):
            item.add_marker(skip_web)


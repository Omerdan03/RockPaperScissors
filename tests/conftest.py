"""Shared fixtures for Selenium e2e tests."""

import threading
import time

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

TEST_PORT = 5099
BASE_URL = f"http://localhost:{TEST_PORT}"


def pytest_addoption(parser):
    parser.addoption(
        "--headed", action="store_true", default=False,
        help="Run browser in headed (visible) mode instead of headless",
    )


@pytest.fixture(scope="session")
def server():
    """Start the Flask app in a background thread for the test session."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app import app

    thread = threading.Thread(
        target=lambda: app.run(port=TEST_PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    # Wait for server to be ready
    import urllib.request
    for _ in range(20):
        try:
            urllib.request.urlopen(BASE_URL)
            break
        except Exception:
            time.sleep(0.25)
    yield BASE_URL


@pytest.fixture
def driver(request):
    """Create a Chrome WebDriver instance (headless by default, --headed to show)."""
    options = Options()
    if not request.config.getoption("--headed"):
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        d = webdriver.Chrome(service=service, options=options)
    except Exception:
        # Fallback: expect chromedriver on PATH
        d = webdriver.Chrome(options=options)

    d.implicitly_wait(3)
    yield d
    d.quit()


class GameHelper:
    """Helper to drive the game UI via Selenium."""

    def __init__(self, driver, base_url):
        self.driver = driver
        self.url = base_url
        self.wait = WebDriverWait(driver, 8)

    # ── navigation ──

    def open_local(self):
        self.driver.get(f"{self.url}/local")

    # ── setup helpers ──

    def start_game(self, size="8"):
        """Click Start on config screen, arrive at P1 transition."""
        select = self.driver.find_element(By.ID, "board-size")
        for opt in select.find_elements(By.TAG_NAME, "option"):
            if opt.get_attribute("value") == str(size):
                opt.click()
                break
        self.driver.find_element(By.ID, "start-btn").click()
        self.wait.until(EC.visibility_of_element_located((By.ID, "transition-screen")))

    def click_ready(self):
        btn = self.wait.until(EC.element_to_be_clickable((By.ID, "transition-btn")))
        btn.click()
        self.wait.until(EC.visibility_of_element_located((By.ID, "game-screen")))

    def place_setup_pieces(self, player_num):
        """Place flag + 2 bombs for the given player in their zone."""
        zone = f".cell.zone-p{player_num}.setup-target"

        # Place flag (first palette item)
        self._click_first_palette_item()
        self._click_first_target(zone)

        # Bomb is auto-selected after flag is placed; place 2
        for _ in range(2):
            self._click_first_target(zone)

        # Click Done Placing
        done = self.wait.until(EC.element_to_be_clickable((By.ID, "done-setup-btn")))
        done.click()
        time.sleep(0.3)

    def complete_both_setups(self):
        """Run through both players' setups and arrive at P1 play transition."""
        self.start_game()
        for p in (1, 2):
            self.click_ready()
            self.place_setup_pieces(p)

    def enter_play_phase(self):
        """Full flow: open local, complete setups, click Ready for P1 turn."""
        self.open_local()
        self.complete_both_setups()
        self.click_ready()  # P1's first turn

    # ── private ──

    def _click_first_palette_item(self):
        items = self.wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "#setup-palette .palette-item")
            )
        )
        items[0].click()
        time.sleep(0.15)

    def _click_first_target(self, selector):
        time.sleep(0.3)
        cells = self.driver.find_elements(By.CSS_SELECTOR, selector)
        if not cells:
            raise AssertionError(f"No target cells found for {selector}")
        cells[0].click()
        time.sleep(0.3)


@pytest.fixture
def game(driver, server):
    """Provide a GameHelper wired to the running server + driver."""
    return GameHelper(driver, server)

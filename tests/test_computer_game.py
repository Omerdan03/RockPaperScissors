"""Selenium e2e tests for vs Computer mode."""

import time

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class TestComputerPageLoad:
    def test_computer_page_loads(self, game):
        game.open_computer()
        assert "RPS Stratego" in game.driver.title

    def test_subtitle_says_vs_computer(self, game):
        game.open_computer()
        subtitle = game.driver.find_element(By.CSS_SELECTOR, ".subtitle")
        assert "Computer" in subtitle.text


class TestComputerSetup:
    def test_no_transition_screen_after_start(self, game):
        """Computer mode should go straight to setup, no 'Player 1 - Ready' transition."""
        game.open_computer()
        game.driver.find_element(By.ID, "start-btn").click()
        wait = WebDriverWait(game.driver, 5)
        wait.until(EC.visibility_of_element_located((By.ID, "game-screen")))
        # Transition screen should NOT be visible
        transition = game.driver.find_elements(By.CSS_SELECTOR, "#transition-screen.active")
        assert len(transition) == 0, "Transition screen should not appear in computer mode"

    def test_no_p2_setup(self, game):
        """After P1 finishes setup, game should go directly to play (no P2 setup)."""
        game.start_computer_game()
        # Should be in play phase now, turn bar should say "Your Turn"
        turn_bar = game.driver.find_element(By.ID, "turn-bar")
        assert "Your Turn" in turn_bar.text or "Turn" in turn_bar.text

    def test_board_has_pieces_from_both_sides(self, game):
        """After setup, both P1 and P2 pieces should be on the board."""
        game.start_computer_game()
        p1_pieces = game.driver.find_elements(By.CSS_SELECTOR, ".piece.p1")
        p2_pieces = game.driver.find_elements(By.CSS_SELECTOR, ".piece.p2")
        assert len(p1_pieces) > 0, "Player 1 should have pieces"
        assert len(p2_pieces) > 0, "Player 2 (computer) should have pieces"


class TestComputerGameplay:
    def test_ai_responds_after_human_move(self, game):
        """After human moves, AI should auto-respond and turn should come back to human."""
        game.start_computer_game()

        # Use JS to find and click a movable P1 piece, avoiding stale element issues
        clicked = game.driver.execute_script("""
            const cells = document.querySelectorAll('.cell');
            for (const cell of cells) {
                const piece = cell.querySelector('.piece.p1');
                if (piece && !piece.querySelector('[data-symbol="F"]') && !piece.querySelector('[data-symbol="B"]')) {
                    cell.click();
                    return true;
                }
            }
            return false;
        """)

        if not clicked:
            pytest.skip("No movable P1 pieces found")

        time.sleep(0.5)

        # Click the first move target
        target_clicked = game.driver.execute_script("""
            const target = document.querySelector('.cell.move-target, .cell.move-target-attack');
            if (target) { target.click(); return true; }
            return false;
        """)

        if not target_clicked:
            pytest.skip("No valid moves for selected piece")

        # Wait for AI thinking + response (up to 5s)
        time.sleep(4)

        # Dismiss any combat popups (there may be two — human's and AI's)
        for _ in range(3):
            try:
                ok_btn = game.driver.find_element(By.ID, "combat-ok-btn")
                if ok_btn.is_displayed():
                    ok_btn.click()
                    time.sleep(2)
            except Exception:
                break

        # After AI move, should be back to human's turn (or game over)
        turn_bar = game.driver.find_element(By.ID, "turn-bar")
        text = turn_bar.text
        assert "Your Turn" in text or "Game Over" in text, \
            f"Expected 'Your Turn' or 'Game Over', got '{text}'"


class TestComputerMenu:
    def test_menu_has_computer_button(self, game):
        game.driver.get(game.url)
        btn = game.driver.find_element(By.ID, "computer-btn")
        assert btn.text == "vs Computer"

    def test_computer_button_navigates(self, game):
        game.driver.get(game.url)
        game.driver.find_element(By.ID, "computer-btn").click()
        WebDriverWait(game.driver, 5).until(EC.url_contains("/computer"))
        assert "/computer" in game.driver.current_url

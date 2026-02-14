"""Selenium tests for the local game flow, character sprites, and combat animation."""

import time

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ── Setup & Navigation ────────────────────────────────────────────────────────


class TestGameCreation:
    def test_local_page_loads(self, game):
        game.open_local()
        assert "RPS Stratego" in game.driver.title

    def test_start_shows_transition(self, game):
        game.open_local()
        game.start_game()
        text = game.driver.find_element(By.ID, "transition-title").text
        assert "Player 1" in text and "Setup" in text

    def test_config_preview_updates(self, game):
        game.open_local()
        preview = game.driver.find_element(By.ID, "piece-preview")
        assert preview.text  # should have content after page load


class TestPlayerSetup:
    def test_player1_can_place_flag(self, game):
        game.open_local()
        game.start_game()
        game.click_ready()

        # Palette should show flag & bomb
        items = game.driver.find_elements(By.CSS_SELECTOR, "#setup-palette .palette-item")
        assert len(items) >= 1

        # Place flag
        items[0].click()
        cells = game.driver.find_elements(By.CSS_SELECTOR, ".cell.zone-p1.setup-target")
        assert len(cells) > 0
        cells[0].click()
        time.sleep(0.4)

        # Verify a character piece appeared
        pieces = game.driver.find_elements(By.CSS_SELECTOR, ".piece .char")
        assert len(pieces) > 0

    def test_player2_can_place_pieces(self, game):
        """Regression: Player 2 placement was broken when token mapped only to P1."""
        game.open_local()
        game.start_game()
        game.click_ready()
        game.place_setup_pieces(1)  # finish P1

        # Now at Player 2 transition
        game.click_ready()

        # Player 2 zone targets should exist (rows 0-1)
        targets = game.driver.find_elements(By.CSS_SELECTOR, ".cell.zone-p2.setup-target")
        assert len(targets) > 0, "Player 2 should see setup-target cells in their zone"

        # Place flag
        items = game.driver.find_elements(By.CSS_SELECTOR, "#setup-palette .palette-item")
        assert len(items) >= 1
        items[0].click()
        targets[0].click()
        time.sleep(0.4)

        # Verify piece appeared with visible scroll (own piece)
        own = game.driver.find_elements(By.CSS_SELECTOR, '[data-scroll="visible"]')
        assert len(own) > 0, "Player 2's own pieces should show visible scrolls"

    def test_full_setup_reaches_play(self, game):
        game.open_local()
        game.complete_both_setups()
        # Should be at transition for P1's first turn
        title = game.driver.find_element(By.ID, "transition-title").text
        assert "Player 1" in title


# ── Character Sprite Rendering ────────────────────────────────────────────────


class TestCharacterSprites:
    def test_pieces_render_as_characters(self, game):
        game.enter_play_phase()
        chars = game.driver.find_elements(By.CSS_SELECTOR, ".piece .char")
        assert len(chars) > 0, "Board pieces should contain .char elements"

    def test_own_pieces_have_visible_scrolls(self, game):
        game.enter_play_phase()
        visible = game.driver.find_elements(By.CSS_SELECTOR, '[data-scroll="visible"]')
        assert len(visible) > 0, "Own pieces should have data-scroll=visible"

    def test_enemy_pieces_have_hidden_scrolls(self, game):
        game.enter_play_phase()
        hidden = game.driver.find_elements(By.CSS_SELECTOR, '[data-scroll="hidden"]')
        assert len(hidden) > 0, "Enemy pieces should have data-scroll=hidden"

    def test_scroll_symbols_valid(self, game):
        game.enter_play_phase()
        scrolls = game.driver.find_elements(By.CSS_SELECTOR, '[data-scroll="visible"] .scroll')
        assert len(scrolls) > 0
        for s in scrolls:
            sym = s.get_attribute("data-symbol")
            assert sym in ("R", "P", "S", "F", "B"), f"Bad symbol: {sym}"

    def test_idle_animation_applied(self, game):
        game.enter_play_phase()
        char = game.driver.find_element(By.CSS_SELECTOR, ".piece .char")
        anim = char.value_of_css_property("animation-name")
        assert "char-idle" in anim

    def test_staggered_animation_delays(self, game):
        game.enter_play_phase()
        chars = game.driver.find_elements(By.CSS_SELECTOR, ".piece .char")
        delays = {c.value_of_css_property("animation-delay") for c in chars[:6]}
        # With staggering, we expect more than one unique delay value
        assert len(delays) > 1, "Chars should have staggered animation delays"

    def test_piece_shape_is_rounded_square(self, game):
        game.enter_play_phase()
        piece = game.driver.find_element(By.CSS_SELECTOR, ".piece")
        radius = piece.value_of_css_property("border-radius")
        assert "8px" in radius, f"Piece should have rounded-square border-radius, got {radius}"


# ── Combat Animation ──────────────────────────────────────────────────────────


class TestCombatOverlayStructure:
    def test_arena_html_structure(self, game):
        game.open_local()
        arena = game.driver.find_element(By.CSS_SELECTOR, "#combat-overlay .combat-arena")
        chars = arena.find_elements(By.CSS_SELECTOR, ".combat-char")
        assert len(chars) == 2
        vs = arena.find_element(By.CSS_SELECTOR, ".combat-vs-text")
        # Use textContent since the overlay is hidden (display:none) and .text returns ''
        assert game.driver.execute_script("return arguments[0].textContent", vs) == "VS"

    def test_combat_chars_have_scroll_and_char(self, game):
        game.open_local()
        for cid in ("combat-atk", "combat-def"):
            el = game.driver.find_element(By.ID, cid)
            assert el.find_element(By.CSS_SELECTOR, ".char")
            assert el.find_element(By.CSS_SELECTOR, ".scroll")


class TestCombatAnimation:
    """Invoke showCombatPopup via JS to test the 5-phase animation."""

    def _trigger_combat(self, game, attacker="rock", defender="scissors",
                        outcome="win", action="combat"):
        game.driver.execute_script(f"""
            showCombatPopup({{
                action: '{action}',
                attacker: '{attacker}',
                defender: '{defender}',
                outcome: '{outcome}'
            }}, 1);
        """)

    def test_overlay_activates(self, game):
        game.enter_play_phase()
        self._trigger_combat(game)
        wait = WebDriverWait(game.driver, 3)
        overlay = wait.until(EC.visibility_of_element_located((By.ID, "combat-overlay")))
        assert overlay.is_displayed()

    def test_phase1_enter_animations(self, game):
        game.enter_play_phase()
        self._trigger_combat(game)
        time.sleep(0.2)
        atk = game.driver.find_element(By.ID, "combat-atk")
        defn = game.driver.find_element(By.ID, "combat-def")
        assert "enter-left" in atk.get_attribute("class")
        assert "enter-right" in defn.get_attribute("class")

    def test_phase2_taunt(self, game):
        game.enter_play_phase()
        self._trigger_combat(game)
        time.sleep(0.5)
        atk = game.driver.find_element(By.ID, "combat-atk")
        assert "taunt" in atk.get_attribute("class")

    def test_phase3_scroll_reveal(self, game):
        game.enter_play_phase()
        self._trigger_combat(game, "rock", "paper", "lose")
        time.sleep(1.2)
        atk_scroll = game.driver.find_element(By.CSS_SELECTOR, "#combat-atk .scroll")
        def_scroll = game.driver.find_element(By.CSS_SELECTOR, "#combat-def .scroll")
        assert atk_scroll.get_attribute("data-symbol") == "R"
        assert def_scroll.get_attribute("data-symbol") == "P"
        atk = game.driver.find_element(By.ID, "combat-atk")
        assert "scroll-show" in atk.get_attribute("class")

    def test_phase4_result_text_win(self, game):
        game.enter_play_phase()
        self._trigger_combat(game, "rock", "scissors", "win")
        time.sleep(1.8)
        text = game.driver.find_element(By.ID, "combat-result-text")
        assert "beats" in text.text
        assert "win" in text.get_attribute("class")
        atk = game.driver.find_element(By.ID, "combat-atk")
        assert "winner" in atk.get_attribute("class")

    def test_phase4_result_text_lose(self, game):
        game.enter_play_phase()
        self._trigger_combat(game, "scissors", "rock", "lose")
        time.sleep(1.8)
        text = game.driver.find_element(By.ID, "combat-result-text")
        assert "loses" in text.text
        assert "lose" in text.get_attribute("class")

    def test_phase4_result_text_draw(self, game):
        game.enter_play_phase()
        self._trigger_combat(game, "rock", "rock", "draw")
        time.sleep(1.8)
        text = game.driver.find_element(By.ID, "combat-result-text")
        assert "DRAW" in text.text
        assert "draw" in text.get_attribute("class")

    def test_phase4_bomb(self, game):
        game.enter_play_phase()
        self._trigger_combat(game, "rock", "bomb", "", "bomb")
        time.sleep(1.8)
        text = game.driver.find_element(By.ID, "combat-result-text")
        assert "BOMB" in text.text
        assert "bomb" in text.get_attribute("class")

    def test_phase4_flag_capture(self, game):
        game.enter_play_phase()
        self._trigger_combat(game, "rock", "flag", "", "capture_flag")
        time.sleep(1.8)
        text = game.driver.find_element(By.ID, "combat-result-text")
        assert "FLAG" in text.text

    def test_phase5_ok_button_appears(self, game):
        game.enter_play_phase()
        self._trigger_combat(game)
        time.sleep(2.4)
        ok_btn = game.driver.find_element(By.ID, "combat-ok-btn")
        assert ok_btn.is_displayed()

    def test_dismiss_resets_overlay(self, game):
        game.enter_play_phase()
        self._trigger_combat(game)
        time.sleep(2.4)
        game.driver.find_element(By.ID, "combat-ok-btn").click()
        time.sleep(0.3)
        overlay = game.driver.find_element(By.ID, "combat-overlay")
        assert "active" not in (overlay.get_attribute("class") or "")

    def test_player_colors_set(self, game):
        game.enter_play_phase()
        # Attacker is P1 (red), defender is P2 (blue)
        self._trigger_combat(game)
        time.sleep(0.3)
        atk_char = game.driver.find_element(By.CSS_SELECTOR, "#combat-atk .char")
        def_char = game.driver.find_element(By.CSS_SELECTOR, "#combat-def .char")
        atk_color = atk_char.value_of_css_property("color")
        def_color = def_char.value_of_css_property("color")
        # They should be different colors (P1 vs P2)
        assert atk_color != def_color, "Attacker and defender should have different colors"


# ── Mobile Viewport ───────────────────────────────────────────────────────────


class TestMobileViewport:
    def test_pieces_scale_down_on_mobile(self, game):
        game.driver.set_window_size(375, 667)
        game.enter_play_phase()
        piece = game.driver.find_element(By.CSS_SELECTOR, ".piece")
        width = piece.value_of_css_property("width")
        # On mobile the piece should be 40px (down from 50px)
        assert "40px" in width, f"Mobile piece should be 40px, got {width}"
        game.driver.set_window_size(1280, 900)  # restore

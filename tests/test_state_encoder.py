"""Tests for nn/state_encoder.py."""

import torch
from models import Game, Piece
from nn.state_encoder import encode_state, PIECE_INDEX


def _make_game(size=6):
    game = Game(size, "computer")
    # Clear the grid
    for r in range(size):
        for c in range(size):
            game.grid[r][c] = None
    game.phase = "play"
    game.current_player = 1
    game.turn_count = 0
    return game


class TestStateEncoder:
    def test_output_shape(self):
        game = _make_game()
        state = encode_state(game, 1)
        assert state.shape == (15, 6, 6)

    def test_empty_board_channel(self):
        game = _make_game()
        state = encode_state(game, 1)
        # All cells empty -> channel 11 should be all 1s
        assert state[11].sum().item() == 36

    def test_own_piece_encoding(self):
        game = _make_game()
        game.grid[5][0] = Piece("rock", 1)
        state = encode_state(game, 1)

        idx = PIECE_INDEX["rock"]
        assert state[idx, 5, 0] == 1.0
        # Should not be in empty channel
        assert state[11, 5, 0] == 0.0

    def test_enemy_revealed_piece(self):
        game = _make_game()
        p = Piece("paper", 2)
        p.revealed = True
        game.grid[0][3] = p
        state = encode_state(game, 1)

        idx = PIECE_INDEX["paper"]
        assert state[5 + idx, 0, 3] == 1.0

    def test_enemy_unknown_piece(self):
        game = _make_game()
        game.grid[1][2] = Piece("scissors", 2)
        state = encode_state(game, 1)

        # Channel 10 = unknown enemy
        assert state[10, 1, 2] == 1.0
        # Should NOT appear in revealed channels
        for i in range(5, 10):
            assert state[i, 1, 2] == 0.0

    def test_p2_board_flip(self):
        """Player 2's view should flip the board vertically."""
        game = _make_game()
        game.grid[0][0] = Piece("rock", 2)  # P2's piece at top-left

        state = encode_state(game, 2)

        # For P2, row 0 becomes row 5 (bottom), and this is an own piece
        idx = PIECE_INDEX["rock"]
        assert state[idx, 5, 0] == 1.0
        # Original position should be empty in the flipped view
        assert state[idx, 0, 0] == 0.0

    def test_turn_count_channel(self):
        game = _make_game()
        game.turn_count = 100
        state = encode_state(game, 1)

        expected = 100 / 200.0
        assert abs(state[12, 0, 0].item() - expected) < 1e-6
        # Broadcast: all cells same
        assert abs(state[12, 3, 3].item() - expected) < 1e-6

    def test_piece_count_channels(self):
        game = _make_game()
        game.grid[5][0] = Piece("rock", 1)
        game.grid[5][1] = Piece("paper", 1)
        game.grid[0][0] = Piece("scissors", 2)

        state = encode_state(game, 1)

        # Channel 13: own count / 12
        assert abs(state[13, 0, 0].item() - 2 / 12.0) < 1e-6
        # Channel 14: enemy count / 12
        assert abs(state[14, 0, 0].item() - 1 / 12.0) < 1e-6

    def test_all_piece_types_own(self):
        game = _make_game()
        for i, ptype in enumerate(["rock", "paper", "scissors", "flag", "bomb"]):
            game.grid[5][i] = Piece(ptype, 1)

        state = encode_state(game, 1)

        for i, ptype in enumerate(["rock", "paper", "scissors", "flag", "bomb"]):
            assert state[i, 5, i] == 1.0

    def test_channels_are_mutually_exclusive(self):
        """Each cell should only have a 1 in exactly one of channels 0-11."""
        game = _make_game()
        game.grid[5][0] = Piece("rock", 1)
        game.grid[0][0] = Piece("paper", 2)
        p = Piece("scissors", 2)
        p.revealed = True
        game.grid[0][1] = p

        state = encode_state(game, 1)

        for r in range(6):
            for c in range(6):
                # Channels 0-11 should sum to exactly 1
                total = state[0:12, r, c].sum().item()
                assert abs(total - 1.0) < 1e-6, f"Cell ({r},{c}) has sum {total}"

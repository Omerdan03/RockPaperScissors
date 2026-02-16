"""Tests for nn/action_encoder.py."""

import torch
from models import Game, Piece
from nn.action_encoder import move_to_action, action_to_move, get_valid_action_mask


def _make_game(size=6):
    game = Game(size, "computer")
    for r in range(size):
        for c in range(size):
            game.grid[r][c] = None
    game.phase = "play"
    game.current_player = 1
    return game


class TestMoveToAction:
    def test_round_trip_p1(self):
        """action_to_move(move_to_action(move)) should recover the original move for P1."""
        for fr in range(6):
            for fc in range(6):
                for di, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    tr, tc = fr + dr, fc + dc
                    if 0 <= tr < 6 and 0 <= tc < 6:
                        idx = move_to_action(fr, fc, tr, tc, 1, 6)
                        rfr, rfc, rtr, rtc = action_to_move(idx, 1, 6)
                        assert (rfr, rfc, rtr, rtc) == (fr, fc, tr, tc)

    def test_round_trip_p2(self):
        """Round-trip for P2 (with flipping)."""
        for fr in range(6):
            for fc in range(6):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    tr, tc = fr + dr, fc + dc
                    if 0 <= tr < 6 and 0 <= tc < 6:
                        idx = move_to_action(fr, fc, tr, tc, 2, 6)
                        rfr, rfc, rtr, rtc = action_to_move(idx, 2, 6)
                        assert (rfr, rfc, rtr, rtc) == (fr, fc, tr, tc)

    def test_action_range(self):
        """All action indices should be in [0, 144)."""
        for fr in range(6):
            for fc in range(6):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    tr, tc = fr + dr, fc + dc
                    if 0 <= tr < 6 and 0 <= tc < 6:
                        for player in [1, 2]:
                            idx = move_to_action(fr, fc, tr, tc, player, 6)
                            assert 0 <= idx < 144

    def test_unique_indices(self):
        """Each valid move for a player should map to a unique index."""
        indices = set()
        for fr in range(6):
            for fc in range(6):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    tr, tc = fr + dr, fc + dc
                    if 0 <= tr < 6 and 0 <= tc < 6:
                        idx = move_to_action(fr, fc, tr, tc, 1, 6)
                        assert idx not in indices, f"Duplicate index {idx}"
                        indices.add(idx)


class TestValidActionMask:
    def test_empty_board_no_actions(self):
        game = _make_game()
        mask = get_valid_action_mask(game, 1)
        assert mask.sum().item() == 0

    def test_single_piece_center(self):
        game = _make_game()
        game.grid[3][3] = Piece("rock", 1)
        game.current_player = 1
        mask = get_valid_action_mask(game, 1)

        # Center piece has 4 moves (up/down/left/right)
        assert mask.sum().item() == 4

    def test_corner_piece(self):
        game = _make_game()
        game.grid[0][0] = Piece("rock", 1)
        game.current_player = 1
        mask = get_valid_action_mask(game, 1)

        # Corner piece: only 2 valid directions (down and right)
        assert mask.sum().item() == 2

    def test_immovable_pieces_excluded(self):
        game = _make_game()
        game.grid[3][3] = Piece("flag", 1)
        game.grid[3][4] = Piece("bomb", 1)
        game.current_player = 1
        mask = get_valid_action_mask(game, 1)

        assert mask.sum().item() == 0

    def test_no_friendly_fire(self):
        game = _make_game()
        game.grid[3][3] = Piece("rock", 1)
        game.grid[3][4] = Piece("paper", 1)  # Friendly piece to the right
        game.current_player = 1
        mask = get_valid_action_mask(game, 1)

        # rock at (3,3): up, down, left are valid but right blocked by friendly
        # paper at (3,4): up, down, right valid but left blocked by friendly
        assert mask.sum().item() == 6

    def test_mask_matches_game_valid_moves(self):
        """Mask should have exactly same moves as game.get_valid_moves()."""
        game = _make_game()
        game.grid[2][2] = Piece("scissors", 1)
        game.grid[1][2] = Piece("rock", 2)  # Enemy above
        game.current_player = 1

        mask = get_valid_action_mask(game, 1)
        game_moves = game.get_valid_moves(2, 2)

        # Convert game moves to action indices
        game_indices = set()
        for m in game_moves:
            idx = move_to_action(2, 2, m["r"], m["c"], 1, 6)
            game_indices.add(idx)

        mask_indices = set(i for i in range(144) if mask[i])
        assert mask_indices == game_indices

    def test_p2_mask(self):
        """P2 mask should work with flipped coordinates."""
        game = _make_game()
        game.grid[1][1] = Piece("paper", 2)
        game.current_player = 2

        mask = get_valid_action_mask(game, 2)
        assert mask.sum().item() == 4  # Center-ish piece, all 4 directions valid

        # Verify round-trip: each masked action should decode to a valid move
        for idx in range(144):
            if mask[idx]:
                fr, fc, tr, tc = action_to_move(idx, 2, 6)
                assert game.grid[fr][fc] is not None
                assert game.grid[fr][fc].owner == 2

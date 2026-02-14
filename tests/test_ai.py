"""Unit tests for AI decision engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Game, Piece
from ai import pick_ai_move


class TestAiFlagCapture:
    def test_captures_adjacent_flag(self):
        """AI should prioritize capturing an adjacent flag."""
        game = Game(6, mode="computer")
        # Clear the board
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # Place AI rock at (2, 2) and enemy flag at (3, 2)
        game.grid[2][2] = Piece("rock", 2)
        game.grid[3][2] = Piece("flag", 1)

        move = pick_ai_move(game, 2)
        assert move is not None
        assert move["tr"] == 3 and move["tc"] == 2, "AI should move to capture the flag"

    def test_flag_priority_over_advance(self):
        """Flag capture should be chosen over advancing."""
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # AI rock at (2, 2), flag below, empty below-right
        game.grid[2][2] = Piece("rock", 2)
        game.grid[3][2] = Piece("flag", 1)

        move = pick_ai_move(game, 2)
        assert move["tr"] == 3 and move["tc"] == 2


class TestAiWinningAttacks:
    def test_attacks_revealed_weaker_piece(self):
        """AI should attack a revealed piece it can beat."""
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # AI rock at (2, 2), revealed scissors at (3, 2)
        game.grid[2][2] = Piece("rock", 2)
        enemy = Piece("scissors", 1)
        enemy.revealed = True
        game.grid[3][2] = enemy

        move = pick_ai_move(game, 2)
        assert move is not None
        assert move["tr"] == 3 and move["tc"] == 2

    def test_paper_beats_rock(self):
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        game.grid[2][2] = Piece("paper", 2)
        enemy = Piece("rock", 1)
        enemy.revealed = True
        game.grid[3][2] = enemy

        move = pick_ai_move(game, 2)
        assert move["tr"] == 3 and move["tc"] == 2


class TestAiAvoidsLosses:
    def test_avoids_revealed_stronger_piece(self):
        """AI should prefer other moves over attacking a piece that beats it."""
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # AI scissors at (2, 2), revealed rock at (3, 2), empty at (2, 3)
        game.grid[2][2] = Piece("scissors", 2)
        enemy = Piece("rock", 1)
        enemy.revealed = True
        game.grid[3][2] = enemy

        move = pick_ai_move(game, 2)
        assert move is not None
        # Should prefer any non-losing move
        assert not (move["tr"] == 3 and move["tc"] == 2), \
            "AI should avoid attacking a revealed stronger piece when other moves exist"

    def test_attacks_stronger_as_last_resort(self):
        """If no other moves, AI will attack a stronger piece."""
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # AI scissors cornered at (0, 0), only option is revealed rock at (1, 0)
        game.grid[0][0] = Piece("scissors", 2)
        enemy = Piece("rock", 1)
        enemy.revealed = True
        game.grid[1][0] = enemy
        # Block other direction with own piece
        game.grid[0][1] = Piece("bomb", 2)

        move = pick_ai_move(game, 2)
        assert move is not None
        assert move["tr"] == 1 and move["tc"] == 0


class TestAiAdvance:
    def test_prefers_advancing_toward_enemy(self):
        """AI (player 2) should prefer moving toward higher rows."""
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # AI rock in the middle, only empty cells around
        game.grid[2][3] = Piece("rock", 2)

        move = pick_ai_move(game, 2)
        assert move is not None
        # Advance moves (toward P1 territory = higher rows) should be preferred
        assert move["tr"] == 3, "AI should advance toward enemy territory"


class TestAiNoMoves:
    def test_returns_none_when_no_moves(self):
        """AI returns None when it has no movable pieces."""
        game = Game(6, mode="computer")
        game.grid = [[None] * 6 for _ in range(6)]
        game.phase = "play"
        game.current_player = 2

        # Only bombs and flags (not movable)
        game.grid[0][0] = Piece("flag", 2)
        game.grid[0][1] = Piece("bomb", 2)

        move = pick_ai_move(game, 2)
        assert move is None

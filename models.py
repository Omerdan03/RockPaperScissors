"""Game models: Piece, Game, and GameRoom classes."""

import random
import time


def generate_token() -> str:
    import string
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))


class Piece:
    def __init__(self, piece_type: str, owner: int):
        self.type = piece_type        # rock | paper | scissors | flag | bomb
        self.owner = owner            # 1 or 2
        self.revealed = False         # visible to both players once revealed

    def to_dict(self, viewer: int) -> dict:
        """Serialize piece, hiding type from opponent if not revealed."""
        can_see = (self.owner == viewer) or self.revealed
        return {
            "type": self.type if can_see else "unknown",
            "owner": self.owner,
            "revealed": self.revealed,
        }

    def is_movable(self) -> bool:
        return self.type in ("rock", "paper", "scissors")


class Game:
    def __init__(self, board_size: int, mode: str = "local"):
        self.size = board_size
        self.mode = mode
        self.grid: list[list[Piece | None]] = [
            [None] * board_size for _ in range(board_size)
        ]
        self.phase = "setup"          # setup | play | over
        self.setup_player = 1         # which player is setting up (local mode)
        self.current_player = 1
        self.turn_count = 0
        self.winner = None
        self.log: list[str] = []
        self.distribution = self._calc_distribution()
        self.last_combat = None       # last combat result for polling
        # Online concurrent setup tracking
        self.p1_setup_done = False
        self.p2_setup_done = False

    # ── piece distribution ──────────────────────────────────────────────────

    def _calc_distribution(self) -> dict[str, int]:
        total = self.size * 2
        remaining = total - 3  # minus 1 flag + 2 bombs
        base, rem = divmod(remaining, 3)
        dist = {"rock": base, "paper": base, "scissors": base, "flag": 1, "bomb": 2}
        types = ["rock", "paper", "scissors"]
        for _ in range(rem):
            dist[random.choice(types)] += 1
        return dist

    # ── deployment rows ─────────────────────────────────────────────────────

    def deploy_rows(self, player: int) -> list[int]:
        if player == 1:
            return [self.size - 2, self.size - 1]
        return [0, 1]

    # ── setup ───────────────────────────────────────────────────────────────

    def setup_remaining(self, player: int | None = None) -> dict[str, int]:
        """Count how many flag/bomb pieces the given player still needs to place."""
        if player is None:
            player = self.setup_player
        placed = {"flag": 0, "bomb": 0}
        for r in range(self.size):
            for c in range(self.size):
                p = self.grid[r][c]
                if p and p.owner == player and p.type in placed:
                    placed[p.type] += 1
        return {
            "flag": 1 - placed["flag"],
            "bomb": 2 - placed["bomb"],
        }

    def place_setup_piece(self, r: int, c: int, piece_type: str, player: int | None = None) -> dict:
        if self.phase != "setup":
            return {"error": "Not in setup phase"}
        if player is None:
            player = self.setup_player
        if piece_type not in ("flag", "bomb"):
            return {"error": "Can only manually place flag or bomb"}
        rows = self.deploy_rows(player)
        if r not in rows:
            return {"error": "Must place in your deployment zone"}
        if c < 0 or c >= self.size:
            return {"error": "Column out of bounds"}
        if self.grid[r][c] is not None:
            return {"error": "Cell already occupied"}

        remaining = self.setup_remaining(player)
        if remaining.get(piece_type, 0) <= 0:
            return {"error": f"No more {piece_type} pieces to place"}

        self.grid[r][c] = Piece(piece_type, player)
        return {"ok": True}

    def remove_setup_piece(self, r: int, c: int, player: int | None = None) -> dict:
        """Remove a piece placed during setup (allow undo)."""
        if self.phase != "setup":
            return {"error": "Not in setup phase"}
        if player is None:
            player = self.setup_player
        rows = self.deploy_rows(player)
        if r not in rows:
            return {"error": "Not in your deployment zone"}
        p = self.grid[r][c]
        if p is None:
            return {"error": "No piece there"}
        if p.owner != player:
            return {"error": "Not your piece"}
        if p.type not in ("flag", "bomb"):
            return {"error": "Can only remove manually placed pieces"}
        self.grid[r][c] = None
        return {"ok": True}

    def _fill_rps(self, player: int):
        """Fill remaining deployment cells for a player with RPS pieces."""
        dist = self._calc_distribution()
        pool = (
            ["rock"] * dist["rock"]
            + ["paper"] * dist["paper"]
            + ["scissors"] * dist["scissors"]
        )
        random.shuffle(pool)

        rows = self.deploy_rows(player)
        idx = 0
        for row in rows:
            for c in range(self.size):
                if self.grid[row][c] is None:
                    self.grid[row][c] = Piece(pool[idx], player)
                    idx += 1

    def auto_setup_player(self, player: int):
        """Randomly place 1 flag + 2 bombs, then fill with RPS pieces."""
        rows = self.deploy_rows(player)
        cells = [(r, c) for r in rows for c in range(self.size)]
        random.shuffle(cells)

        for piece_type, count in [("flag", 1), ("bomb", 2)]:
            for _ in range(count):
                r, c = cells.pop()
                self.grid[r][c] = Piece(piece_type, player)

        self._fill_rps(player)

    def finish_setup(self, player: int | None = None) -> dict:
        if player is None:
            player = self.setup_player

        remaining = self.setup_remaining(player)
        if remaining["flag"] > 0 or remaining["bomb"] > 0:
            return {"error": "Place all flag and bomb pieces first"}

        self._fill_rps(player)

        if self.mode == "computer":
            # Computer mode: auto-setup player 2 and go straight to play
            self.auto_setup_player(2)
            self.phase = "play"
            self.current_player = 1
            self.turn_count = 0
            self._add_log("Game started!")
            return {"next": "play"}
        elif self.mode == "online":
            # Mark this player as done
            if player == 1:
                self.p1_setup_done = True
            else:
                self.p2_setup_done = True

            if self.p1_setup_done and self.p2_setup_done:
                self.phase = "play"
                self.current_player = 1
                self.turn_count = 0
                self._add_log("Game started!")
                return {"next": "play", "bothDone": True}
            else:
                return {"next": "waiting", "bothDone": False}
        else:
            # Local mode: sequential setup
            if player == 1:
                self.setup_player = 2
                return {"next": "setup", "player": 2}
            else:
                self.phase = "play"
                self.current_player = 1
                self.turn_count = 0
                self._add_log("Game started!")
                return {"next": "play"}

    # ── moves ───────────────────────────────────────────────────────────────

    def get_valid_moves(self, r: int, c: int) -> list[dict]:
        p = self.grid[r][c]
        if not p or p.owner != self.current_player or not p.is_movable():
            return []
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                target = self.grid[nr][nc]
                if target is None or target.owner != self.current_player:
                    moves.append({"r": nr, "c": nc})
        return moves

    def make_move(self, fr: int, fc: int, tr: int, tc: int) -> dict:
        if self.phase != "play":
            return {"error": "Game is not in play phase"}

        piece = self.grid[fr][fc]
        if not piece or piece.owner != self.current_player:
            return {"error": "Not your piece"}
        if not piece.is_movable():
            return {"error": "This piece cannot move"}

        valid = self.get_valid_moves(fr, fc)
        if not any(m["r"] == tr and m["c"] == tc for m in valid):
            return {"error": "Invalid move"}

        defender = self.grid[tr][tc]

        if defender is None:
            # simple move
            self.grid[tr][tc] = piece
            self.grid[fr][fc] = None
            self._add_log(
                f"P{self.current_player} moved {piece.type} ({fr},{fc})->({tr},{tc})"
            )
            result = {"action": "move"}
            self.last_combat = None
        else:
            result = self._resolve_combat(fr, fc, tr, tc, piece, defender)

        if self.phase == "over":
            return result

        self.turn_count += 1
        self.current_player = 2 if self.current_player == 1 else 1

        # check if next player can move
        if not self._player_can_move(self.current_player):
            other = 2 if self.current_player == 1 else 1
            self._end_game(other, f"Player {self.current_player} has no movable pieces!")
            result["game_over"] = True
            result["winner"] = other

        return result

    def _resolve_combat(self, fr, fc, tr, tc, attacker, defender) -> dict:
        attacker.revealed = True
        defender.revealed = True

        # vs flag
        if defender.type == "flag":
            self.grid[tr][tc] = attacker
            self.grid[fr][fc] = None
            self._add_log(
                f"P{attacker.owner} {attacker.type} captured P{defender.owner}'s FLAG!"
            )
            self._end_game(attacker.owner, f"Player {attacker.owner} captured the flag!")
            result = {
                "action": "capture_flag",
                "attacker": attacker.type,
                "defender": "flag",
                "game_over": True,
                "winner": attacker.owner,
            }
            self.last_combat = result
            return result

        # vs bomb
        if defender.type == "bomb":
            self.grid[fr][fc] = None
            self.grid[tr][tc] = None
            self._add_log(
                f"P{attacker.owner} {attacker.type} hit a BOMB! Both destroyed."
            )
            result = {
                "action": "bomb",
                "attacker": attacker.type,
                "defender": "bomb",
            }
            self.last_combat = result
            return result

        # RPS
        outcome = self._rps(attacker.type, defender.type)
        if outcome == "win":
            self.grid[tr][tc] = attacker
            self.grid[fr][fc] = None
            self._add_log(
                f"{attacker.type} (P{attacker.owner}) beats {defender.type} (P{defender.owner})"
            )
        elif outcome == "lose":
            self.grid[fr][fc] = None
            self._add_log(
                f"{attacker.type} (P{attacker.owner}) loses to {defender.type} (P{defender.owner})"
            )
        else:
            # draw — both survive in place
            self._add_log(
                f"{attacker.type} vs {defender.type} — DRAW! Both revealed."
            )

        result = {
            "action": "combat",
            "attacker": attacker.type,
            "defender": defender.type,
            "outcome": outcome,
        }
        self.last_combat = result
        return result

    @staticmethod
    def _rps(a: str, d: str) -> str:
        if a == d:
            return "draw"
        wins = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}
        return "win" if (a, d) in wins else "lose"

    def _player_can_move(self, player: int) -> bool:
        for r in range(self.size):
            for c in range(self.size):
                p = self.grid[r][c]
                if p and p.owner == player and p.is_movable():
                    if self.get_valid_moves(r, c):
                        return True
        return False

    def _end_game(self, winner: int, msg: str):
        self.phase = "over"
        self.winner = winner
        # reveal everything
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c]:
                    self.grid[r][c].revealed = True
        self._add_log(msg)

    def _add_log(self, text: str):
        self.log.append(text)

    # ── serialization ───────────────────────────────────────────────────────

    def to_dict(self, viewer: int) -> dict:
        grid_data = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                p = self.grid[r][c]
                row.append(p.to_dict(viewer) if p else None)
            grid_data.append(row)

        d = {
            "size": self.size,
            "grid": grid_data,
            "phase": self.phase,
            "currentPlayer": self.current_player,
            "turnCount": self.turn_count,
            "winner": self.winner,
            "log": self.log[-20:],  # last 20 entries
            "distribution": self.distribution,
            "mode": self.mode,
            "lastCombat": self.last_combat,
        }

        if self.phase == "setup":
            if self.mode == "online":
                d["setupPlayer"] = viewer  # each player sets up their own
                d["setupRemaining"] = self.setup_remaining(viewer)
                d["p1SetupDone"] = self.p1_setup_done
                d["p2SetupDone"] = self.p2_setup_done
            else:
                d["setupPlayer"] = self.setup_player
                d["setupRemaining"] = self.setup_remaining(self.setup_player)

        return d


class GameRoom:
    def __init__(self, game_id: str, size: int, mode: str):
        self.id = game_id
        self.game = Game(size, mode)
        self.mode = mode  # "local" or "online"
        self.host_token = generate_token()
        self.guest_token: str | None = None
        self.player_tokens: dict[str, int] = {self.host_token: 1}
        self.created_at = time.time()
        self.last_activity = time.time()
        self.version = 0

    def touch(self):
        self.last_activity = time.time()

    def bump_version(self):
        self.version += 1
        self.touch()

    def player_for_token(self, token: str) -> int | None:
        return self.player_tokens.get(token)

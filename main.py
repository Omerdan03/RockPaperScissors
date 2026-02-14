"""RPS Stratego - A tactical board game server combining Stratego hidden information with RPS combat."""

import random
import json
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Game state (single game in memory) ──────────────────────────────────────

game = None


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
    def __init__(self, board_size: int):
        self.size = board_size
        self.grid: list[list[Piece | None]] = [
            [None] * board_size for _ in range(board_size)
        ]
        self.phase = "setup"          # setup | play | over
        self.setup_player = 1         # which player is setting up
        self.current_player = 1
        self.turn_count = 0
        self.winner = None
        self.log: list[str] = []
        self.distribution = self._calc_distribution()

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

    def setup_remaining(self) -> dict[str, int]:
        """Count how many flag/bomb pieces the setup_player still needs to place."""
        placed = {"flag": 0, "bomb": 0}
        for r in range(self.size):
            for c in range(self.size):
                p = self.grid[r][c]
                if p and p.owner == self.setup_player and p.type in placed:
                    placed[p.type] += 1
        return {
            "flag": 1 - placed["flag"],
            "bomb": 2 - placed["bomb"],
        }

    def place_setup_piece(self, r: int, c: int, piece_type: str) -> dict:
        if self.phase != "setup":
            return {"error": "Not in setup phase"}
        if piece_type not in ("flag", "bomb"):
            return {"error": "Can only manually place flag or bomb"}
        rows = self.deploy_rows(self.setup_player)
        if r not in rows:
            return {"error": "Must place in your deployment zone"}
        if c < 0 or c >= self.size:
            return {"error": "Column out of bounds"}
        if self.grid[r][c] is not None:
            return {"error": "Cell already occupied"}

        remaining = self.setup_remaining()
        if remaining.get(piece_type, 0) <= 0:
            return {"error": f"No more {piece_type} pieces to place"}

        self.grid[r][c] = Piece(piece_type, self.setup_player)
        return {"ok": True}

    def remove_setup_piece(self, r: int, c: int) -> dict:
        """Remove a piece placed during setup (allow undo)."""
        if self.phase != "setup":
            return {"error": "Not in setup phase"}
        rows = self.deploy_rows(self.setup_player)
        if r not in rows:
            return {"error": "Not in your deployment zone"}
        p = self.grid[r][c]
        if p is None:
            return {"error": "No piece there"}
        if p.owner != self.setup_player:
            return {"error": "Not your piece"}
        if p.type not in ("flag", "bomb"):
            return {"error": "Can only remove manually placed pieces"}
        self.grid[r][c] = None
        return {"ok": True}

    def finish_setup(self) -> dict:
        remaining = self.setup_remaining()
        if remaining["flag"] > 0 or remaining["bomb"] > 0:
            return {"error": "Place all flag and bomb pieces first"}

        # fill remaining cells with RPS
        dist = self._calc_distribution()
        pool = (
            ["rock"] * dist["rock"]
            + ["paper"] * dist["paper"]
            + ["scissors"] * dist["scissors"]
        )
        random.shuffle(pool)

        rows = self.deploy_rows(self.setup_player)
        idx = 0
        for row in rows:
            for c in range(self.size):
                if self.grid[row][c] is None:
                    self.grid[row][c] = Piece(pool[idx], self.setup_player)
                    idx += 1

        if self.setup_player == 1:
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
            return {
                "action": "capture_flag",
                "attacker": attacker.type,
                "defender": "flag",
                "game_over": True,
                "winner": attacker.owner,
            }

        # vs bomb
        if defender.type == "bomb":
            self.grid[fr][fc] = None
            self.grid[tr][tc] = None
            self._add_log(
                f"P{attacker.owner} {attacker.type} hit a BOMB! Both destroyed."
            )
            return {
                "action": "bomb",
                "attacker": attacker.type,
                "defender": "bomb",
            }

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

        return {
            "action": "combat",
            "attacker": attacker.type,
            "defender": defender.type,
            "outcome": outcome,
        }

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

        return {
            "size": self.size,
            "grid": grid_data,
            "phase": self.phase,
            "setupPlayer": self.setup_player if self.phase == "setup" else None,
            "setupRemaining": self.setup_remaining() if self.phase == "setup" else None,
            "currentPlayer": self.current_player,
            "turnCount": self.turn_count,
            "winner": self.winner,
            "log": self.log[-20:],  # last 20 entries
            "distribution": self.distribution,
        }


# ── API routes ──────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/new-game", methods=["POST"])
def new_game():
    global game
    data = request.get_json(force=True)
    size = data.get("size", 8)
    if size < 4 or size > 16:
        return jsonify({"error": "Board size must be between 4 and 16"}), 400
    game = Game(size)
    return jsonify({"ok": True, "size": size, "distribution": game.distribution})


@app.route("/api/state")
def get_state():
    if game is None:
        return jsonify({"error": "No game in progress"}), 400
    viewer = int(request.args.get("viewer", 1))
    return jsonify(game.to_dict(viewer))


@app.route("/api/setup/place", methods=["POST"])
def setup_place():
    if game is None:
        return jsonify({"error": "No game in progress"}), 400
    data = request.get_json(force=True)
    result = game.place_setup_piece(data["r"], data["c"], data["type"])
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/setup/remove", methods=["POST"])
def setup_remove():
    if game is None:
        return jsonify({"error": "No game in progress"}), 400
    data = request.get_json(force=True)
    result = game.remove_setup_piece(data["r"], data["c"])
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/setup/done", methods=["POST"])
def setup_done():
    if game is None:
        return jsonify({"error": "No game in progress"}), 400
    result = game.finish_setup()
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/moves", methods=["GET"])
def get_moves():
    if game is None:
        return jsonify({"error": "No game in progress"}), 400
    r = int(request.args["r"])
    c = int(request.args["c"])
    return jsonify({"moves": game.get_valid_moves(r, c)})


@app.route("/api/move", methods=["POST"])
def make_move():
    if game is None:
        return jsonify({"error": "No game in progress"}), 400
    data = request.get_json(force=True)
    result = game.make_move(data["fr"], data["fc"], data["tr"], data["tc"])
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    print("Starting RPS Stratego server on http://localhost:5000")
    app.run(debug=True, port=5000)

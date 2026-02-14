"""AI decision engine for computer opponent.

Priority-based heuristic with full board visibility:
1. Capture flag if adjacent
2. Win attacks — attack revealed enemy pieces we beat (RPS advantage)
3. Advance — move toward enemy territory
4. Neutral — attack unknown pieces or lateral moves
5. Avoid — don't attack revealed pieces that beat us (last resort)
"""

import random

from models import Game

WINS = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}


def pick_ai_move(game: Game, player: int = 2) -> dict | None:
    """Choose a move for the AI player.  Returns {fr, fc, tr, tc} or None."""
    buckets: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}

    for r in range(game.size):
        for c in range(game.size):
            piece = game.grid[r][c]
            if not piece or piece.owner != player or not piece.is_movable():
                continue

            for move in game.get_valid_moves(r, c):
                tr, tc = move["r"], move["c"]
                target = game.grid[tr][tc]
                entry = {"fr": r, "fc": c, "tr": tr, "tc": tc}

                if target is None:
                    # Empty cell — score by advancement toward enemy
                    if player == 2:
                        advance = tr - r  # positive = toward P1 (higher rows)
                    else:
                        advance = r - tr
                    if advance > 0:
                        buckets[3].append(entry)
                    else:
                        buckets[4].append(entry)
                elif target.type == "flag":
                    buckets[1].append(entry)
                elif target.revealed and target.type in ("rock", "paper", "scissors"):
                    if (piece.type, target.type) in WINS:
                        buckets[2].append(entry)
                    elif piece.type == target.type:
                        buckets[4].append(entry)
                    else:
                        buckets[5].append(entry)
                elif target.type == "bomb":
                    # Revealed bomb — avoid it
                    buckets[5].append(entry)
                else:
                    # Unknown enemy piece
                    buckets[4].append(entry)

    for priority in (1, 2, 3, 4, 5):
        if buckets[priority]:
            return random.choice(buckets[priority])
    return None

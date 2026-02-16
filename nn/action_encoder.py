import torch
from models import Game

# Directions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# For P2 flipping: up<->down swap
P2_DIR_MAP = {0: 1, 1: 0, 2: 2, 3: 3}


def move_to_action(fr: int, fc: int, tr: int, tc: int, player: int, board_size: int) -> int:
    """Convert a move (fr,fc)->(tr,tc) to a canonical action index.

    For player 2, coordinates are flipped vertically and up/down swapped
    so the action space is always from the perspective of player at bottom.
    """
    dr, dc = tr - fr, tc - fc
    direction = DIRECTIONS.index((dr, dc))

    if player == 2:
        fr = board_size - 1 - fr
        direction = P2_DIR_MAP[direction]

    return fr * (board_size * 4) + fc * 4 + direction


def action_to_move(action: int, player: int, board_size: int) -> tuple[int, int, int, int]:
    """Convert a canonical action index back to (fr, fc, tr, tc) in real coordinates."""
    direction = action % 4
    remaining = action // 4
    fc = remaining % board_size
    fr = remaining // board_size

    if player == 2:
        fr = board_size - 1 - fr
        direction = P2_DIR_MAP[direction]

    dr, dc = DIRECTIONS[direction]
    tr, tc = fr + dr, fc + dc
    return fr, fc, tr, tc


def get_valid_action_mask(game: Game, player: int) -> torch.Tensor:
    """Return a boolean tensor of shape (num_actions,) marking valid actions."""
    size = game.size
    num_actions = size * size * 4
    mask = torch.zeros(num_actions, dtype=torch.bool)

    saved_player = game.current_player
    game.current_player = player

    for r in range(size):
        for c in range(size):
            piece = game.grid[r][c]
            if piece is None or piece.owner != player or not piece.is_movable():
                continue
            for move in game.get_valid_moves(r, c):
                tr, tc = move["r"], move["c"]
                idx = move_to_action(r, c, tr, tc, player, size)
                mask[idx] = True

    game.current_player = saved_player
    return mask

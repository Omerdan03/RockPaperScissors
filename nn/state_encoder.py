import torch
from models import Game


PIECE_TYPES = ["rock", "paper", "scissors", "flag", "bomb"]
PIECE_INDEX = {t: i for i, t in enumerate(PIECE_TYPES)}


def encode_state(game: Game, player: int) -> torch.Tensor:
    """Encode the game state from `player`'s perspective as a (15, 6, 6) tensor.

    Board is canonicalized so the current player is always at the bottom.
    For player 2, the board is flipped vertically.

    Channels:
        0-4:  Own pieces (rock, paper, scissors, flag, bomb)
        5-9:  Enemy revealed pieces (rock, paper, scissors, flag, bomb)
        10:   Enemy unknown (unrevealed) pieces
        11:   Empty cells
        12:   Turn count / 200 (broadcast)
        13:   Own piece count / 12 (broadcast)
        14:   Enemy piece count / 12 (broadcast)
    """
    size = game.size
    state = torch.zeros(15, size, size)

    own_count = 0
    enemy_count = 0

    for r in range(size):
        for c in range(size):
            # Flip row for player 2 so own pieces are always at bottom
            cr = (size - 1 - r) if player == 2 else r

            piece = game.grid[r][c]
            if piece is None:
                state[11, cr, c] = 1.0
            elif piece.owner == player:
                idx = PIECE_INDEX[piece.type]
                state[idx, cr, c] = 1.0
                own_count += 1
            else:
                enemy_count += 1
                if piece.revealed:
                    idx = PIECE_INDEX[piece.type]
                    state[5 + idx, cr, c] = 1.0
                else:
                    state[10, cr, c] = 1.0

    state[12] = game.turn_count / 200.0
    state[13] = own_count / 12.0
    state[14] = enemy_count / 12.0

    return state

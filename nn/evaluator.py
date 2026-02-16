import torch
import torch.nn.functional as F

from models import Game
from ai import pick_ai_move
from nn.config import Config
from nn.state_encoder import encode_state
from nn.action_encoder import get_valid_action_mask, action_to_move
from nn.network import RPSNet


def evaluate_vs_heuristic(
    model: RPSNet,
    config: Config,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Evaluate the NN against the heuristic AI over multiple games.

    Plays config.eval_games total (half as P1, half as P2).
    Returns dict with win_rate, loss_rate, draw_rate, avg_game_length.
    """
    wins = 0
    losses = 0
    draws = 0
    total_length = 0
    half = config.eval_games // 2

    for game_idx in range(config.eval_games):
        nn_player = 1 if game_idx < half else 2
        heuristic_player = 3 - nn_player

        game = Game(config.board_size, "computer")
        game.auto_setup_player(1)
        game.auto_setup_player(2)
        game.phase = "play"
        game.current_player = 1

        for turn in range(config.max_turns):
            if game.phase == "over":
                break

            current = game.current_player

            if current == nn_player:
                move = _pick_nn_move(model, game, nn_player, config, device)
            else:
                move = pick_ai_move(game, heuristic_player)

            if move is None:
                break

            result = game.make_move(move["fr"], move["fc"], move["tr"], move["tc"])
            if "error" in result:
                break

        total_length += game.turn_count

        if game.winner == nn_player:
            wins += 1
        elif game.winner == heuristic_player:
            losses += 1
        else:
            draws += 1

    total = config.eval_games
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "avg_game_length": total_length / total,
    }


def _pick_nn_move(
    model: RPSNet,
    game: Game,
    player: int,
    config: Config,
    device: torch.device,
) -> dict | None:
    """Pick a move using the neural network with low temperature (near greedy)."""
    state = encode_state(game, player).to(device)
    mask = get_valid_action_mask(game, player).to(device)

    if not mask.any():
        return None

    with torch.no_grad():
        logits, _ = model(state.unsqueeze(0))
    logits = logits.squeeze(0)

    logits[~mask] = float("-inf")
    probs = F.softmax(logits / max(config.eval_temperature, 1e-8), dim=0)

    action = probs.argmax().item()
    fr, fc, tr, tc = action_to_move(action, player, config.board_size)
    return {"fr": fr, "fc": fc, "tr": tr, "tc": tc}

"""Thin wrapper to use the trained neural network as an AI player."""

import os

import torch
import torch.nn.functional as F

from models import Game
from nn.config import Config
from nn.state_encoder import encode_state
from nn.action_encoder import get_valid_action_mask, action_to_move
from nn.network import RPSNet

_model: RPSNet | None = None
_config: Config | None = None
_device: torch.device | None = None


def _load_model() -> tuple[RPSNet, Config, torch.device]:
    """Lazy-load the model singleton."""
    global _model, _config, _device
    if _model is not None:
        return _model, _config, _device

    _config = Config()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = RPSNet(_config).to(_device)

    model_path = os.path.join(_config.checkpoint_dir, "best_model.pt")
    if os.path.exists(model_path):
        _model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
    _model.eval()

    return _model, _config, _device


def pick_nn_move(game: Game, player: int = 2) -> dict | None:
    """Choose a move using the neural network. Same signature as pick_ai_move."""
    model, config, device = _load_model()

    state = encode_state(game, player).to(device)
    mask = get_valid_action_mask(game, player).to(device)

    if not mask.any():
        return None

    with torch.no_grad():
        logits, _ = model(state.unsqueeze(0))
    logits = logits.squeeze(0)

    logits[~mask] = float("-inf")
    action = logits.argmax().item()

    fr, fc, tr, tc = action_to_move(action, player, config.board_size)
    return {"fr": fr, "fc": fc, "tr": tr, "tc": tc}

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ai import pick_ai_move
from models import Game
from nn.config import Config
from nn.state_encoder import encode_state
from nn.action_encoder import get_valid_action_mask, action_to_move, move_to_action
from nn.network import RPSNet


@dataclass
class Transition:
    state: torch.Tensor       # (15, 6, 6)
    action_mask: torch.Tensor # (144,) bool
    action: int
    log_prob: float
    value: float
    reward: float
    player: int               # 1 or 2


def play_episode(
    model: RPSNet,
    config: Config,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> list[Transition]:
    """Run one self-play episode and return transitions with assigned rewards."""
    game = Game(config.board_size, "computer")
    game.auto_setup_player(1)
    game.auto_setup_player(2)
    game.phase = "play"
    game.current_player = 1

    transitions: list[Transition] = []

    for _ in range(config.max_turns):
        player = game.current_player
        state = encode_state(game, player).to(device)
        mask = get_valid_action_mask(game, player).to(device)

        if not mask.any():
            break

        with torch.no_grad():
            logits, value = model(state.unsqueeze(0))
        logits = logits.squeeze(0)
        value = value.item()

        # Mask invalid actions and apply temperature
        logits[~mask] = float("-inf")
        probs = F.softmax(logits / max(temperature, 1e-8), dim=0)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        action_idx = action.item()

        fr, fc, tr, tc = action_to_move(action_idx, player, config.board_size)
        result = game.make_move(fr, fc, tr, tc)

        if "error" in result:
            # Should not happen with proper masking, but be safe
            break

        transitions.append(Transition(
            state=state.cpu(),
            action_mask=mask.cpu(),
            action=action_idx,
            log_prob=log_prob,
            value=value,
            reward=0.0,  # assigned after episode ends
            player=player,
        ))

        if game.phase == "over":
            break

    # Assign rewards
    _assign_rewards(transitions, game, config.gamma)
    return transitions


def _assign_rewards(transitions: list[Transition], game: Game, gamma: float) -> None:
    """Assign discounted rewards from the game outcome."""
    if not transitions:
        return

    if game.winner is not None:
        winner = game.winner
        # Walk backward, applying discounted reward
        discount = 1.0
        for t in reversed(transitions):
            if t.player == winner:
                t.reward = discount
            else:
                t.reward = -discount
            discount *= gamma
    else:
        # Timeout: small penalty for both
        discount = 1.0
        for t in reversed(transitions):
            t.reward = -0.1 * discount
            discount *= gamma


def play_vs_heuristic(
    model: RPSNet,
    config: Config,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> list[Transition]:
    """Run one episode of NN vs heuristic AI. NN plays a random side."""
    game = Game(config.board_size, "computer")
    game.auto_setup_player(1)
    game.auto_setup_player(2)
    game.phase = "play"
    game.current_player = 1

    nn_player = random.choice([1, 2])
    transitions: list[Transition] = []

    for _ in range(config.max_turns):
        player = game.current_player

        if player != nn_player:
            # Heuristic move
            move = pick_ai_move(game, player)
            if move is None:
                break
            result = game.make_move(move["fr"], move["fc"], move["tr"], move["tc"])
            if "error" in result:
                break
            if game.phase == "over":
                break
            continue

        # NN move
        state = encode_state(game, player).to(device)
        mask = get_valid_action_mask(game, player).to(device)

        if not mask.any():
            break

        with torch.no_grad():
            logits, value = model(state.unsqueeze(0))
        logits = logits.squeeze(0)
        value = value.item()

        logits[~mask] = float("-inf")
        probs = F.softmax(logits / max(temperature, 1e-8), dim=0)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        action_idx = action.item()

        fr, fc, tr, tc = action_to_move(action_idx, player, config.board_size)
        result = game.make_move(fr, fc, tr, tc)

        if "error" in result:
            break

        transitions.append(Transition(
            state=state.cpu(),
            action_mask=mask.cpu(),
            action=action_idx,
            log_prob=log_prob,
            value=value,
            reward=0.0,
            player=player,
        ))

        if game.phase == "over":
            break

    # Assign rewards from NN player's perspective
    _assign_rewards(transitions, game, config.gamma)
    return transitions


def collect_episodes(
    model: RPSNet,
    config: Config,
    temperature: float,
    device: torch.device = torch.device("cpu"),
) -> list[Transition]:
    """Collect transitions from a mix of self-play and vs-heuristic episodes."""
    all_transitions: list[Transition] = []
    n = config.episodes_per_iter
    n_heuristic = int(n * config.heuristic_ratio)
    n_selfplay = n - n_heuristic

    for i in range(n):
        if i < n_heuristic:
            episode = play_vs_heuristic(model, config, temperature, device)
        else:
            episode = play_episode(model, config, temperature, device)
        all_transitions.extend(episode)
        if (i + 1) % max(n // 4, 1) == 0:
            print(f"  episodes {i + 1}/{n} done (h={min(i+1, n_heuristic)}/{n_heuristic})", flush=True)
    return all_transitions

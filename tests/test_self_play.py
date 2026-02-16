"""Tests for nn/self_play.py."""

import torch
from nn.config import Config
from nn.network import RPSNet
from nn.self_play import play_episode, Transition
from nn.action_encoder import action_to_move


class TestSelfPlay:
    def _make_config(self):
        config = Config()
        config.max_turns = 50  # Shorter for tests
        return config

    def test_episode_completes(self):
        config = self._make_config()
        model = RPSNet(config)
        model.eval()

        transitions = play_episode(model, config, temperature=1.0)

        assert len(transitions) > 0

    def test_transitions_have_correct_fields(self):
        config = self._make_config()
        model = RPSNet(config)
        model.eval()

        transitions = play_episode(model, config, temperature=1.0)

        for t in transitions:
            assert isinstance(t, Transition)
            assert t.state.shape == (15, 6, 6)
            assert t.action_mask.shape == (144,)
            assert 0 <= t.action < 144
            assert t.player in (1, 2)
            assert isinstance(t.log_prob, float)
            assert isinstance(t.value, float)
            assert isinstance(t.reward, float)

    def test_rewards_assigned(self):
        config = self._make_config()
        model = RPSNet(config)
        model.eval()

        transitions = play_episode(model, config, temperature=1.0)

        # At least some transitions should have non-zero rewards
        rewards = [t.reward for t in transitions]
        assert any(r != 0.0 for r in rewards)

    def test_all_actions_valid(self):
        """Every sampled action should be in the valid action mask."""
        config = self._make_config()
        model = RPSNet(config)
        model.eval()

        transitions = play_episode(model, config, temperature=1.0)

        for t in transitions:
            assert t.action_mask[t.action], f"Action {t.action} not in mask"

    def test_alternating_players(self):
        """Players should mostly alternate (some may be skipped if game ends)."""
        config = self._make_config()
        model = RPSNet(config)
        model.eval()

        transitions = play_episode(model, config, temperature=1.0)

        if len(transitions) >= 2:
            # First two transitions should be different players
            assert transitions[0].player != transitions[1].player

    def test_actions_decode_to_valid_board_positions(self):
        config = self._make_config()
        model = RPSNet(config)
        model.eval()

        transitions = play_episode(model, config, temperature=1.0)

        for t in transitions:
            fr, fc, tr, tc = action_to_move(t.action, t.player, config.board_size)
            assert 0 <= fr < 6 and 0 <= fc < 6
            assert 0 <= tr < 6 and 0 <= tc < 6

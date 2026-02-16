"""Tests for nn/trainer.py — verify PPO training runs without error."""

import math
import torch
from nn.config import Config
from nn.network import RPSNet
from nn.self_play import play_episode
from nn.trainer import PPOTrainer


class TestTraining:
    def _make_config(self):
        config = Config()
        config.max_turns = 30
        config.episodes_per_iter = 2
        config.ppo_epochs = 2
        return config

    def test_training_runs(self):
        """3 iterations of collect + train should run without error."""
        config = self._make_config()
        model = RPSNet(config)
        trainer = PPOTrainer(model, config)

        for _ in range(3):
            model.eval()
            transitions = []
            for _ in range(config.episodes_per_iter):
                transitions.extend(play_episode(model, config, temperature=1.0))

            model.train()
            metrics = trainer.train_on_transitions(transitions)

            assert math.isfinite(metrics["total_loss"])
            assert math.isfinite(metrics["policy_loss"])
            assert math.isfinite(metrics["value_loss"])
            assert math.isfinite(metrics["entropy"])

    def test_loss_is_finite(self):
        config = self._make_config()
        model = RPSNet(config)
        trainer = PPOTrainer(model, config)

        model.eval()
        transitions = play_episode(model, config, temperature=1.0)
        model.train()
        metrics = trainer.train_on_transitions(transitions)

        for key, val in metrics.items():
            assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_empty_transitions(self):
        config = self._make_config()
        model = RPSNet(config)
        trainer = PPOTrainer(model, config)

        metrics = trainer.train_on_transitions([])
        assert metrics["total_loss"] == 0.0

    def test_model_parameters_change(self):
        """After training, model parameters should have changed."""
        config = self._make_config()
        model = RPSNet(config)
        trainer = PPOTrainer(model, config)

        # Snapshot initial parameters
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        model.eval()
        transitions = []
        for _ in range(config.episodes_per_iter):
            transitions.extend(play_episode(model, config, temperature=1.0))

        model.train()
        trainer.train_on_transitions(transitions)

        # At least some parameters should have changed
        changed = False
        for n, p in model.named_parameters():
            if not torch.equal(p, initial_params[n]):
                changed = True
                break
        assert changed, "No parameters changed after training"

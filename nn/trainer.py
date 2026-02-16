import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from nn.config import Config
from nn.network import RPSNet
from nn.self_play import Transition


class PPOTrainer:
    def __init__(self, model: RPSNet, config: Config, device: torch.device = torch.device("cpu")):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.total_iterations)

    def train_on_transitions(self, transitions: list[Transition]) -> dict[str, float]:
        """Run PPO training on collected transitions. Returns loss metrics."""
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # Stack transition data into batches
        states = torch.stack([t.state for t in transitions]).to(self.device)
        masks = torch.stack([t.action_mask for t in transitions]).to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=self.device)
        values_old = torch.tensor([t.value for t in transitions], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)

        # Compute advantages (simple: return - value_estimate)
        advantages = rewards - values_old
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        for _ in range(self.config.ppo_epochs):
            logits, values = self.model(states)
            values = values.squeeze(-1)

            # Mask invalid actions
            logits[~masks] = float("-inf")
            log_probs_all = F.log_softmax(logits, dim=1)
            new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, rewards)

            # Entropy (over valid actions only)
            probs = F.softmax(logits, dim=1)
            # 0 * -inf = nan for masked entries, replace before summing
            log_probs_safe = torch.where(masks, log_probs_all, torch.zeros_like(log_probs_all))
            probs_safe = torch.where(masks, probs, torch.zeros_like(probs))
            entropy = -(probs_safe * log_probs_safe).sum(dim=1).mean()

            loss = policy_loss + self.config.value_coeff * value_loss - self.config.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_metrics["policy_loss"] += policy_loss.item()
            total_metrics["value_loss"] += value_loss.item()
            total_metrics["entropy"] += entropy.item()
            total_metrics["total_loss"] += loss.item()

        # Average over epochs
        for k in total_metrics:
            total_metrics[k] /= self.config.ppo_epochs

        self.scheduler.step()
        return total_metrics

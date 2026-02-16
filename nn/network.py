import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.config import Config


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class RPSNet(nn.Module):
    def __init__(self, config: Config | None = None):
        super().__init__()
        cfg = config or Config()

        # Input projection
        self.input_conv = nn.Conv2d(cfg.input_channels, cfg.hidden_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(cfg.hidden_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(cfg.hidden_channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * cfg.board_size * cfg.board_size, cfg.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(cfg.hidden_channels, 16, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * cfg.board_size * cfg.board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 15, 6, 6) state tensor

        Returns:
            policy_logits: (batch, 144) raw logits
            value: (batch, 1) in [-1, 1]
        """
        # Shared trunk
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

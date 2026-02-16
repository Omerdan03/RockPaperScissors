"""Entry point for training the RPS Stratego neural network via self-play RL."""

import os
import torch

from nn.config import Config
from nn.network import RPSNet
from nn.self_play import collect_episodes
from nn.trainer import PPOTrainer
from nn.evaluator import evaluate_vs_heuristic


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RPSNet(config).to(device)
    trainer = PPOTrainer(model, config, device)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_win_rate = 0.0

    for iteration in range(1, config.total_iterations + 1):
        # Anneal temperature
        progress = min(iteration / config.temp_anneal_iters, 1.0)
        temperature = config.temp_start + (config.temp_end - config.temp_start) * progress

        # Self-play
        model.eval()
        transitions = collect_episodes(model, config, temperature, device)

        # Train
        model.train()
        metrics = trainer.train_on_transitions(transitions)

        avg_reward = sum(t.reward for t in transitions) / max(len(transitions), 1)

        print(
            f"Iter {iteration:4d} | "
            f"loss={metrics['total_loss']:.4f} | "
            f"pi={metrics['policy_loss']:.4f} | "
            f"v={metrics['value_loss']:.4f} | "
            f"ent={metrics['entropy']:.4f} | "
            f"avg_ret={avg_reward:.4f} | "
            f"temp={temperature:.2f} | "
            f"transitions={len(transitions)}"
        )

        # Evaluate
        if iteration % config.eval_every == 0:
            model.eval()
            eval_result = evaluate_vs_heuristic(model, config, device)
            win_rate = eval_result["win_rate"]
            print(
                f"  EVAL | win={win_rate:.1%} | "
                f"loss={eval_result['loss_rate']:.1%} | "
                f"draw={eval_result['draw_rate']:.1%} | "
                f"avg_len={eval_result['avg_game_length']:.1f}"
            )

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                path = os.path.join(config.checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), path)
                print(f"  New best model saved ({win_rate:.1%})")

        # Checkpoint
        if iteration % config.checkpoint_every == 0:
            path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}.pt")
            torch.save(model.state_dict(), path)

    print("Training complete.")


if __name__ == "__main__":
    main()

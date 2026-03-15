from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import torch

from slay_rl.config import Config, get_default_config
from slay_rl.sts_env import STSEnv
from slay_rl.models.combat_model import CombatModel
from slay_rl.agents.combat_agent import CombatAgent, RuleBasedCombatAgent

EPISODE_DIR = Path("data/episodes")
EPISODE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Episode result.txt
# =========================================================

@dataclass
class EpisodeStep:
    step_idx: int
    action_index: int
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    won: bool
    lost: bool
    final_player_hp: float
    history: List[EpisodeStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "won": self.won,
            "lost": self.lost,
            "final_player_hp": self.final_player_hp,
            "history": [
                {
                    "step_idx": x.step_idx,
                    "action_index": x.action_index,
                    "reward": x.reward,
                    "done": x.done,
                    "info": x.info,
                }
                for x in self.history
            ],
        }


# =========================================================
# Controller
# =========================================================

class RunController:
    def __init__(
        self,
        cfg: Optional[Config] = None,
        mode: str = "mock",
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.cfg = cfg or get_default_config()
        self.mode = mode
        self.seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        if device is None:
            if torch.cuda.is_available() and self.cfg.train.device == "cuda":
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        self.env = STSEnv(cfg=self.cfg, mode=mode, seed=seed)
        self.model = CombatModel(cfg=self.cfg).to(self.device)
        self.agent = CombatAgent(self.model, cfg=self.cfg, device=self.device)
        self.rule_agent = RuleBasedCombatAgent(cfg=self.cfg)

    # -----------------------------------------------------
    # Core loops
    # -----------------------------------------------------

    def run_episode_with_model(
        self,
        deterministic: bool = False,
        max_steps: int = 500,
        render: bool = True,
    ) -> EpisodeResult:
        state = self.env.reset()
        history: List[EpisodeStep] = []
        total_reward = 0.0

        if render:
            print("\n=== MODEL EPISODE START ===")
            self.env.render_text()

        for step_idx in range(max_steps):
            decision = self.agent.choose_action(state, deterministic=deterministic)
            action_index = int(decision["action_index"])

            next_state, reward, done, info = self.env.step(action_index)

            total_reward += reward
            history.append(
                EpisodeStep(
                    step_idx=step_idx,
                    action_index=action_index,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )

            if render:
                print(f"\n[MODEL] step={step_idx} action={action_index} reward={reward:.3f} done={done}")
                print("command =", info.get("command"))
                print("illegal =", info.get("illegal_action"))
                self.env.render_text()

            state = next_state
            if done:
                break

        return self._build_episode_result(total_reward, history, state)

    def run_episode_with_rule_agent(
        self,
        max_steps: int = 500,
        render: bool = True,
    ) -> EpisodeResult:
        state = self.env.reset()
        history: List[EpisodeStep] = []
        total_reward = 0.0

        if render:
            print("\n=== RULE EPISODE START ===")
            self.env.render_text()

        for step_idx in range(max_steps):
            action_index = int(self.rule_agent.choose_action_index(state))
            next_state, reward, done, info = self.env.step(action_index)

            total_reward += reward
            history.append(
                EpisodeStep(
                    step_idx=step_idx,
                    action_index=action_index,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )

            if render:
                print(f"\n[RULE] step={step_idx} action={action_index} reward={reward:.3f} done={done}")
                print("command =", info.get("command"))
                print("illegal =", info.get("illegal_action"))
                self.env.render_text()

            state = next_state
            if done:
                break

        return self._build_episode_result(total_reward, history, state)

    def run_episode_random(
        self,
        max_steps: int = 500,
        render: bool = True,
    ) -> EpisodeResult:
        state = self.env.reset()
        history: List[EpisodeStep] = []
        total_reward = 0.0

        if render:
            print("\n=== RANDOM EPISODE START ===")
            self.env.render_text()

        for step_idx in range(max_steps):
            encoded = self.agent.encoder.encode(state, device="cpu")
            valid_mask = encoded["valid_action_mask"].tolist()

            valid_actions = [i for i, x in enumerate(valid_mask) if x > 0.0]
            if len(valid_actions) == 0:
                end_turn_idx = (
                        self.cfg.combat_obs.max_hand_cards
                        + self.cfg.combat_obs.max_hand_cards * self.cfg.combat_obs.max_enemies
                )
                action_index = end_turn_idx
            else:
                action_index = random.choice(valid_actions)

            next_state, reward, done, info = self.env.step(action_index)

            total_reward += reward
            history.append(
                EpisodeStep(
                    step_idx=step_idx,
                    action_index=action_index,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )

            if render:
                print(f"\n[RANDOM] step={step_idx} action={action_index} reward={reward:.3f} done={done}")
                print("command =", info.get("command"))
                print("illegal =", info.get("illegal_action"))
                self.env.render_text()

            state = next_state
            if done:
                break

        return self._build_episode_result(total_reward, history, state)

    # -----------------------------------------------------
    # Utility methods
    # -----------------------------------------------------

    def save_combat_model(self, path: str) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config_build_target": self.cfg.game.build_target,
            "device": self.device,
        }
        torch.save(payload, path)
        print(f"Saved model to: {path}")

    def load_combat_model(self, path: str, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=strict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model from: {path}")

    def set_eval_mode(self) -> None:
        self.model.eval()

    def set_train_mode(self) -> None:
        self.model.train()

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    def _build_episode_result(
        self,
        total_reward: float,
        history: List[EpisodeStep],
        final_state: Dict[str, Any],
    ) -> EpisodeResult:
        player = final_state.get("player", {})
        final_hp = float(player.get("current_hp", player.get("currentHealth", player.get("hp", 0.0))))

        won = False
        lost = False
        if len(history) > 0:
            last_info = history[-1].info
            won = bool(last_info.get("combat_won", False))
            lost = bool(last_info.get("combat_lost", False))

        return EpisodeResult(
            total_reward=total_reward,
            steps=len(history),
            won=won,
            lost=lost,
            final_player_hp=final_hp,
            history=history,
        )

# =========================================================
# Convenience functions
# =========================================================

def save_episode(result, episode_idx: int, mode: str):
    path = EPISODE_DIR / f"{mode}_episode_{episode_idx:06d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[episode saved] {path}")

def run_game(
    mode: str = "mock",
    agent_type: str = "rule",
    deterministic: bool = False,
    max_steps: int = 500,
    render: bool = True,
    seed: int = 42,
) -> EpisodeResult:
    cfg = get_default_config()
    controller = RunController(cfg=cfg, mode=mode, seed=seed)

    if agent_type == "model":
        return controller.run_episode_with_model(
            deterministic=deterministic,
            max_steps=max_steps,
            render=render,
        )

    if agent_type == "rule":
        return controller.run_episode_with_rule_agent(
            max_steps=max_steps,
            render=render,
        )

    if agent_type == "random":
        return controller.run_episode_random(
            max_steps=max_steps,
            render=render,
        )

    raise ValueError(f"Unknown agent_type: {agent_type}")


def quick_test() -> None:
    print("\n######## QUICK TEST: RULE AGENT ########")
    result = run_game(
        mode="mock",
        agent_type="rule",
        deterministic=True,
        max_steps=100,
        render=True,
        seed=42,
    )
    print("\nFinal result.txt:")
    print(result.to_dict())

# =========================================================
# Debug main
# =========================================================

if __name__ == "__main__":
    quick_test()
from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam

from slay_rl.config import Config, get_default_config, CHECKPOINT_DIR, LOG_DIR
from slay_rl.sts_env import STSEnv
from slay_rl.models.combat_model import CombatModel, compute_ppo_loss, stack_encoded_obs
from slay_rl.agents.combat_agent import CombatAgent

# =========================================================
# Rollout storage
# =========================================================

@dataclass
class Transition:
    encoded_obs: Dict[str, torch.Tensor]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    terminated: bool


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        rewards = [t.reward for t in self.transitions]
        values = [t.value for t in self.transitions]
        terminateds = [t.terminated for t in self.transitions]

        advantages = [0.0 for _ in self.transitions]
        returns = [0.0 for _ in self.transitions]

        gae = 0.0
        next_value = last_value

        for t in reversed(range(len(self.transitions))):
            mask = 0.0 if terminateds[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        batch_obs = stack_encoded_obs([t.encoded_obs for t in self.transitions])

        batch_actions = torch.tensor(
            [t.action for t in self.transitions],
            dtype=torch.long,
        )
        batch_old_log_probs = torch.tensor(
            [t.log_prob for t in self.transitions],
            dtype=torch.float32,
        )
        batch_returns = torch.tensor(
            returns,
            dtype=torch.float32,
        )
        batch_advantages = torch.tensor(
            advantages,
            dtype=torch.float32,
        )

        return {
            "obs": batch_obs,
            "actions": batch_actions,
            "old_log_probs": batch_old_log_probs,
            "returns": batch_returns,
            "advantages": batch_advantages,
        }


# =========================================================
# Train stats
# =========================================================

@dataclass
class TrainStats:
    update_idx: int
    avg_episode_reward: float
    avg_episode_len: float
    win_rate: float
    episodes_finished: int
    truncations: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_idx": self.update_idx,
            "avg_episode_reward": self.avg_episode_reward,
            "avg_episode_len": self.avg_episode_len,
            "win_rate": self.win_rate,
            "episodes_finished": self.episodes_finished,
            "truncations": self.truncations,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
        }


# =========================================================
# Dummy vectorized env
# =========================================================

class DummyVecCombatEnv:
    def __init__(
        self,
        num_envs: int,
        cfg: Optional[Config] = None,
        seed: int = 42,
    ):
        self.cfg = cfg or get_default_config()
        self.num_envs = int(num_envs)
        self.envs = [
            STSEnv(cfg=self.cfg, seed=seed + i)
            for i in range(self.num_envs)
        ]
        self.episode_steps = [0 for _ in range(self.num_envs)]
        self.max_episode_steps = int(self.cfg.train.max_episode_steps)

    def set_training_progress(self, update_idx: int) -> None:
        for env in self.envs:
            if hasattr(env, "set_training_progress"):
                env.set_training_progress(update_idx)

    def reset(self) -> List[Dict[str, Any]]:
        self.episode_steps = [0 for _ in range(self.num_envs)]
        return [env.reset() for env in self.envs]

    def step(
            self,
            actions: List[int],
    ) -> tuple[List[Dict[str, Any]], List[float], List[bool], List[Dict[str, Any]]]:
        next_states = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(action)

            self.episode_steps[i] += 1

            info = dict(info)
            terminated = bool(info.get("combat_won", False) or info.get("combat_lost", False))
            truncated = False

            if not done and self.episode_steps[i] >= self.max_episode_steps:
                done = True
                truncated = True

            info["time_limit_truncated"] = truncated
            info["terminated"] = bool(terminated or truncated)

            if done:
                info["terminal_state"] = next_state
                next_state = env.reset()
                self.episode_steps[i] = 0

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return next_states, rewards, dones, infos

    def step_commands(self, commands, action_indices=None):
        next_states = []
        rewards = []
        dones = []
        infos = []

        for i, (env, command) in enumerate(zip(self.envs, commands)):
            forced_idx = None
            if action_indices is not None:
                forced_idx = int(action_indices[i])

            next_state, reward, done, info = env.step_command(
                command,
                forced_action_index=forced_idx,
            )

            self.episode_steps[i] += 1

            info = dict(info)
            terminated = bool(info.get("combat_won", False) or info.get("combat_lost", False))
            truncated = False

            if not done and self.episode_steps[i] >= self.max_episode_steps:
                done = True
                truncated = True

            info["time_limit_truncated"] = truncated
            info["terminated"] = terminated

            if done:
                info["terminal_state"] = next_state
                next_state = env.reset()
                self.episode_steps[i] = 0

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return next_states, rewards, dones, infos


# =========================================================
# Trainer
# =========================================================

class CombatTrainer:
    def __init__(
        self,
        cfg: Optional[Config] = None,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.cfg = cfg or get_default_config()
        self.seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        if device is None:
            if torch.cuda.is_available() and self.cfg.train.device == "cuda":
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        self.num_envs = int(self.cfg.train.num_envs)

        if self.cfg.ppo.rollout_steps % self.num_envs != 0:
            raise ValueError(
                f"rollout_steps ({self.cfg.ppo.rollout_steps}) must be divisible by "
                f"num_envs ({self.num_envs})."
            )

        self.vec_env = DummyVecCombatEnv(
            num_envs=self.num_envs,
            cfg=self.cfg,
            seed=seed,
        )

        self.current_states = self.vec_env.reset()

        self.model = CombatModel(cfg=self.cfg).to(self.device)
        self.agent = CombatAgent(self.model, cfg=self.cfg, device=self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.ppo.lr)

        self.checkpoint_dir = Path(CHECKPOINT_DIR) / self.cfg.train.run_name
        self.log_dir = Path(LOG_DIR) / self.cfg.train.run_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Run name: {self.cfg.train.run_name}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Log dir: {self.log_dir}")

        self.best_eval_win_rate = -1.0
        self.best_eval_reward = float("-inf")
        self.best_robust_eval_win_rate = -1.0
        self.best_robust_eval_reward = float("-inf")
        self.metrics_csv_path = self.log_dir / "train_metrics.csv"

    # -----------------------------------------------------
    # Collection
    # -----------------------------------------------------

    def _batched_rollout_step(
            self,
            states: List[Dict[str, Any]],
            deterministic: bool = False,
    ) -> Dict[str, Any]:
        encoded_list = [
            self.agent.encoder.encode(state, device=self.device)
            for state in states
        ]

        batch_obs = stack_encoded_obs(encoded_list)
        batch_obs = {k: v.to(self.device) for k, v in batch_obs.items()}

        with torch.no_grad():
            act_out = self.model.act(batch_obs, deterministic=deterministic)

        actions = act_out["action"].detach().cpu().tolist()
        log_probs = act_out["log_prob"].detach().cpu().tolist()
        values = act_out["value"].detach().cpu().tolist()

        return {
            "encoded_list": encoded_list,
            "actions": [int(a) for a in actions],
            "log_probs": [float(lp) for lp in log_probs],
            "values": [float(v) for v in values],
        }

    def _concat_rollout_batches(
            self,
            env_batches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if len(env_batches) == 0:
            raise RuntimeError("No env batches to concatenate.")

        obs_keys = env_batches[0]["obs"].keys()

        batch_obs = {
            k: torch.cat([b["obs"][k] for b in env_batches], dim=0)
            for k in obs_keys
        }

        return {
            "obs": batch_obs,
            "actions": torch.cat([b["actions"] for b in env_batches], dim=0),
            "old_log_probs": torch.cat([b["old_log_probs"] for b in env_batches], dim=0),
            "returns": torch.cat([b["returns"] for b in env_batches], dim=0),
            "advantages": torch.cat([b["advantages"] for b in env_batches], dim=0),
        }

    def collect_rollout(
            self,
            rollout_steps: Optional[int] = None,
            update_idx: int = 0,
    ) -> Dict[str, Any]:
        rollout_steps = rollout_steps or self.cfg.ppo.rollout_steps

        buffers = [RolloutBuffer() for _ in range(self.num_envs)]
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        episode_wins: List[int] = []
        truncations = 0

        self.vec_env.set_training_progress(update_idx)
        states = self.current_states

        current_ep_rewards = [0.0 for _ in range(self.num_envs)]
        current_ep_lengths = [0 for _ in range(self.num_envs)]

        self.model.eval()
        potion_actions = 0

        steps_per_env = max(1, rollout_steps // self.num_envs)

        for _ in range(steps_per_env):
            rollout_out = self._batched_rollout_step(states, deterministic=False)

            actions = rollout_out["actions"]
            log_probs = rollout_out["log_probs"]
            values = rollout_out["values"]
            encoded_list = rollout_out["encoded_list"]

            for action in actions:
                if self._is_potion_action(action):
                    potion_actions += 1

            commands = [
                self.agent.decode_action_index(action, state)
                for action, state in zip(actions, states)
            ]

            next_states, rewards, dones, infos = self.vec_env.step_commands(
                commands,
                actions,
            )

            for env_idx in range(self.num_envs):
                buffers[env_idx].add(
                    Transition(
                        encoded_obs=encoded_list[env_idx],
                        action=actions[env_idx],
                        log_prob=log_probs[env_idx],
                        value=values[env_idx],
                        reward=rewards[env_idx],
                        done=dones[env_idx],
                        terminated=bool(infos[env_idx].get("terminated", False)),
                    )
                )

                current_ep_rewards[env_idx] += rewards[env_idx]
                current_ep_lengths[env_idx] += 1

                if dones[env_idx]:
                    episode_rewards.append(current_ep_rewards[env_idx])
                    episode_lengths.append(current_ep_lengths[env_idx])
                    episode_wins.append(1 if infos[env_idx].get("combat_won", False) else 0)

                    if infos[env_idx].get("time_limit_truncated", False):
                        truncations += 1

                    current_ep_rewards[env_idx] = 0.0
                    current_ep_lengths[env_idx] = 0

            states = next_states

        # bootstrap value par environnement
        with torch.no_grad():
            encoded_list = [
                self.agent.encoder.encode(state, device=self.device)
                for state in states
            ]
            batch_obs = stack_encoded_obs(encoded_list)
            batch_obs = {k: v.to(self.device) for k, v in batch_obs.items()}

            out = self.model.forward(batch_obs)
            last_values = [float(v) for v in out["value"].detach().cpu().tolist()]

        env_batches: List[Dict[str, Any]] = []

        for env_idx in range(self.num_envs):
            env_batch = buffers[env_idx].compute_returns_and_advantages(
                gamma=self.cfg.ppo.gamma,
                gae_lambda=self.cfg.ppo.gae_lambda,
                last_value=last_values[env_idx],
            )
            env_batches.append(env_batch)

        batch = self._concat_rollout_batches(env_batches)

        batch["obs"] = {k: v.to(self.device) for k, v in batch["obs"].items()}
        batch["actions"] = batch["actions"].to(self.device)
        batch["old_log_probs"] = batch["old_log_probs"].to(self.device)
        batch["returns"] = batch["returns"].to(self.device)
        batch["advantages"] = batch["advantages"].to(self.device)

        # normalize advantages globalement sur tout le batch concaténé
        adv_mean = batch["advantages"].mean()
        adv_std = batch["advantages"].std(unbiased=False)
        batch["advantages"] = (batch["advantages"] - adv_mean) / (adv_std + 1e-8)

        self.current_states = states

        return {
            "batch": batch,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_wins": episode_wins,
            "episodes_finished": len(episode_rewards),
            "truncations": truncations,
            "num_transitions": sum(len(buf) for buf in buffers),
            "potion_actions": potion_actions,
        }

    # -----------------------------------------------------
    # PPO update
    # -----------------------------------------------------

    def ppo_update(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        self.model.train()

        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        n = actions.shape[0]
        batch_size = min(self.cfg.ppo.batch_size, n)
        indices = torch.arange(n, device=self.device)

        last_metrics: Optional[Dict[str, torch.Tensor]] = None

        for _ in range(self.cfg.ppo.epochs):
            perm = indices[torch.randperm(n, device=self.device)]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                mb_idx = perm[start:end]

                mini_obs = {k: v[mb_idx] for k, v in obs.items()}
                mini_actions = actions[mb_idx]
                mini_old_log_probs = old_log_probs[mb_idx]
                mini_returns = returns[mb_idx]
                mini_advantages = advantages[mb_idx]

                metrics = compute_ppo_loss(
                    model=self.model,
                    batch_obs=mini_obs,
                    batch_actions=mini_actions,
                    batch_old_log_probs=mini_old_log_probs,
                    batch_returns=mini_returns,
                    batch_advantages=mini_advantages,
                    clip_eps=self.cfg.ppo.clip_eps,
                    value_coef=self.cfg.ppo.value_coef,
                    entropy_coef=self.cfg.ppo.entropy_coef,
                )

                loss = metrics["loss"]

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.ppo.max_grad_norm)
                self.optimizer.step()

                last_metrics = metrics

        if last_metrics is None:
            raise RuntimeError("PPO update produced no metrics.")

        assert last_metrics is not None

        return {
            "policy_loss": float(last_metrics["policy_loss"].item()),
            "value_loss": float(last_metrics["value_loss"].item()),
            "entropy": float(last_metrics["entropy"].item()),
            "approx_kl": float(last_metrics["approx_kl"].item()),
            "clip_fraction": float(last_metrics["clip_fraction"].item()),
        }

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    @torch.no_grad()
    def evaluate(
            self,
            num_episodes: Optional[int] = None,
            deterministic: bool = True,
            seed_start: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.eval()

        num_episodes = num_episodes or self.cfg.train.eval_num_episodes
        seed_start = seed_start if seed_start is not None else self.cfg.train.eval_seed_start

        rewards = []
        lengths = []
        wins = []
        win_lengths = []
        final_hp_ratios = []
        episode_damage_taken_values = []
        potion_actions = 0
        truncations = 0

        eval_seeds = [seed_start + i for i in range(num_episodes)]

        for episode_seed in eval_seeds:
            eval_env = STSEnv(cfg=self.cfg, seed=episode_seed)

            state = eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_damage_taken = 0.0
            last_info = {}

            while not done and ep_len < 500:
                encoded = self.agent.encoder.encode(state, device=self.device)
                act_out = self.model.act(encoded, deterministic=deterministic)
                action_index = int(act_out["action"].item())

                if self._is_potion_action(action_index):
                    potion_actions += 1

                command = self.agent.decode_action_index(action_index, state)

                next_state, reward, done, info = eval_env.step_command(
                    command,
                    forced_action_index=action_index,
                )

                reward_breakdown = info.get("reward_breakdown", {})
                player_hp_before = float(reward_breakdown.get("player_hp_before", 0.0))
                player_hp_after = float(reward_breakdown.get("player_hp_after", player_hp_before))
                step_damage_taken = max(0.0, player_hp_before - player_hp_after)
                ep_damage_taken += step_damage_taken

                ep_reward += reward
                ep_len += 1
                state = next_state
                last_info = info

            if ep_len >= 500 and not done:
                truncations += 1

            rewards.append(ep_reward)
            lengths.append(ep_len)
            episode_damage_taken_values.append(ep_damage_taken)

            combat_won = bool(last_info.get("combat_won", False))
            wins.append(1 if combat_won else 0)

            player = state.get("player", {})
            final_hp = float(player.get("current_hp", player.get("currentHealth", player.get("hp", 0.0))))
            max_hp = float(player.get("max_hp", player.get("maxHealth", 1.0)))
            final_hp_ratios.append(final_hp / max(max_hp, 1.0))

            if combat_won:
                win_lengths.append(ep_len)

        avg_reward = sum(rewards) / max(len(rewards), 1)
        avg_len = sum(lengths) / max(len(lengths), 1)
        win_rate = sum(wins) / max(len(wins), 1)

        avg_final_hp_ratio = sum(final_hp_ratios) / max(len(final_hp_ratios), 1)
        avg_damage_taken = sum(episode_damage_taken_values) / max(len(episode_damage_taken_values), 1)
        avg_win_len = sum(win_lengths) / max(len(win_lengths), 1)

        return {
            "avg_reward": avg_reward,
            "avg_len": avg_len,
            "win_rate": win_rate,
            "avg_final_hp_ratio": avg_final_hp_ratio,
            "avg_damage_taken": avg_damage_taken,
            "avg_win_len": avg_win_len,
            "potion_actions_total": float(potion_actions),
            "potion_actions_per_episode": float(potion_actions) / max(len(rewards), 1),
            "truncations": truncations,
        }

    # -----------------------------------------------------
    # Train loop
    # -----------------------------------------------------

    @staticmethod
    def _fmt_metric(x: float, pct: bool = False, digits: int = 3) -> str:
        if isinstance(x, float) and x != x:
            return "n/a"
        if pct:
            return f"{100.0 * x:.2f}%"
        return f"{x:.{digits}f}"

    def train(self) -> List[TrainStats]:
        stats_history: List[TrainStats] = []

        total_updates = self.cfg.train.total_updates
        train_start_time = time.time()

        pbar = tqdm(
            range(1, total_updates + 1),
            desc="Training",
            unit="update",
            dynamic_ncols=True,
        )

        for update_idx in pbar:
            update_start_time = time.time()

            rollout_out = self.collect_rollout(
                rollout_steps=self.cfg.ppo.rollout_steps,
                update_idx=update_idx,
            )
            batch = rollout_out["batch"]

            metrics = self.ppo_update(batch)

            ep_rewards = rollout_out["episode_rewards"]
            ep_lengths = rollout_out["episode_lengths"]
            ep_wins = rollout_out["episode_wins"]
            episodes_finished = int(rollout_out["episodes_finished"])
            truncations = int(rollout_out["truncations"])

            avg_episode_reward = float(sum(ep_rewards) / len(ep_rewards)) if len(ep_rewards) > 0 else float("nan")
            avg_episode_len = float(sum(ep_lengths) / len(ep_lengths)) if len(ep_lengths) > 0 else float("nan")
            win_rate = float(sum(ep_wins) / len(ep_wins)) if len(ep_wins) > 0 else float("nan")

            stat = TrainStats(
                update_idx=update_idx,
                avg_episode_reward=avg_episode_reward,
                avg_episode_len=avg_episode_len,
                win_rate=win_rate,
                episodes_finished=episodes_finished,
                truncations=truncations,
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                entropy=metrics["entropy"],
                approx_kl=metrics["approx_kl"],
                clip_fraction=metrics["clip_fraction"],
            )

            stats_history.append(stat)

            eval_metrics = {
                "avg_reward": 0.0,
                "avg_len": 0.0,
                "win_rate": 0.0,
                "avg_final_hp_ratio": 0.0,
                "avg_damage_taken": 0.0,
                "avg_win_len": 0.0,
                "potion_actions_total": 0.0,
                "potion_actions_per_episode": 0.0,
            }

            robust_eval_metrics = {
                "avg_reward": 0.0,
                "avg_len": 0.0,
                "win_rate": 0.0,
                "avg_final_hp_ratio": 0.0,
                "avg_damage_taken": 0.0,
                "avg_win_len": 0.0,
                "potion_actions_total": 0.0,
                "potion_actions_per_episode": 0.0,
            }

            if update_idx % self.cfg.train.eval_every == 0 or update_idx == 1:
                eval_metrics = self.evaluate(
                    num_episodes=self.cfg.train.eval_num_episodes,
                    deterministic=True,
                    seed_start=self.cfg.train.eval_seed_start,
                )

                if (
                        eval_metrics["win_rate"] > self.best_eval_win_rate
                        or (
                        eval_metrics["win_rate"] == self.best_eval_win_rate
                        and eval_metrics["avg_reward"] > self.best_eval_reward
                )
                ):
                    self.best_eval_win_rate = eval_metrics["win_rate"]
                    self.best_eval_reward = eval_metrics["avg_reward"]
                    self.save_best_checkpoint(
                        "best",
                        win_rate=self.best_eval_win_rate,
                        reward=self.best_eval_reward,
                    )

            if update_idx % self.cfg.train.robust_eval_every == 0:
                robust_eval_metrics = self.evaluate(
                    num_episodes=self.cfg.train.robust_eval_num_episodes,
                    deterministic=True,
                    seed_start=self.cfg.train.robust_eval_seed_start,
                )

                if (
                        robust_eval_metrics["win_rate"] > self.best_robust_eval_win_rate
                        or (
                        robust_eval_metrics["win_rate"] == self.best_robust_eval_win_rate
                        and robust_eval_metrics["avg_reward"] > self.best_robust_eval_reward
                )
                ):
                    self.best_robust_eval_win_rate = robust_eval_metrics["win_rate"]
                    self.best_robust_eval_reward = robust_eval_metrics["avg_reward"]
                    self.save_best_checkpoint(
                        "best_robust",
                        win_rate=self.best_robust_eval_win_rate,
                        reward=self.best_robust_eval_reward,
                    )

            if update_idx % self.cfg.train.robust_eval_every == 0:
                tqdm.write(
                    f"[robust_eval {update_idx}] "
                    f"avg_reward={robust_eval_metrics['avg_reward']:.3f} | "
                    f"avg_len={robust_eval_metrics['avg_len']:.2f} | "
                    f"win_rate={robust_eval_metrics['win_rate']:.2%} | "
                    f"avg_final_hp_ratio={robust_eval_metrics['avg_final_hp_ratio']:.3f} | "
                    f"avg_damage_taken={robust_eval_metrics['avg_damage_taken']:.2f} | "
                    f"avg_win_len={robust_eval_metrics['avg_win_len']:.2f} | "
                    f"potion_actions_total={robust_eval_metrics['potion_actions_total']} | "
                    f"potion_actions_per_episode={robust_eval_metrics['potion_actions_per_episode']:.2f}"
                )

            train_num_episodes = len(ep_rewards)
            train_potion_actions_total = float(rollout_out["potion_actions"])
            train_potion_actions_per_episode = (
                train_potion_actions_total / train_num_episodes
                if train_num_episodes > 0
                else float("nan")
            )

            self._append_metrics_csv(
                stat,
                train_potion_actions_total=train_potion_actions_total,
                train_potion_actions_per_episode=train_potion_actions_per_episode,
                eval_metrics=eval_metrics,
                robust_eval_metrics=robust_eval_metrics,
            )

            if update_idx % self.cfg.train.save_every == 0 or update_idx == total_updates:
                self.save_checkpoint(update_idx)

            update_time = time.time() - update_start_time
            elapsed = time.time() - train_start_time
            avg_time_per_update = elapsed / update_idx
            remaining_updates = total_updates - update_idx
            eta_seconds = remaining_updates * avg_time_per_update

            pbar.set_postfix({
                "ep_fin": episodes_finished,
                "trunc": truncations,
                "reward": self._fmt_metric(avg_episode_reward, digits=2),
                "win": self._fmt_metric(win_rate, pct=True),
                "upd_s": f"{update_time:.1f}",
                "eta_m": f"{eta_seconds / 60:.1f}",
            })

            if update_idx % self.cfg.train.log_every == 0 or update_idx == 1:
                tqdm.write(
                    f"[update {update_idx}] "
                    f"episodes_finished={episodes_finished} | "
                    f"truncations={truncations} | "
                    f"avg_reward={self._fmt_metric(stat.avg_episode_reward)} | "
                    f"avg_len={self._fmt_metric(stat.avg_episode_len, digits=2)} | "
                    f"win_rate={self._fmt_metric(stat.win_rate, pct=True)} | "
                    f"policy_loss={stat.policy_loss:.4f} | "
                    f"value_loss={stat.value_loss:.4f} | "
                    f"entropy={stat.entropy:.4f} | "
                    f"kl={stat.approx_kl:.6f} | "
                    f"clip_frac={stat.clip_fraction:.4f} | "
                    f"update_time={update_time:.2f}s"
                )

            if update_idx % self.cfg.train.eval_every == 0 or update_idx == 1:
                tqdm.write(
                    f"[eval {update_idx}] "
                    f"avg_reward={eval_metrics['avg_reward']:.3f} | "
                    f"avg_len={eval_metrics['avg_len']:.2f} | "
                    f"win_rate={eval_metrics['win_rate']:.2%} | "
                    f"avg_final_hp_ratio={eval_metrics['avg_final_hp_ratio']:.3f} | "
                    f"avg_damage_taken={eval_metrics['avg_damage_taken']:.2f} | "
                    f"avg_win_len={eval_metrics['avg_win_len']:.2f}"
                )

        pbar.close()
        return stats_history

    # -----------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------

    def save_checkpoint(self, update_idx: int) -> Path:
        path = self.checkpoint_dir / f"combat_model_update_{update_idx}.pt"
        payload = {
            "update_idx": update_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_build_target": self.cfg.game.build_target,
            "device": self.device,
        }
        torch.save(payload, path)
        print(f"[checkpoint] saved: {path}")
        return path

    def save_best_checkpoint(
            self,
            tag: str = "best",
            win_rate: Optional[float] = None,
            reward: Optional[float] = None,
    ) -> Path:
        if win_rate is None:
            win_rate = self.best_eval_win_rate
        if reward is None:
            reward = self.best_eval_reward

        path = self.checkpoint_dir / f"combat_model_{tag}.pt"
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_build_target": self.cfg.game.build_target,
            "device": self.device,
            "best_win_rate": win_rate,
            "best_reward": reward,
            "best_tag": tag,
        }
        torch.save(payload, path)
        print(f"[BEST MODEL {tag}] win_rate={win_rate:.2%} reward={reward:.3f} -> {path}")
        return path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.model.to(self.device)
        print(f"[checkpoint] loaded: {path}")

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    def _is_potion_action(self, action_index: int) -> bool:
        end_turn_idx = (
                self.cfg.combat_obs.max_hand_cards
                + self.cfg.combat_obs.max_hand_cards * self.cfg.combat_obs.max_enemies
        )
        potion_base = end_turn_idx + 1
        return action_index >= potion_base

    def _append_metrics_csv(
            self,
            stat: TrainStats,
            train_potion_actions_total: float = 0.0,
            train_potion_actions_per_episode: float = 0.0,
            eval_metrics: Optional[Dict[str, float]] = None,
            robust_eval_metrics: Optional[Dict[str, float]] = None,
    ) -> None:

        eval_metrics = eval_metrics or {
            "avg_reward": 0.0,
            "avg_len": 0.0,
            "win_rate": 0.0,
            "avg_final_hp_ratio": 0.0,
            "avg_damage_taken": 0.0,
            "avg_win_len": 0.0,
            "potion_actions_total": 0.0,
            "potion_actions_per_episode": 0.0,
        }

        robust_eval_metrics = robust_eval_metrics or {
            "avg_reward": 0.0,
            "avg_len": 0.0,
            "win_rate": 0.0,
            "avg_final_hp_ratio": 0.0,
            "avg_damage_taken": 0.0,
            "avg_win_len": 0.0,
            "potion_actions_total": 0.0,
            "potion_actions_per_episode": 0.0,
        }

        row = {
            "update_idx": stat.update_idx,
            "train_avg_reward": stat.avg_episode_reward,
            "train_avg_len": stat.avg_episode_len,
            "train_win_rate": stat.win_rate,
            "episodes_finished": stat.episodes_finished,
            "truncations": stat.truncations,
            "policy_loss": stat.policy_loss,
            "value_loss": stat.value_loss,
            "entropy": stat.entropy,
            "approx_kl": stat.approx_kl,
            "clip_fraction": stat.clip_fraction,

            "train_potion_actions_total": train_potion_actions_total,
            "train_potion_actions_per_episode": train_potion_actions_per_episode,

            "eval_avg_reward": eval_metrics["avg_reward"],
            "eval_avg_len": eval_metrics["avg_len"],
            "eval_win_rate": eval_metrics["win_rate"],
            "eval_avg_final_hp_ratio": eval_metrics["avg_final_hp_ratio"],
            "eval_avg_damage_taken": eval_metrics["avg_damage_taken"],
            "eval_avg_win_len": eval_metrics["avg_win_len"],
            "eval_potion_actions_total": eval_metrics["potion_actions_total"],
            "eval_potion_actions_per_episode": eval_metrics["potion_actions_per_episode"],

            "robust_eval_avg_reward": robust_eval_metrics["avg_reward"],
            "robust_eval_avg_len": robust_eval_metrics["avg_len"],
            "robust_eval_win_rate": robust_eval_metrics["win_rate"],
            "robust_eval_avg_final_hp_ratio": robust_eval_metrics["avg_final_hp_ratio"],
            "robust_eval_avg_damage_taken": robust_eval_metrics["avg_damage_taken"],
            "robust_eval_avg_win_len": robust_eval_metrics["avg_win_len"],
            "robust_eval_potion_actions_total": robust_eval_metrics["potion_actions_total"],
            "robust_eval_potion_actions_per_episode": robust_eval_metrics["potion_actions_per_episode"],
        }

        file_exists = self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

# =========================================================
# Convenience entrypoint
# =========================================================

def train_combat() -> List[TrainStats]:
    cfg = get_default_config()
    trainer = CombatTrainer(cfg=cfg, seed=cfg.train.seed)
    return trainer.train()


# =========================================================
# Debug main
# =========================================================

if __name__ == "__main__":
    stats = train_combat()
    print(f"\nTraining finished. Collected {len(stats)} updates.")
    if len(stats) > 0:
        print("Last stats:", stats[-1].to_dict())
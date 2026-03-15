from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from slay_rl.config import Config, get_default_config
from slay_rl.features.combat_encoder import CombatEncoder, flatten_combat_obs


LOGIT_NEG = -1e9


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class CombatModel(nn.Module):
    """
    PPO-style combat network.

    Input:
        encoded obs dict from CombatEncoder

    Output:
        {
            "logits": [B, total_actions],
            "value":  [B],
            "probs":  [B, total_actions] (optional helper)
        }
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = cfg or get_default_config()
        self.encoder = CombatEncoder(self.cfg)

        self.obs_dim = self._infer_obs_dim()
        self.total_actions = self.cfg.combat_action.total_actions

        # Shared torso
        self.backbone = nn.Sequential(
            MLPBlock(self.obs_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            MLPBlock(hidden_dim, hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, self.total_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            MLPBlock(hidden_dim, hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    # =========================================================
    # Public forward
    # =========================================================

    def forward(
        self,
        encoded_obs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        encoded_obs values can be shaped:
        - unbatched: [D], [N,M], ...
        - batched:   [B,D], [B,N,M], ...

        Returns:
            logits: [B, A]
            value:  [B]
            probs:  [B, A]
        """
        flat_obs = self._flatten_batch(encoded_obs)
        valid_action_mask = self._extract_action_mask(encoded_obs)

        features = self.backbone(flat_obs)
        raw_logits = self.policy_head(features)
        masked_logits = self._apply_action_mask(raw_logits, valid_action_mask)

        value = self.value_head(features).squeeze(-1)
        probs = torch.softmax(masked_logits, dim=-1)

        return {
            "logits": masked_logits,
            "value": value,
            "probs": probs,
        }

    # =========================================================
    # Action helpers
    # =========================================================

    @torch.no_grad()
    def act(
        self,
        encoded_obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "action": [B],
                "log_prob": [B],
                "value": [B],
                "logits": [B, A],
                "probs": [B, A],
            }
        """
        outputs = self.forward(encoded_obs)
        logits = outputs["logits"]
        value = outputs["value"]
        probs = outputs["probs"]

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "logits": logits,
            "probs": probs,
        }

    def evaluate_actions(
        self,
        encoded_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        For PPO training.

        Args:
            encoded_obs: batch of encoded observations
            actions: [B]

        Returns:
            {
                "log_prob": [B],
                "entropy": [B],
                "value": [B],
                "logits": [B, A],
            }
        """
        outputs = self.forward(encoded_obs)
        logits = outputs["logits"]
        value = outputs["value"]

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
        }

    # =========================================================
    # Encoding convenience
    # =========================================================

    def encode_state(
        self,
        state: Dict,
        device: Optional[torch.device | str] = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        return self.encoder.encode(state, device=str(device))

    @torch.no_grad()
    def act_from_state(
        self,
        state: Dict,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_state(state)
        return self.act(encoded, deterministic=deterministic)

    # =========================================================
    # Internal helpers
    # =========================================================

    def _infer_obs_dim(self) -> int:
        sample_state = {
            "energy": 3,
            "player": {
                "current_hp": 70,
                "max_hp": 80,
                "block": 0,
                "powers": [],
                "relics": [{"name": "Burning Blood"}],
            },
            "hand": [
                {"id": "Strike_R", "cost": 1, "type": "ATTACK"},
                {"id": "Defend_R", "cost": 1, "type": "SKILL"},
            ],
            "draw_pile": [{"id": "Bash"}],
            "discard_pile": [],
            "exhaust_pile": [],
            "monsters": [
                {
                    "name": "Jaw Worm",
                    "current_hp": 40,
                    "max_hp": 42,
                    "block": 0,
                    "intent": "ATTACK",
                    "intent_base_damage": 11,
                    "powers": [],
                }
            ],
            "potions": [],
        }
        encoded = self.encoder.encode(sample_state, device="cpu")
        flat = flatten_combat_obs(encoded)
        return int(flat.shape[0])

    def _flatten_batch(self, encoded_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert encoded dict to [B, obs_dim].
        """
        sample_tensor = encoded_obs["player_scalars"]

        # Unbatched case
        if sample_tensor.dim() == 1:
            flat = flatten_combat_obs(encoded_obs).unsqueeze(0)
            return flat

        # Batched case
        batch_size = sample_tensor.shape[0]
        flats = []
        for i in range(batch_size):
            item = {k: v[i] for k, v in encoded_obs.items()}
            flats.append(flatten_combat_obs(item))
        return torch.stack(flats, dim=0)

    def _extract_action_mask(self, encoded_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = encoded_obs["valid_action_mask"]

        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        return mask

    def _apply_action_mask(
        self,
        logits: torch.Tensor,
        valid_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        valid_action_mask: 1 for valid, 0 for invalid
        """
        masked_logits = logits.masked_fill(valid_action_mask <= 0.0, LOGIT_NEG)

        # Failsafe: if a whole row is invalid, enable dynamic end-turn index
        invalid_rows = torch.all(valid_action_mask <= 0.0, dim=-1)
        if invalid_rows.any():
            masked_logits = masked_logits.clone()

            max_hand_cards = self.cfg.combat_obs.max_hand_cards
            max_enemies = self.cfg.combat_obs.max_enemies
            fallback_idx = max_hand_cards + max_hand_cards * max_enemies

            if fallback_idx >= self.total_actions:
                fallback_idx = 0

            masked_logits[invalid_rows, fallback_idx] = 0.0

        return masked_logits

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Slightly smaller init for policy logits
        last_policy_linear = None
        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear):
                last_policy_linear = m
        if last_policy_linear is not None:
            nn.init.orthogonal_(last_policy_linear.weight, gain=0.01)
            nn.init.zeros_(last_policy_linear.bias)

        last_value_linear = None
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                last_value_linear = m
        if last_value_linear is not None:
            nn.init.orthogonal_(last_value_linear.weight, gain=1.0)
            nn.init.zeros_(last_value_linear.bias)


# =========================================================
# PPO loss helper
# =========================================================

def compute_ppo_loss(
    model: CombatModel,
    batch_obs: Dict[str, torch.Tensor],
    batch_actions: torch.Tensor,
    batch_old_log_probs: torch.Tensor,
    batch_returns: torch.Tensor,
    batch_advantages: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, torch.Tensor]:
    """
    Standard PPO clipped objective.
    """
    eval_out = model.evaluate_actions(batch_obs, batch_actions)
    new_log_probs = eval_out["log_prob"]
    entropy = eval_out["entropy"]
    values = eval_out["value"]

    ratios = torch.exp(new_log_probs - batch_old_log_probs)

    surr1 = ratios * batch_advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, batch_returns)

    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    approx_kl = (batch_old_log_probs - new_log_probs).mean()
    clip_fraction = ((ratios - 1.0).abs() > clip_eps).float().mean()

    return {
        "loss": total_loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.mean().detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
    }


# =========================================================
# Batch collation helper
# =========================================================

def stack_encoded_obs(encoded_list: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack a list of encoded obs dicts into a batch dict.
    """
    if len(encoded_list) == 0:
        raise ValueError("encoded_list is empty")

    keys = encoded_list[0].keys()
    batch = {}
    for key in keys:
        batch[key] = torch.stack([item[key] for item in encoded_list], dim=0)
    return batch


# =========================================================
# Debug run
# =========================================================

if __name__ == "__main__":
    cfg = get_default_config()
    model = CombatModel(cfg)

    sample_state = {
        "energy": 3,
        "player": {
            "current_hp": 68,
            "max_hp": 80,
            "block": 4,
            "powers": [
                {"id": "Strength", "amount": 2},
                {"id": "Dexterity", "amount": 1},
            ],
            "relics": [{"name": "Burning Blood"}],
        },
        "hand": [
            {"id": "Strike_R", "cost": 1, "type": "ATTACK"},
            {"id": "Bash", "cost": 2, "type": "ATTACK"},
            {"id": "Defend_R", "cost": 1, "type": "SKILL"},
            {"id": "Inflame", "cost": 1, "type": "POWER"},
        ],
        "draw_pile": [{"id": "Strike_R"}, {"id": "Defend_R"}],
        "discard_pile": [],
        "exhaust_pile": [],
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 38,
                "max_hp": 42,
                "block": 0,
                "intent": "ATTACK",
                "intent_base_damage": 11,
                "powers": [],
            },
            {
                "name": "Cultist",
                "current_hp": 45,
                "max_hp": 48,
                "block": 0,
                "intent": "BUFF",
                "intent_base_damage": 0,
                "powers": [],
            },
        ],
        "potions": [],
    }

    encoded = model.encode_state(sample_state, device="cpu")
    outputs = model.forward(encoded)
    act_out = model.act(encoded, deterministic=False)

    print("obs_dim:", model.obs_dim)
    print("logits shape:", tuple(outputs["logits"].shape))
    print("value shape:", tuple(outputs["value"].shape))
    print("sampled action:", act_out["action"])
    print("log_prob:", act_out["log_prob"])
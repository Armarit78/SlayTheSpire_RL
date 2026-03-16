from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from slay_rl.config import Config, get_default_config
from slay_rl.features.combat_encoder import CombatEncoder


LOGIT_NEG = -1e9


# =========================================================
# Basic blocks
# =========================================================

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


class SlotEncoder(nn.Module):
    """
    Encode per-slot features (cards, enemies, potions).
    Input:  [B, N, D]
    Output: [B, N, H]
    """
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(in_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.view(b * n, d)
        x = self.net(x)
        x = x.view(b, n, -1)
        return x


class ScalarEncoder(nn.Module):
    """
    Encode single-vector features.
    Input:  [B, D]
    Output: [B, H]
    """
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(in_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedMeanPool(nn.Module):
    """
    Mean pool with binary mask.
    values: [B, N, H]
    mask:   [B, N]
    out:    [B, H]
    """
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)  # [B, N, 1]
        weighted = values * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return weighted.sum(dim=1) / denom


class MaskedMaxPool(nn.Module):
    """
    Max pool with binary mask.
    values: [B, N, H]
    mask:   [B, N]
    out:    [B, H]
    """
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)  # [B, N, 1]
        neg = torch.full_like(values, -1e9)
        masked_values = torch.where(mask > 0.0, values, neg)
        pooled = masked_values.max(dim=1).values

        # si tout est masqué, on renvoie 0
        no_valid = (mask.sum(dim=1) <= 0.0).expand_as(pooled)
        pooled = torch.where(no_valid, torch.zeros_like(pooled), pooled)
        return pooled


# =========================================================
# Structured combat model
# =========================================================

class CombatModel(nn.Module):
    """
    Structured PPO combat network.

    Instead of flattening everything immediately, this model:
    - encodes player scalars
    - encodes combat context
    - encodes deck/discard/exhaust/relic summary
    - encodes hand slots separately
    - encodes enemy slots separately
    - encodes potion slots separately
    - pools slot groups with masks
    - fuses everything into a shared combat representation
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        hidden_dim: int = 384,
        slot_hidden_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = cfg or get_default_config()
        self.encoder = CombatEncoder(self.cfg)
        self.total_actions = self.cfg.combat_action.total_actions

        # dims from encoder/config
        self.player_dim = self.cfg.combat_obs.player_scalar_dim
        self.enemy_dim = self.cfg.combat_obs.enemy_scalar_dim
        self.potion_dim = self.cfg.combat_obs.potion_scalar_dim
        self.context_dim = self.cfg.combat_obs.combat_context_dim

        self.hand_dim = self.encoder.card_feature_dim
        self.deck_dim = self.cfg.combat_obs.card_vocab_size
        self.relic_dim = self.cfg.combat_obs.relic_vocab_size

        # scalar/global encoders
        self.player_encoder = ScalarEncoder(self.player_dim, hidden_dim, dropout=dropout)
        self.context_encoder = ScalarEncoder(self.context_dim, hidden_dim, dropout=dropout)

        self.piles_encoder = ScalarEncoder(
            self.deck_dim * 3 + self.relic_dim,
            hidden_dim,
            dropout=dropout,
        )

        # slot encoders
        self.hand_encoder = SlotEncoder(self.hand_dim, slot_hidden_dim, dropout=dropout)
        self.enemy_encoder = SlotEncoder(self.enemy_dim, slot_hidden_dim, dropout=dropout)
        self.potion_encoder = SlotEncoder(self.potion_dim, slot_hidden_dim, dropout=dropout)

        # pooling
        self.mean_pool = MaskedMeanPool()
        self.max_pool = MaskedMaxPool()

        # project pooled slot groups into common hidden space
        self.hand_pool_proj = ScalarEncoder(slot_hidden_dim * 2, hidden_dim, dropout=dropout)
        self.enemy_pool_proj = ScalarEncoder(slot_hidden_dim * 2, hidden_dim, dropout=dropout)
        self.potion_pool_proj = ScalarEncoder(slot_hidden_dim * 2, hidden_dim, dropout=dropout)

        # fusion
        fusion_in_dim = hidden_dim * 6
        self.fusion = nn.Sequential(
            MLPBlock(fusion_in_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )

        # heads
        self.policy_head = nn.Sequential(
            MLPBlock(hidden_dim, hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, self.total_actions),
        )

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
        obs = self._ensure_batched(encoded_obs)

        valid_action_mask = self._extract_action_mask(obs)

        player_repr = self.player_encoder(obs["player_scalars"])
        context_repr = self.context_encoder(obs["combat_context"])

        piles_input = torch.cat(
            [
                obs["deck_counts"],
                obs["discard_counts"],
                obs["exhaust_counts"],
                obs["relics"],
            ],
            dim=-1,
        )
        piles_repr = self.piles_encoder(piles_input)

        hand_repr = self.hand_encoder(obs["hand_cards"])
        enemy_repr = self.enemy_encoder(obs["enemies"])
        potion_repr = self.potion_encoder(obs["potions"])

        hand_mean = self.mean_pool(hand_repr, obs["hand_mask"])
        hand_max = self.max_pool(hand_repr, obs["hand_mask"])
        hand_summary = self.hand_pool_proj(torch.cat([hand_mean, hand_max], dim=-1))

        enemy_mean = self.mean_pool(enemy_repr, obs["enemy_mask"])
        enemy_max = self.max_pool(enemy_repr, obs["enemy_mask"])
        enemy_summary = self.enemy_pool_proj(torch.cat([enemy_mean, enemy_max], dim=-1))

        potion_mean = self.mean_pool(potion_repr, obs["potion_mask"])
        potion_max = self.max_pool(potion_repr, obs["potion_mask"])
        potion_summary = self.potion_pool_proj(torch.cat([potion_mean, potion_max], dim=-1))

        fused = torch.cat(
            [
                player_repr,
                context_repr,
                piles_repr,
                hand_summary,
                enemy_summary,
                potion_summary,
            ],
            dim=-1,
        )

        features = self.fusion(fused)
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

    def _ensure_batched(self, encoded_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert encoded dict to batched tensors.
        """
        out: Dict[str, torch.Tensor] = {}
        for k, v in encoded_obs.items():
            if k in {"hand_cards", "enemies", "potions"}:
                if v.dim() == 2:
                    out[k] = v.unsqueeze(0)
                else:
                    out[k] = v
            elif k in {"hand_mask", "enemy_mask", "potion_mask"}:
                if v.dim() == 1:
                    out[k] = v.unsqueeze(0)
                else:
                    out[k] = v
            elif k == "valid_action_mask":
                if v.dim() == 1:
                    out[k] = v.unsqueeze(0)
                else:
                    out[k] = v
            else:
                if v.dim() == 1:
                    out[k] = v.unsqueeze(0)
                else:
                    out[k] = v
        return out

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
    eval_out = model.evaluate_actions(batch_obs, batch_actions)
    new_log_probs = eval_out["log_prob"]
    entropy = eval_out["entropy"]
    values = eval_out["value"]

    ratios = torch.exp(new_log_probs - batch_old_log_probs)

    surr1 = ratios * batch_advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Huber loss à la place de MSE
    value_loss = F.smooth_l1_loss(values, batch_returns)

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
        raise ValueError("encoded_list must not be empty")

    keys = encoded_list[0].keys()
    out: Dict[str, torch.Tensor] = {}

    for key in keys:
        out[key] = torch.stack([item[key] for item in encoded_list], dim=0)

    return out


# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":
    cfg = get_default_config()
    model = CombatModel(cfg, hidden_dim=256, slot_hidden_dim=128, dropout=0.0)

    sample_state = {
        "turn": 2,
        "energy": 3,
        "player": {
            "current_hp": 61,
            "max_hp": 80,
            "block": 2,
            "powers": [
                {"id": "Strength", "amount": 1},
                {"id": "Rage", "amount": 3},
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
        "combat_meta": {
            "cards_played_this_turn": 1,
            "attacks_played_this_turn": 1,
            "double_tap_charges": 0,
            "cannot_draw_more_this_turn": False,
            "last_x_energy_spent": 0,
            "attack_counter": 1,
            "next_attack_double": False,
            "first_attack_done": True,
            "is_elite": False,
            "is_boss": False,
        },
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 12,
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
                "powers": [{"id": "Ritual", "amount": 3}],
            },
        ],
        "potions": [
            {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Dexterity Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Uncommon"},
        ],
    }

    encoded = model.encode_state(sample_state, device="cpu")
    out = model.forward(encoded)
    act = model.act(encoded, deterministic=False)

    print("logits shape:", tuple(out["logits"].shape))
    print("value shape:", tuple(out["value"].shape))
    print("probs shape:", tuple(out["probs"].shape))
    print("sampled action:", int(act["action"].item()))
    print("sampled value:", float(act["value"].item()))
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from slay_rl.config import Config, get_default_config
from slay_rl.features.combat_encoder import CombatEncoder, ParsedCard, ParsedEnemy
from slay_rl.models.combat_model import CombatModel


# =========================================================
# Action command representation
# =========================================================

@dataclass
class CombatCommand:
    """
    Normalized command returned by the combat agent.

    command_type:
        - "play_card"
        - "end_turn"
        - "use_potion"
        - "choose_hand_card"
        - "choose_option"
        - "choose_discard_target"
        - "choose_exhaust_target"
        - "wait"
    """
    command_type: str
    hand_index: Optional[int] = None
    target_index: Optional[int] = None
    potion_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_type": self.command_type,
            "hand_index": self.hand_index,
            "target_index": self.target_index,
            "potion_index": self.potion_index,
        }

# =========================================================
# Main combat agent
# =========================================================

class CombatAgent:
    """
    Learned combat agent backed by CombatModel.

    Responsibilities:
    - encode state
    - sample / choose an action
    - decode the flat action index into a structured combat command
    - provide training-ready outputs
    """

    def __init__(
        self,
        model: CombatModel,
        cfg: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg or get_default_config()
        self.model = model
        self.encoder = CombatEncoder(self.cfg)

        if device is None:
            try:
                device = next(model.parameters()).device.type
            except StopIteration:
                device = self.cfg.train.device
        self.device = device

        self.max_hand_cards = self.cfg.combat_obs.max_hand_cards
        self.max_enemies = self.cfg.combat_obs.max_enemies
        self.total_actions = self.cfg.combat_action.total_actions

    # =====================================================
    # Public API
    # =====================================================

    @torch.no_grad()
    def choose_action(
        self,
        state: Dict[str, Any],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns a rich dictionary:
        {
            "action_index": int,
            "command": CombatCommand,
            "log_prob": float,
            "value": float,
            "probs": tensor[A],
            "valid_action_mask": tensor[A],
        }
        """
        encoded = self.encoder.encode(state, device=self.device)
        act_out = self.model.act(encoded, deterministic=deterministic)

        action_index = int(act_out["action"].item())
        command = self.decode_action_index(action_index, state)

        return {
            "action_index": action_index,
            "command": command,
            "log_prob": float(act_out["log_prob"].item()),
            "value": float(act_out["value"].item()),
            "probs": act_out["probs"].squeeze(0).detach().cpu(),
            "valid_action_mask": encoded["valid_action_mask"].detach().cpu(),
        }

    def choose_command(
        self,
        state: Dict[str, Any],
        deterministic: bool = False,
    ) -> CombatCommand:
        result = self.choose_action(state, deterministic=deterministic)
        return result["command"]

    def decode_action_index(
            self,
            action_index: int,
            state: Dict[str, Any],
    ) -> CombatCommand:
        hand = self.encoder._parse_hand(state)
        enemies = self.encoder._parse_enemies(state)

        max_potions = self.cfg.combat_obs.max_potions
        max_choose_hand = self.cfg.combat_action.max_choose_hand_actions
        max_choose_option = self.cfg.combat_action.max_choose_option_actions
        max_choose_discard = self.cfg.combat_action.max_choose_discard_actions
        max_choose_exhaust = self.cfg.combat_action.max_choose_exhaust_actions

        targeted_base = self.max_hand_cards
        targeted_size = self.max_hand_cards * self.max_enemies
        end_turn_idx = targeted_base + targeted_size

        potion_base = end_turn_idx + 1
        potion_target_base = potion_base + max_potions
        potion_target_size = max_potions * self.max_enemies

        choose_hand_base = potion_target_base + potion_target_size
        choose_option_base = choose_hand_base + max_choose_hand
        choose_discard_base = choose_option_base + max_choose_option
        choose_exhaust_base = choose_discard_base + max_choose_discard

        pending = self.encoder._extract_pending_choice(state)
        choice_type = self.encoder._normalize_choice_type(pending)
        valid_hand_indices = self.encoder._pending_choice_hand_indices(state, pending) if pending else []
        options = self.encoder._pending_choice_options(pending) if pending else []

        if 0 <= action_index < self.max_hand_cards:
            hand_index = action_index
            if hand_index < len(hand) and hand[hand_index].is_playable:
                return CombatCommand(
                    command_type="play_card",
                    hand_index=hand_index,
                    target_index=None,
                )
            return CombatCommand(command_type="wait")

        if targeted_base <= action_index < targeted_base + targeted_size:
            relative = action_index - targeted_base
            hand_index = relative // self.max_enemies
            target_index = relative % self.max_enemies

            if (
                    hand_index < len(hand)
                    and target_index < len(enemies)
                    and hand[hand_index].is_playable
                    and enemies[target_index].alive == 1
            ):
                return CombatCommand(
                    command_type="play_card",
                    hand_index=hand_index,
                    target_index=target_index,
                )
            return CombatCommand(command_type="wait")

        if action_index == end_turn_idx:
            return CombatCommand(command_type="end_turn")

        if potion_base <= action_index < potion_base + max_potions:
            return CombatCommand(
                command_type="use_potion",
                potion_index=action_index - potion_base,
                target_index=None,
            )

        if potion_target_base <= action_index < potion_target_base + potion_target_size:
            relative = action_index - potion_target_base
            potion_index = relative // self.max_enemies
            target_index = relative % self.max_enemies

            if target_index < len(enemies) and enemies[target_index].alive == 1:
                return CombatCommand(
                    command_type="use_potion",
                    potion_index=potion_index,
                    target_index=target_index,
                )
            return CombatCommand(command_type="wait")

        if choose_hand_base <= action_index < choose_hand_base + max_choose_hand:
            hand_index = action_index - choose_hand_base
            if choice_type == "choose_hand_card" and hand_index in valid_hand_indices:
                return CombatCommand(command_type="choose_hand_card", hand_index=hand_index)
            return CombatCommand(command_type="wait")

        if choose_option_base <= action_index < choose_option_base + max_choose_option:
            option_index = action_index - choose_option_base
            if choice_type == "choose_option" and option_index < len(options):
                return CombatCommand(command_type="choose_option", target_index=option_index)
            return CombatCommand(command_type="wait")

        if choose_discard_base <= action_index < choose_discard_base + max_choose_discard:
            hand_index = action_index - choose_discard_base
            if choice_type == "choose_discard_target" and hand_index in valid_hand_indices:
                return CombatCommand(command_type="choose_discard_target", hand_index=hand_index)
            return CombatCommand(command_type="wait")

        if choose_exhaust_base <= action_index < choose_exhaust_base + max_choose_exhaust:
            hand_index = action_index - choose_exhaust_base
            if choice_type == "choose_exhaust_target" and hand_index in valid_hand_indices:
                return CombatCommand(command_type="choose_exhaust_target", hand_index=hand_index)
            return CombatCommand(command_type="wait")

        return CombatCommand(command_type="wait")

    def command_to_spirecomm_action(
        self,
        command: CombatCommand,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generic bridge payload.

        This is intentionally neutral because exact spirecomm integration
        can vary depending on the coordinator / communication wrapper used.

        You can later adapt this method if your real coordinator expects
        a specific action object or schema.
        """
        if command.command_type == "play_card":
            return {
                "action_type": "play_card",
                "card_index": command.hand_index,
                "target_index": command.target_index,
            }

        if command.command_type == "end_turn":
            return {
                "action_type": "end_turn",
            }

        if command.command_type == "use_potion":
            return {
                "action_type": "use_potion",
                "potion_index": command.potion_index,
                "target_index": command.target_index,
            }

        if command.command_type in {
            "choose_hand_card",
            "choose_option",
            "choose_discard_target",
            "choose_exhaust_target",
        }:
            return {
                "action_type": "choose",
                "choice_type": command.command_type,
                "hand_index": command.hand_index,
                "target_index": command.target_index,
            }

        return {
            "action_type": "wait",
        }

    def choose_spirecomm_action(
        self,
        state: Dict[str, Any],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        result = self.choose_action(state, deterministic=deterministic)
        return self.command_to_spirecomm_action(result["command"], state)

    # =====================================================
    # Training helper
    # =====================================================

    @torch.no_grad()
    def rollout_step(
        self,
        state: Dict[str, Any],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Useful during PPO collection.

        Returns:
        {
            "encoded_obs": dict[tensor],
            "action": int,
            "log_prob": float,
            "value": float,
            "command": CombatCommand,
        }
        """
        encoded = self.encoder.encode(state, device=self.device)
        act_out = self.model.act(encoded, deterministic=deterministic)

        action_index = int(act_out["action"].item())
        command = self.decode_action_index(action_index, state)

        return {
            "encoded_obs": encoded,
            "action": action_index,
            "log_prob": float(act_out["log_prob"].item()),
            "value": float(act_out["value"].item()),
            "command": command,
        }


# =========================================================
# Rule-based baseline combat agent
# =========================================================

class RuleBasedCombatAgent:
    """
    Simple baseline for debugging.

    Policy:
    1. If lethal targeted attack exists, use it
    2. Else play best affordable attack
    3. Else play best affordable defense/power
    4. Else end turn
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_default_config()
        self.encoder = CombatEncoder(self.cfg)

    def choose_command(self, state: Dict[str, Any]) -> CombatCommand:
        hand = self.encoder._parse_hand(state)
        enemies = self.encoder._parse_enemies(state)

        alive_targets = [i for i, e in enumerate(enemies) if e.alive == 1]
        if len(alive_targets) == 0:
            return CombatCommand(command_type="end_turn")

        # 1) lethal targeted attack
        for h_idx, card in enumerate(hand):
            if not card.is_playable:
                continue
            if not card.has_target:
                continue

            estimated_damage = estimate_card_damage(card, state=state)
            if estimated_damage <= 0:
                continue

            for e_idx in alive_targets:
                enemy_hp = enemies[e_idx].hp + enemies[e_idx].block
                if estimated_damage >= enemy_hp:
                    return CombatCommand(
                        command_type="play_card",
                        hand_index=h_idx,
                        target_index=e_idx,
                    )

        # 2) best affordable attack
        best_attack = None
        best_attack_score = -1.0
        best_attack_target = None

        for h_idx, card in enumerate(hand):
            if not card.is_playable:
                continue

            score = score_card_basic(card, state=state)
            if score <= 0:
                continue

            if card.has_target:
                target_idx = choose_lowest_hp_enemy(enemies)
                if target_idx is None:
                    continue
                if score > best_attack_score:
                    best_attack_score = score
                    best_attack = h_idx
                    best_attack_target = target_idx
            else:
                if score > best_attack_score:
                    best_attack_score = score
                    best_attack = h_idx
                    best_attack_target = None

        if best_attack is not None:
            return CombatCommand(
                command_type="play_card",
                hand_index=best_attack,
                target_index=best_attack_target,
            )

        # 3) best non-attack useful card
        best_support = None
        best_support_score = -1.0

        for h_idx, card in enumerate(hand):
            if not card.is_playable:
                continue
            score = score_non_attack_basic(card, state=state)
            if score > best_support_score:
                best_support_score = score
                best_support = h_idx

        if best_support is not None and best_support_score > 0:
            return CombatCommand(
                command_type="play_card",
                hand_index=best_support,
                target_index=None,
            )

        # 4) end turn
        return CombatCommand(command_type="end_turn")

    def choose_action_index(self, state: Dict[str, Any]) -> int:
        command = self.choose_command(state)
        return encode_command_to_action_index(command, self.cfg)


# =========================================================
# Command <-> flat action helpers
# =========================================================

def encode_command_to_action_index(
    command: CombatCommand,
    cfg: Optional[Config] = None,
) -> int:
    cfg = cfg or get_default_config()
    max_hand_cards = cfg.combat_obs.max_hand_cards
    max_enemies = cfg.combat_obs.max_enemies
    max_potions = cfg.combat_obs.max_potions

    max_choose_hand = cfg.combat_action.max_choose_hand_actions
    max_choose_option = cfg.combat_action.max_choose_option_actions
    max_choose_discard = cfg.combat_action.max_choose_discard_actions
    max_choose_exhaust = cfg.combat_action.max_choose_exhaust_actions

    targeted_base = max_hand_cards
    targeted_size = max_hand_cards * max_enemies
    end_turn_idx = targeted_base + targeted_size

    potion_base = end_turn_idx + 1
    potion_target_base = potion_base + max_potions
    potion_target_size = max_potions * max_enemies

    choose_hand_base = potion_target_base + potion_target_size
    choose_option_base = choose_hand_base + max_choose_hand
    choose_discard_base = choose_option_base + max_choose_option
    choose_exhaust_base = choose_discard_base + max_choose_discard

    if command.command_type == "play_card":
        if command.hand_index is None:
            return end_turn_idx

        if command.target_index is None:
            if 0 <= command.hand_index < max_hand_cards:
                return command.hand_index
            return end_turn_idx

        if (
            0 <= command.hand_index < max_hand_cards
            and 0 <= command.target_index < max_enemies
        ):
            return targeted_base + command.hand_index * max_enemies + command.target_index
        return end_turn_idx

    if command.command_type == "end_turn":
        return end_turn_idx

    if command.command_type == "use_potion":
        if command.potion_index is None or not (0 <= command.potion_index < max_potions):
            return end_turn_idx

        if command.target_index is None:
            return potion_base + command.potion_index

        if 0 <= command.target_index < max_enemies:
            return potion_target_base + command.potion_index * max_enemies + command.target_index

        return end_turn_idx

    if command.command_type == "choose_hand_card":
        if command.hand_index is not None and 0 <= command.hand_index < max_choose_hand:
            return choose_hand_base + command.hand_index
        return end_turn_idx

    if command.command_type == "choose_option":
        if command.target_index is not None and 0 <= command.target_index < max_choose_option:
            return choose_option_base + command.target_index
        return end_turn_idx

    if command.command_type == "choose_discard_target":
        if command.hand_index is not None and 0 <= command.hand_index < max_choose_discard:
            return choose_discard_base + command.hand_index
        return end_turn_idx

    if command.command_type == "choose_exhaust_target":
        if command.hand_index is not None and 0 <= command.hand_index < max_choose_exhaust:
            return choose_exhaust_base + command.hand_index
        return end_turn_idx

    return end_turn_idx

# =========================================================
# Heuristics for rule-based baseline
# =========================================================

BAD_STATUS_OR_CURSE = {
    "Wound",
    "Dazed",
    "Burn",
    "Slimed",
    "Void",
    "AscendersBane",
    "Clumsy",
    "Curse of the Bell",
    "Decay",
    "Doubt",
    "Injury",
    "Normality",
    "Pain",
    "Parasite",
    "Pride",
    "Regret",
    "Shame",
    "Writhe",
}

TARGETED_ATTACK_DAMAGE = {
    "Strike_R": 6,
    "Bash": 8,
    "Anger": 6,
    "Body Slam": 7,   # estimation simple, géré plus bas aussi
    "Clash": 14,
    "Clothesline": 12,
    "Headbutt": 9,
    "Heavy Blade": 14,
    "Iron Wave": 5,
    "Perfected Strike": 10,
    "Pommel Strike": 9,
    "Twin Strike": 10,
    "Sword Boomerang": 9,
    "Carnage": 20,
    "Dropkick": 5,
    "Hemokinesis": 15,
    "Pummel": 8,
    "Rampage": 8,
    "Reckless Charge": 7,
    "Searing Blow": 12,
    "Sever Soul": 16,
    "Uppercut": 13,
    "Bludgeon": 32,
    "Feed": 10,
    "Fiend Fire": 20,
    "Wild Strike": 12,
    "Blood for Blood": 18,
}

NON_TARGET_DAMAGE = {
    "Cleave": 8,
    "Thunderclap": 4,
    "Whirlwind": 6,
    "Immolate": 21,
    "Reaper": 4,
}

DEFENSE_OR_POWER_SCORE = {
    "Defend_R": 5,
    "Shrug It Off": 11,
    "Ghostly Armor": 10,
    "Flame Barrier": 14,
    "Impervious": 30,
    "Inflame": 10,
    "Metallicize": 9,
    "Barricade": 13,
    "Demon Form": 15,
    "Offering": 16,
    "Battle Trance": 11,
    "Armaments": 8,
    "Shockwave": 14,
    "Feel No Pain": 10,
    "Corruption": 12,
    "Power Through": 13,
    "Entrench": 9,
    "Rage": 6,
    "Evolve": 6,
    "Combust": 8,
    "Dark Embrace": 8,
    "Juggernaut": 10,
    "Brutality": 8,
}

DRAW_CARDS = {
    "Pommel Strike",
    "Shrug It Off",
    "Battle Trance",
    "Burning Pact",
    "Warcry",
    "Offering",
    "Dropkick",
}

WEAK_CARDS = {
    "Clothesline",
    "Uppercut",
    "Shockwave",
    "Intimidate",
}

VULN_CARDS = {
    "Bash",
    "Thunderclap",
    "Uppercut",
    "Shockwave",
}

HP_LOSS_CARDS = {
    "Hemokinesis",
    "Bloodletting",
    "Offering",
}

EXHAUST_SETUP_CARDS = {
    "Burning Pact",
    "Second Wind",
    "True Grit",
    "Fiend Fire",
    "Sever Soul",
}

BLOCK_CARDS = {
    "Defend_R",
    "Shrug It Off",
    "Ghostly Armor",
    "Flame Barrier",
    "Impervious",
    "Power Through",
    "Entrench",
    "Iron Wave",
    "Armaments",
    "True Grit",
    "Sentinel",
}

def estimate_card_damage(card: ParsedCard, state: Optional[Dict[str, Any]] = None) -> float:
    if card.card_id == "Body Slam" and state is not None:
        block = float(state.get("player", {}).get("block", 0))
        dmg = block
    elif card.card_id in TARGETED_ATTACK_DAMAGE:
        dmg = TARGETED_ATTACK_DAMAGE[card.card_id]
    elif card.card_id in NON_TARGET_DAMAGE:
        dmg = NON_TARGET_DAMAGE[card.card_id]
    else:
        dmg = 0.0

    if card.upgraded:
        dmg *= 1.2

    return float(dmg)


def score_card_basic(card: ParsedCard, state: Optional[Dict[str, Any]] = None) -> float:
    """
    Score pour cartes offensives.
    """
    if card.card_id in BAD_STATUS_OR_CURSE:
        return -100.0

    base = estimate_card_damage(card, state=state)

    # bonus utilitaires offensifs
    if card.card_id in DRAW_CARDS:
        base += 2.0
    if card.card_id in WEAK_CARDS:
        base += 2.5
    if card.card_id in VULN_CARDS:
        base += 3.0

    # malus cartes qui coûtent de la vie
    if card.card_id in HP_LOSS_CARDS:
        base -= 2.0

    # exhaust setup modéré
    if card.card_id in EXHAUST_SETUP_CARDS:
        base += 1.0

    if base > 0:
        if card.cost > 0 and card.cost < 99:
            return base / max(card.cost, 1)
        if card.cost == 0:
            return base + 2.0
        if card.cost == -1:
            return base + 1.0
        return base

    if card.has_target and card.is_playable:
        return 1.0

    return 0.0


def score_non_attack_basic(card: ParsedCard, state: Optional[Dict[str, Any]] = None) -> float:
    if card.card_id in BAD_STATUS_OR_CURSE:
        return -100.0

    score = float(DEFENSE_OR_POWER_SCORE.get(card.card_id, 0.0))

    if card.card_id in BLOCK_CARDS:
        score += 1.5
    if card.card_id in DRAW_CARDS:
        score += 1.5
    if card.card_id in WEAK_CARDS or card.card_id in VULN_CARDS:
        score += 1.0

    if card.upgraded:
        score += 1.0

    if card.ethereal:
        score -= 0.5

    if card.exhausts:
        score -= 0.25

    if card.card_id in HP_LOSS_CARDS:
        score -= 2.0

    return score


def choose_lowest_hp_enemy(enemies: List[ParsedEnemy]) -> Optional[int]:
    best_idx = None
    best_hp = None

    for i, enemy in enumerate(enemies):
        if enemy.alive != 1:
            continue
        hp = enemy.hp + enemy.block
        if best_hp is None or hp < best_hp:
            best_hp = hp
            best_idx = i

    return best_idx


# =========================================================
# Optional adapter for future spirecomm action objects
# =========================================================

class SpireCommActionAdapter:
    """
    Thin adapter layer to keep your project independent from the exact
    spirecomm action API.

    For now it returns plain dictionaries.
    Later, if you use actual spirecomm action classes, update only here.
    """

    def to_external_action(
        self,
        command: CombatCommand,
        state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if command.command_type == "play_card":
            return {
                "action_type": "play_card",
                "card_index": command.hand_index,
                "target_index": command.target_index,
            }

        if command.command_type == "end_turn":
            return {"action_type": "end_turn"}

        if command.command_type == "use_potion":
            return {
                "action_type": "use_potion",
                "potion_index": command.potion_index,
                "target_index": command.target_index,
            }

        if command.command_type == "choose_option":
            return {
                "action_type": "choose",
                "choice_index": command.target_index,
            }

        if command.command_type in {"choose_hand_card", "choose_discard_target", "choose_exhaust_target"}:
            return {
                "action_type": "choose",
                "choice_index": command.hand_index,
            }

        return {"action_type": "wait"}


# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":
    cfg = get_default_config()
    model = CombatModel(cfg, hidden_dim=256, dropout=0.0)
    agent = CombatAgent(model, cfg=cfg, device="cpu")
    rule_agent = RuleBasedCombatAgent(cfg)

    sample_state = {
        "energy": 3,
        "player": {
            "current_hp": 61,
            "max_hp": 80,
            "block": 2,
            "powers": [{"id": "Strength", "amount": 1}],
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
                "current_hp": 12,
                "max_hp": 40,
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
        "potions": [
            {"name": "Fire Potion", "usable": True, "empty": False},
        ],
    }

    learned = agent.choose_action(sample_state, deterministic=False)
    print("=== Learned agent ===")
    print("Action index:", learned["action_index"])
    print("Command:", learned["command"])
    print("Spire payload:", agent.command_to_spirecomm_action(learned["command"], sample_state))

    baseline_cmd = rule_agent.choose_command(sample_state)
    baseline_idx = rule_agent.choose_action_index(sample_state)
    print("\n=== Rule-based agent ===")
    print("Command:", baseline_cmd)
    print("Action index:", baseline_idx)
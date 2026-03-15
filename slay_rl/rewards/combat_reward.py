from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from slay_rl.config import Config, get_default_config


# =========================================================
# Reward output
# =========================================================

@dataclass
class CombatRewardOutput:
    total_reward: float
    damage_dealt_reward: float
    damage_taken_reward: float
    kill_bonus: float
    win_bonus: float
    lose_penalty: float
    illegal_action_penalty: float
    step_penalty: float
    status_curse_hand_reward: float
    self_hp_loss_from_bad_cards_penalty: float
    potion_use_penalty: float

    survival_reward: float
    block_reward: float
    debuff_reward: float
    buff_reward: float
    threat_reward: float

    player_hp_before: float
    player_hp_after: float
    enemy_total_hp_before: float
    enemy_total_hp_after: float
    enemies_killed: int
    combat_won: bool
    combat_lost: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "damage_dealt_reward": self.damage_dealt_reward,
            "damage_taken_reward": self.damage_taken_reward,
            "status_curse_hand_reward": self.status_curse_hand_reward,
            "self_hp_loss_from_bad_cards_penalty": self.self_hp_loss_from_bad_cards_penalty,
            "potion_use_penalty": self.potion_use_penalty,
            "survival_reward": self.survival_reward,
            "block_reward": self.block_reward,
            "debuff_reward": self.debuff_reward,
            "buff_reward": self.buff_reward,
            "threat_reward": self.threat_reward,
            "kill_bonus": self.kill_bonus,
            "win_bonus": self.win_bonus,
            "lose_penalty": self.lose_penalty,
            "illegal_action_penalty": self.illegal_action_penalty,
            "step_penalty": self.step_penalty,
            "player_hp_before": self.player_hp_before,
            "player_hp_after": self.player_hp_after,
            "enemy_total_hp_before": self.enemy_total_hp_before,
            "enemy_total_hp_after": self.enemy_total_hp_after,
            "enemies_killed": self.enemies_killed,
            "combat_won": self.combat_won,
            "combat_lost": self.combat_lost,
        }


# =========================================================
# Calculator
# =========================================================

class CombatRewardCalculator:
    """
    Reward shaping for combat.

    Main idea:
    - reward damage dealt to enemies
    - penalize damage taken
    - bonus for killing enemies
    - strong bonus for winning combat
    - strong penalty for losing combat
    - penalty for illegal actions
    - small per-step penalty to avoid stalling
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_default_config()
        self.r_cfg = self.cfg.combat_reward

    # =====================================================
    # Public API
    # =====================================================

    def compute(
            self,
            prev_state: Dict[str, Any],
            next_state: Dict[str, Any],
            action_info: Optional[Dict[str, Any]] = None,
    ) -> CombatRewardOutput:
        action_info = action_info or {}

        player_hp_before = self._get_player_hp(prev_state)
        player_hp_after = self._get_player_hp(next_state)
        player_max_hp_before = max(1.0, self._get_player_max_hp(prev_state))
        player_max_hp_after = max(1.0, self._get_player_max_hp(next_state))

        enemy_total_hp_before = self._get_total_enemy_effective_hp(prev_state)
        enemy_total_hp_after = self._get_total_enemy_effective_hp(next_state)

        damage_dealt = max(0.0, enemy_total_hp_before - enemy_total_hp_after)
        damage_taken = max(0.0, player_hp_before - player_hp_after)

        damage_dealt_reward = damage_dealt * self.r_cfg.damage_dealt_scale
        damage_taken_reward = damage_taken * self.r_cfg.damage_taken_scale

        enemies_killed = self._count_newly_killed_enemies(prev_state, next_state)
        kill_bonus = enemies_killed * self.r_cfg.kill_enemy_bonus

        combat_won = self._is_combat_won(next_state)
        combat_lost = self._is_combat_lost(next_state)

        win_bonus = self.r_cfg.win_combat_bonus if combat_won else 0.0
        lose_penalty = self.r_cfg.lose_combat_penalty if combat_lost else 0.0

        illegal_action_penalty = (
            self.r_cfg.illegal_action_penalty
            if action_info.get("illegal_action", False)
            else 0.0
        )

        step_penalty = self._compute_step_penalty(action_info, combat_won, combat_lost)

        # reward/penalty related to status+curse management
        bad_hand_before = self._count_bad_cards_in_hand(prev_state)
        bad_hand_after = self._count_bad_cards_in_hand(next_state)
        status_curse_hand_reward = 0.05 * float(bad_hand_before - bad_hand_after)

        # exact hp loss source from env
        hp_loss_breakdown = next_state.get("hp_loss_breakdown", {})
        burn_loss = float(hp_loss_breakdown.get("burn", 0.0))
        pain_loss = float(hp_loss_breakdown.get("pain", 0.0))
        decay_loss = float(hp_loss_breakdown.get("decay", 0.0))

        self_hp_loss_from_bad_cards_penalty = -0.03 * (burn_loss + pain_loss + decay_loss)

        # shared derived quantities
        hp_ratio_before = player_hp_before / player_max_hp_before
        hp_ratio_after = player_hp_after / player_max_hp_after
        incoming_before = self._estimate_incoming_damage(prev_state)

        # discourage wasting potions too early
        potions_before = self._count_non_empty_potions(prev_state)
        potions_after = self._count_non_empty_potions(next_state)
        potions_used = max(0, potions_before - potions_after)

        if potions_used > 0:
            if hp_ratio_before < 0.35 or incoming_before >= 12:
                potion_use_penalty = -0.002 * potions_used
            else:
                potion_use_penalty = -0.01 * potions_used
        else:
            potion_use_penalty = 0.0

        # survival reward
        survival_reward = 0.05 * (hp_ratio_after - hp_ratio_before)

        # useful block reward
        player_block_before = self._get_player_block(prev_state)
        player_block_after = self._get_player_block(next_state)
        block_gain = max(0.0, player_block_after - player_block_before)

        if incoming_before > 0.0:
            useful_block = min(block_gain, incoming_before)
            block_reward = 0.01 * useful_block
        else:
            block_reward = 0.0

        # threat reduction reward
        threat_before = self._estimate_total_enemy_threat(prev_state)
        threat_after = self._estimate_total_enemy_threat(next_state)
        threat_reward = 0.03 * (threat_before - threat_after)

        # debuff reward on enemies
        enemy_vuln_before = self._sum_enemy_power(prev_state, "Vulnerable")
        enemy_vuln_after = self._sum_enemy_power(next_state, "Vulnerable")
        enemy_weak_before = self._sum_enemy_power(prev_state, "Weak")
        enemy_weak_after = self._sum_enemy_power(next_state, "Weak")

        gained_enemy_vuln = max(0.0, enemy_vuln_after - enemy_vuln_before)
        gained_enemy_weak = max(0.0, enemy_weak_after - enemy_weak_before)

        debuff_reward = 0.04 * gained_enemy_vuln + 0.03 * gained_enemy_weak

        # player useful buff reward
        strength_before = self._get_player_power(prev_state, "Strength")
        strength_after = self._get_player_power(next_state, "Strength")

        dex_before = self._get_player_power(prev_state, "Dexterity")
        dex_after = self._get_player_power(next_state, "Dexterity")

        metal_before = self._get_player_power(prev_state, "Metallicize")
        metal_after = self._get_player_power(next_state, "Metallicize")

        plated_before = self._get_player_power(prev_state, "Plated Armor")
        plated_after = self._get_player_power(next_state, "Plated Armor")

        artifact_before = self._get_player_power(prev_state, "Artifact")
        artifact_after = self._get_player_power(next_state, "Artifact")

        intang_before = self._get_player_power(prev_state, "Intangible")
        intang_after = self._get_player_power(next_state, "Intangible")

        gained_strength = max(0.0, strength_after - strength_before)
        gained_dex = max(0.0, dex_after - dex_before)
        gained_metal = max(0.0, metal_after - metal_before)
        gained_plated = max(0.0, plated_after - plated_before)
        gained_artifact = max(0.0, artifact_after - artifact_before)
        gained_intang = max(0.0, intang_after - intang_before)

        buff_reward = (
                0.015 * gained_strength
                + 0.01 * gained_dex
                + 0.008 * gained_metal
                + 0.008 * gained_plated
                + 0.01 * gained_artifact
                + 0.04 * gained_intang
        )

        total_reward = (
                damage_dealt_reward
                + damage_taken_reward
                + kill_bonus
                + win_bonus
                + lose_penalty
                + illegal_action_penalty
                + step_penalty
                + status_curse_hand_reward
                + self_hp_loss_from_bad_cards_penalty
                + potion_use_penalty
                + survival_reward
                + block_reward
                + debuff_reward
                + buff_reward
                + threat_reward
        )

        return CombatRewardOutput(
            total_reward=total_reward,
            damage_dealt_reward=damage_dealt_reward,
            damage_taken_reward=damage_taken_reward,
            kill_bonus=kill_bonus,
            win_bonus=win_bonus,
            lose_penalty=lose_penalty,
            illegal_action_penalty=illegal_action_penalty,
            step_penalty=step_penalty,
            status_curse_hand_reward=status_curse_hand_reward,
            self_hp_loss_from_bad_cards_penalty=self_hp_loss_from_bad_cards_penalty,
            potion_use_penalty=potion_use_penalty,
            survival_reward=survival_reward,
            block_reward=block_reward,
            debuff_reward=debuff_reward,
            buff_reward=buff_reward,
            threat_reward=threat_reward,
            player_hp_before=player_hp_before,
            player_hp_after=player_hp_after,
            enemy_total_hp_before=enemy_total_hp_before,
            enemy_total_hp_after=enemy_total_hp_after,
            enemies_killed=enemies_killed,
            combat_won=combat_won,
            combat_lost=combat_lost,
        )

    # =====================================================
    # Reward terms
    # =====================================================

    def _compute_step_penalty(
        self,
        action_info: Dict[str, Any],
        combat_won: bool,
        combat_lost: bool,
    ) -> float:
        if combat_won or combat_lost:
            return 0.0

        action_type = action_info.get("action_type", action_info.get("command_type", ""))

        # Petit coût global pour éviter de traîner
        penalty = self.r_cfg.end_turn_small_penalty

        # Tu peux rendre "wait" un peu plus puni si tu veux
        if action_type == "wait":
            penalty += self.r_cfg.end_turn_small_penalty

        return penalty

    # =====================================================
    # State extraction
    # =====================================================

    def _get_player_hp(self, state: Dict[str, Any]) -> float:
        player = state.get("player", state)
        return self._safe_float(
            player.get("current_hp", player.get("currentHealth", player.get("hp", 0.0)))
        )

    def _get_player_max_hp(self, state: Dict[str, Any]) -> float:
        player = state.get("player", state)
        return self._safe_float(
            player.get("max_hp", player.get("maxHealth", 1.0))
        )

    def _get_monsters(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        monsters = state.get("monsters", state.get("enemies", []))
        if isinstance(monsters, list):
            return monsters
        return []

    def _get_enemy_effective_hp(self, monster: Dict[str, Any]) -> float:
        hp = self._safe_float(
            monster.get("current_hp", monster.get("currentHealth", monster.get("hp", 0.0)))
        )
        block = self._safe_float(
            monster.get("block", monster.get("currentBlock", 0.0))
        )

        if self._is_enemy_dead(monster):
            return 0.0

        return max(0.0, hp) + max(0.0, block)

    def _get_total_enemy_effective_hp(self, state: Dict[str, Any]) -> float:
        return sum(self._get_enemy_effective_hp(m) for m in self._get_monsters(state))

    def _count_newly_killed_enemies(
        self,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any],
    ) -> int:
        prev_monsters = self._get_monsters(prev_state)
        next_monsters = self._get_monsters(next_state)

        count = 0
        n = min(len(prev_monsters), len(next_monsters))

        for i in range(n):
            was_dead = self._is_enemy_dead(prev_monsters[i])
            now_dead = self._is_enemy_dead(next_monsters[i])

            if (not was_dead) and now_dead:
                count += 1

        # Si la liste change de taille bizarrement, on ignore pour rester safe
        return count

    def _is_enemy_dead(self, monster: Dict[str, Any]) -> bool:
        if monster.get("isDead", False):
            return True
        if monster.get("dead", False):
            return True
        if monster.get("escaped", False):
            return True

        hp = self._safe_float(
            monster.get("current_hp", monster.get("currentHealth", monster.get("hp", 0.0)))
        )
        return hp <= 0.0

    # =====================================================
    # Terminal checks
    # =====================================================

    def _is_combat_won(self, state: Dict[str, Any]) -> bool:
        if state.get("combat_over", False):
            monsters = self._get_monsters(state)
            if len(monsters) == 0:
                return True
            if all(self._is_enemy_dead(m) for m in monsters):
                return True

        monsters = self._get_monsters(state)
        if len(monsters) > 0 and all(self._is_enemy_dead(m) for m in monsters):
            return True

        return False

    def _is_combat_lost(self, state: Dict[str, Any]) -> bool:
        player_hp = self._get_player_hp(state)
        if player_hp <= 0.0:
            return True

        if state.get("game_over", False):
            return True

        if state.get("is_terminal", False) and not self._is_combat_won(state):
            # prudence
            return player_hp <= 0.0

        return False

    def _count_bad_cards_in_hand(self, state: Dict[str, Any]) -> int:
        bad_ids = {
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

        hand = state.get("hand", [])
        if not isinstance(hand, list):
            return 0

        count = 0
        for card in hand:
            if isinstance(card, dict) and card.get("id") in bad_ids:
                count += 1
        return count

    def _get_player_block(self, state: Dict[str, Any]) -> float:
        player = state.get("player", state)
        return self._safe_float(
            player.get("block", player.get("currentBlock", 0.0))
        )

    def _get_powers(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        powers = entity.get("powers", [])
        if isinstance(powers, list):
            return powers
        return []

    def _get_power_amount(self, entity: Dict[str, Any], power_name: str) -> float:
        for p in self._get_powers(entity):
            if p.get("id") == power_name:
                return self._safe_float(p.get("amount", 0.0))
        return 0.0

    def _get_player_power(self, state: Dict[str, Any], power_name: str) -> float:
        player = state.get("player", state)
        return self._get_power_amount(player, power_name)

    def _sum_enemy_power(self, state: Dict[str, Any], power_name: str) -> float:
        total = 0.0
        for m in self._get_monsters(state):
            if self._is_enemy_dead(m):
                continue
            total += self._get_power_amount(m, power_name)
        return total

    def _estimate_incoming_damage(self, state: Dict[str, Any]) -> float:
        total = 0.0
        for m in self._get_monsters(state):
            if self._is_enemy_dead(m):
                continue

            intent = str(m.get("intent", ""))
            dmg = self._safe_float(m.get("intent_base_damage", 0.0))

            if intent in {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}:
                total += max(0.0, dmg)

        return total

    def _estimate_enemy_threat(self, monster: Dict[str, Any]) -> float:
        if self._is_enemy_dead(monster):
            return 0.0

        intent = str(monster.get("intent", ""))
        dmg = self._safe_float(monster.get("intent_base_damage", 0.0))
        hp = self._safe_float(monster.get("current_hp", 0.0))
        block = self._safe_float(monster.get("block", 0.0))

        strength = self._get_power_amount(monster, "Strength")
        ritual = self._get_power_amount(monster, "Ritual")
        metallicize = self._get_power_amount(monster, "Metallicize")
        plated = self._get_power_amount(monster, "Plated Armor")

        threat = 0.0

        # danger immédiat
        if intent in {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}:
            threat += max(0.0, dmg)

        # scaling / danger futur léger
        threat += 0.8 * strength
        threat += 2.0 * ritual
        threat += 0.5 * metallicize
        threat += 0.5 * plated

        # bonus léger si cible presque finissable
        effective_hp = hp + block
        if effective_hp <= 15:
            threat += 2.0
        if effective_hp <= 6:
            threat += 3.0

        return threat

    def _estimate_total_enemy_threat(self, state: Dict[str, Any]) -> float:
        total = 0.0
        for m in self._get_monsters(state):
            total += self._estimate_enemy_threat(m)
        return total

    def _count_non_empty_potions(self, state: Dict[str, Any]) -> int:
        potions = state.get("potions", state.get("player", {}).get("potions", []))
        if not isinstance(potions, list):
            return 0

        count = 0
        for p in potions:
            if not isinstance(p, dict):
                continue
            if not p.get("empty", False):
                count += 1
        return count

    # =====================================================
    # Utils
    # =====================================================

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default


# =========================================================
# Convenience function
# =========================================================

def compute_combat_reward(
    prev_state: Dict[str, Any],
    next_state: Dict[str, Any],
    action_info: Optional[Dict[str, Any]] = None,
    cfg: Optional[Config] = None,
) -> CombatRewardOutput:
    calculator = CombatRewardCalculator(cfg)
    return calculator.compute(prev_state, next_state, action_info)


# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":
    cfg = get_default_config()
    calc = CombatRewardCalculator(cfg)

    prev_state = {
        "player": {
            "current_hp": 70,
            "max_hp": 80,
            "block": 0,
        },
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 20,
                "max_hp": 40,
                "block": 0,
                "isDead": False,
            },
            {
                "name": "Cultist",
                "current_hp": 45,
                "max_hp": 48,
                "block": 0,
                "isDead": False,
            },
        ],
    }

    next_state = {
        "player": {
            "current_hp": 66,
            "max_hp": 80,
            "block": 0,
        },
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 0,
                "max_hp": 40,
                "block": 0,
                "isDead": True,
            },
            {
                "name": "Cultist",
                "current_hp": 39,
                "max_hp": 48,
                "block": 0,
                "isDead": False,
            },
        ],
    }

    out = calc.compute(
        prev_state=prev_state,
        next_state=next_state,
        action_info={"command_type": "play_card", "illegal_action": False},
    )

    print("=== Combat reward debug ===")
    for k, v in out.to_dict().items():
        print(f"{k}: {v}")
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

    # New richer shaping terms
    setup_reward: float = 0.0
    sequencing_reward: float = 0.0
    energy_reward: float = 0.0
    potion_timing_reward: float = 0.0
    lethal_reward: float = 0.0

    player_hp_before: float = 0.0
    player_hp_after: float = 0.0
    enemy_total_hp_before: float = 0.0
    enemy_total_hp_after: float = 0.0
    enemies_killed: int = 0
    combat_won: bool = False
    combat_lost: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "damage_dealt_reward": self.damage_dealt_reward,
            "damage_taken_reward": self.damage_taken_reward,
            "kill_bonus": self.kill_bonus,
            "win_bonus": self.win_bonus,
            "lose_penalty": self.lose_penalty,
            "illegal_action_penalty": self.illegal_action_penalty,
            "step_penalty": self.step_penalty,
            "status_curse_hand_reward": self.status_curse_hand_reward,
            "self_hp_loss_from_bad_cards_penalty": self.self_hp_loss_from_bad_cards_penalty,
            "potion_use_penalty": self.potion_use_penalty,
            "survival_reward": self.survival_reward,
            "block_reward": self.block_reward,
            "debuff_reward": self.debuff_reward,
            "buff_reward": self.buff_reward,
            "threat_reward": self.threat_reward,
            "setup_reward": self.setup_reward,
            "sequencing_reward": self.sequencing_reward,
            "energy_reward": self.energy_reward,
            "potion_timing_reward": self.potion_timing_reward,
            "lethal_reward": self.lethal_reward,
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
    Richer reward shaping for combat.

    Goals:
    - reward immediate combat progress
    - reward useful setup that improves the next actions
    - reward better sequencing
    - discourage bad end-turn / wasted energy
    - encourage intelligent potion timing
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

        # -------------------------------------------------
        # hand pollution / self-damage
        # -------------------------------------------------
        bad_hand_before = self._count_bad_cards_in_hand(prev_state)
        bad_hand_after = self._count_bad_cards_in_hand(next_state)
        status_curse_hand_reward = (
            float(bad_hand_before - bad_hand_after)
            * self.r_cfg.status_curse_hand_reduce_scale
        )

        hp_loss_breakdown = next_state.get("hp_loss_breakdown", {})
        burn_loss = float(hp_loss_breakdown.get("burn", 0.0))
        pain_loss = float(hp_loss_breakdown.get("pain", 0.0))
        decay_loss = float(hp_loss_breakdown.get("decay", 0.0))

        self_hp_loss_from_bad_cards_penalty = (
            self.r_cfg.self_bad_hp_loss_scale * (burn_loss + pain_loss + decay_loss)
        )

        # -------------------------------------------------
        # shared quantities
        # -------------------------------------------------
        hp_ratio_before = player_hp_before / player_max_hp_before
        hp_ratio_after = player_hp_after / player_max_hp_after

        incoming_before = self._estimate_incoming_damage(prev_state)
        player_block_before = self._get_player_block(prev_state)
        player_block_after = self._get_player_block(next_state)
        energy_before = self._get_player_energy(prev_state)
        energy_after = self._get_player_energy(next_state)

        action_type = self._get_action_type(action_info)
        played_card_name = self._extract_played_card_name(action_info)

        # -------------------------------------------------
        # survival
        # -------------------------------------------------
        survival_reward = self.r_cfg.survival_hp_ratio_scale * (hp_ratio_after - hp_ratio_before)

        # -------------------------------------------------
        # block quality
        # -------------------------------------------------
        block_gain = max(0.0, player_block_after - player_block_before)
        useful_block = min(block_gain, incoming_before)
        overblock = max(0.0, player_block_after - max(0.0, incoming_before))

        block_reward = (
            self.r_cfg.useful_block_scale * useful_block
            + self.r_cfg.overblock_penalty_scale * min(overblock, 25.0)
        )

        # -------------------------------------------------
        # threat reduction
        # -------------------------------------------------
        threat_before = self._estimate_total_enemy_threat(prev_state)
        threat_after = self._estimate_total_enemy_threat(next_state)
        threat_reward = self.r_cfg.threat_reduction_scale * (threat_before - threat_after)

        # -------------------------------------------------
        # enemy debuffs
        # -------------------------------------------------
        enemy_vuln_before = self._sum_enemy_power(prev_state, "Vulnerable")
        enemy_vuln_after = self._sum_enemy_power(next_state, "Vulnerable")
        enemy_weak_before = self._sum_enemy_power(prev_state, "Weak")
        enemy_weak_after = self._sum_enemy_power(next_state, "Weak")

        gained_enemy_vuln = max(0.0, enemy_vuln_after - enemy_vuln_before)
        gained_enemy_weak = max(0.0, enemy_weak_after - enemy_weak_before)

        debuff_reward = (
            self.r_cfg.enemy_vulnerable_scale * gained_enemy_vuln
            + self.r_cfg.enemy_weak_scale * gained_enemy_weak
        )

        # -------------------------------------------------
        # player buffs
        # -------------------------------------------------
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
            self.r_cfg.strength_gain_scale * gained_strength
            + self.r_cfg.dex_gain_scale * gained_dex
            + self.r_cfg.metallicize_gain_scale * gained_metal
            + self.r_cfg.plated_gain_scale * gained_plated
            + self.r_cfg.artifact_gain_scale * gained_artifact
            + self.r_cfg.intangible_gain_scale * gained_intang
        )

        # -------------------------------------------------
        # setup reward
        # -------------------------------------------------
        setup_reward = self._compute_setup_reward(
            prev_state=prev_state,
            next_state=next_state,
            played_card_name=played_card_name,
        )

        # -------------------------------------------------
        # sequencing reward
        # -------------------------------------------------
        sequencing_reward = self._compute_sequencing_reward(
            prev_state=prev_state,
            next_state=next_state,
            played_card_name=played_card_name,
        )

        # -------------------------------------------------
        # energy quality reward
        # -------------------------------------------------
        energy_reward = self._compute_energy_reward(
            prev_state=prev_state,
            next_state=next_state,
            action_type=action_type,
            energy_before=energy_before,
            energy_after=energy_after,
        )

        # -------------------------------------------------
        # potion timing
        # -------------------------------------------------
        potion_use_penalty, potion_timing_reward = self._compute_potion_terms(
            action_type=action_type,
            incoming_before=incoming_before,
            combat_won=combat_won,
        )

        # -------------------------------------------------
        # lethal / focus reward
        # -------------------------------------------------
        lethal_reward = self._compute_lethal_reward(
            prev_state=prev_state,
            next_state=next_state,
            damage_dealt=damage_dealt,
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
            + setup_reward
            + sequencing_reward
            + energy_reward
            + potion_timing_reward
            + lethal_reward
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
            setup_reward=setup_reward,
            sequencing_reward=sequencing_reward,
            energy_reward=energy_reward,
            potion_timing_reward=potion_timing_reward,
            lethal_reward=lethal_reward,
            player_hp_before=player_hp_before,
            player_hp_after=player_hp_after,
            enemy_total_hp_before=enemy_total_hp_before,
            enemy_total_hp_after=enemy_total_hp_after,
            enemies_killed=enemies_killed,
            combat_won=combat_won,
            combat_lost=combat_lost,
        )

    # =====================================================
    # High-level reward terms
    # =====================================================

    def _compute_step_penalty(
            self,
            action_info: Dict[str, Any],
            combat_won: bool,
            combat_lost: bool,
    ) -> float:
        if combat_won or combat_lost:
            return 0.0

        action_type = self._get_action_type(action_info)

        # Cas neutre / action non renseignée dans certains tests
        if action_type == "":
            return 0.5 * self.r_cfg.end_turn_small_penalty

        penalty = self.r_cfg.end_turn_small_penalty

        if action_type == "wait":
            penalty += self.r_cfg.end_turn_small_penalty

        return penalty

    def _compute_setup_reward(
        self,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any],
        played_card_name: str,
    ) -> float:
        next_playable_attacks = self._count_playable_attacks(next_state)
        next_playable_skills = self._count_playable_skills(next_state)
        next_alive_enemies = self._count_alive_enemies(next_state)
        next_best_attack = self._estimate_best_attack_damage(next_state)
        next_bad_cards = self._count_bad_cards_in_hand(next_state)

        reward = 0.0

        gained_rage = max(
            0.0,
            self._get_player_power(next_state, "Rage") - self._get_player_power(prev_state, "Rage")
        )
        gained_double_tap = max(
            0.0,
            self._get_meta(next_state, "double_tap_charges") - self._get_meta(prev_state, "double_tap_charges")
        )
        gained_corruption = max(
            0.0,
            self._has_player_power(next_state, "Corruption") - self._has_player_power(prev_state, "Corruption")
        )
        gained_barricade = max(
            0.0,
            self._has_player_power(next_state, "Barricade") - self._has_player_power(prev_state, "Barricade")
        )
        gained_fnp = max(
            0.0,
            self._get_player_power(next_state, "Feel No Pain") - self._get_player_power(prev_state, "Feel No Pain")
        )
        gained_dark_embrace = max(
            0.0,
            self._get_player_power(next_state, "Dark Embrace") - self._get_player_power(prev_state, "Dark Embrace")
        )
        gained_demon_form = max(
            0.0,
            self._get_player_power(next_state, "Demon Form") - self._get_player_power(prev_state, "Demon Form")
        )
        gained_evolve = max(
            0.0,
            self._get_player_power(next_state, "Evolve") - self._get_player_power(prev_state, "Evolve")
        )
        gained_combust = max(
            0.0,
            self._get_player_power(next_state, "Combust") - self._get_player_power(prev_state, "Combust")
        )
        gained_juggernaut = max(
            0.0,
            self._get_player_power(next_state, "Juggernaut") - self._get_player_power(prev_state, "Juggernaut")
        )
        gained_rupture = max(
            0.0,
            self._get_player_power(next_state, "Rupture") - self._get_player_power(prev_state, "Rupture")
        )

        reward += self.r_cfg.rage_setup_scale * gained_rage * min(float(next_playable_attacks), 3.0)
        reward += self.r_cfg.double_tap_setup_scale * gained_double_tap * min(next_best_attack / 12.0, 2.5)
        reward += self.r_cfg.corruption_setup_scale * gained_corruption * min(float(next_playable_skills), 4.0)
        reward += self.r_cfg.barricade_setup_scale * gained_barricade * min(self._get_player_block(next_state) / 12.0, 2.0)
        reward += self.r_cfg.feel_no_pain_setup_scale * gained_fnp * min(float(next_bad_cards + next_playable_skills), 4.0)
        reward += self.r_cfg.dark_embrace_setup_scale * gained_dark_embrace * min(float(next_bad_cards + next_playable_skills), 4.0)
        reward += self.r_cfg.demon_form_setup_scale * gained_demon_form * min(float(next_alive_enemies), 3.0)
        reward += self.r_cfg.evolve_setup_scale * gained_evolve * min(float(next_bad_cards), 3.0)
        reward += self.r_cfg.combust_setup_scale * gained_combust * min(float(next_alive_enemies), 3.0)
        reward += self.r_cfg.juggernaut_setup_scale * gained_juggernaut * min(self._get_player_block(next_state) / 10.0, 3.0)
        reward += self.r_cfg.rupture_setup_scale * gained_rupture * min(self._estimate_self_hp_loss_risk(next_state), 3.0)

        # card-specific sequencing-friendly setup bonuses
        card = played_card_name.lower()
        if card == "inflame" and next_playable_attacks > 0:
            reward += self.r_cfg.inflame_setup_scale * min(float(next_playable_attacks), 2.0)
        if card == "spot weakness" and next_playable_attacks > 0:
            reward += self.r_cfg.spot_weakness_setup_scale * min(float(next_playable_attacks), 2.0)
        if card == "rage" and next_playable_attacks > 0:
            reward += self.r_cfg.rage_setup_scale * min(float(next_playable_attacks), 2.0)
        if card == "double tap" and next_best_attack >= 8.0:
            reward += self.r_cfg.double_tap_setup_scale * min(next_best_attack / 12.0, 1.5)

        return min(reward, 0.20)

    def _compute_sequencing_reward(
            self,
            prev_state: Dict[str, Any],
            next_state: Dict[str, Any],
            played_card_name: str,
    ) -> float:
        reward = 0.0

        prev_double_tap = self._get_meta(prev_state, "double_tap_charges")
        next_double_tap = self._get_meta(next_state, "double_tap_charges")

        prev_rage = self._get_player_power(prev_state, "Rage")

        attacks_before = self._get_meta(prev_state, "attacks_played_this_turn")
        attacks_after = self._get_meta(next_state, "attacks_played_this_turn")

        # bonus réduit si Double Tap a vraiment servi
        if prev_double_tap > next_double_tap and attacks_after > attacks_before:
            reward += 0.03

        # bonus réduit si Rage existait déjà et qu'on a effectivement attaqué
        if prev_rage > 0 and attacks_after > attacks_before:
            reward += 0.015

        # petit bonus si on a setup et qu'il reste du vrai payoff
        next_best_attack = self._estimate_best_attack_damage(next_state)
        next_playable_attacks = self._count_playable_attacks(next_state)

        card = played_card_name.lower()
        if card in {"inflame", "spot weakness", "rage", "double tap"}:
            reward += 0.003 * min(float(next_playable_attacks), 3.0)
            reward += 0.003 * min(next_best_attack / 10.0, 2.0)

        return reward

    def _compute_energy_reward(
            self,
            prev_state: Dict[str, Any],
            next_state: Dict[str, Any],
            action_type: str,
            energy_before: float,
            energy_after: float,
    ) -> float:
        reward = 0.0

        energy_spent = max(0.0, energy_before - energy_after)

        if energy_spent > 0.0:
            reward += self.r_cfg.good_energy_use_reward_scale * min(energy_spent, 3.0)

        if action_type == "end_turn":
            playable_before = self._count_playable_cards(prev_state)
            incoming_before = self._estimate_incoming_damage(prev_state)
            player_block_before = self._get_player_block(prev_state)

            if playable_before > 0 and energy_before > 0:
                reward += self.r_cfg.wasted_energy_penalty_scale * min(energy_before, 3.0)

            # punir un peu plus les fins de tour "ouvertes" face à du damage entrant
            if incoming_before > player_block_before + 1.0:
                gap = incoming_before - player_block_before
                reward += self.r_cfg.wasted_energy_penalty_scale * 0.5 * min(gap / 5.0, 3.0)

        return reward

    def _compute_potion_terms(
            self,
            action_type: str,
            incoming_before: float,
            combat_won: bool,
    ) -> tuple[float, float]:
        if action_type != "use_potion":
            return 0.0, 0.0

        potion_use_penalty = 0.0
        potion_timing_reward = 0.0

        if incoming_before <= 6.0:
            potion_use_penalty += self.r_cfg.potion_low_threat_penalty
        elif incoming_before <= 14.0:
            potion_use_penalty += self.r_cfg.potion_medium_threat_penalty
        else:
            potion_use_penalty += self.r_cfg.potion_high_threat_penalty

        if incoming_before >= 16.0:
            potion_timing_reward += self.r_cfg.potion_emergency_bonus

        if combat_won:
            potion_timing_reward += self.r_cfg.potion_lethal_bonus

        return potion_use_penalty, potion_timing_reward

    def _compute_lethal_reward(
        self,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any],
        damage_dealt: float,
    ) -> float:
        reward = 0.0

        prev_best_target_ehp = self._min_alive_enemy_effective_hp(prev_state)
        next_best_target_ehp = self._min_alive_enemy_effective_hp(next_state)

        if prev_best_target_ehp is not None and next_best_target_ehp is not None:
            if prev_best_target_ehp > 0 and next_best_target_ehp <= 0:
                reward += self.r_cfg.lethal_reward_scale
            elif prev_best_target_ehp > 0 and next_best_target_ehp <= 8.0:
                reward += self.r_cfg.near_lethal_reward_scale

        if damage_dealt >= 12.0:
            reward += 0.5 * self.r_cfg.near_lethal_reward_scale

        return reward

    # =====================================================
    # State extraction
    # =====================================================

    def _get_action_type(self, action_info: Dict[str, Any]) -> str:
        return str(action_info.get("action_type", action_info.get("command_type", ""))).lower()

    def _extract_played_card_name(self, action_info: Dict[str, Any]) -> str:
        for key in ["card_name", "card_id", "played_card_name"]:
            value = action_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        played_card = action_info.get("played_card")
        if isinstance(played_card, dict):
            for key in ["id", "card_id", "name"]:
                value = played_card.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    def _get_player_hp(self, state: Dict[str, Any]) -> float:
        player = state.get("player", state)
        return self._safe_float(
            player.get("current_hp", player.get("currentHealth", player.get("hp", 0.0)))
        )

    def _get_player_max_hp(self, state: Dict[str, Any]) -> float:
        player = state.get("player", state)
        return self._safe_float(player.get("max_hp", player.get("maxHealth", 1.0)))

    def _get_player_energy(self, state: Dict[str, Any]) -> float:
        player = state.get("player", {})
        return self._safe_float(
            player.get("energy", state.get("energy", state.get("player_energy", 0.0)))
        )

    def _get_player_block(self, state: Dict[str, Any]) -> float:
        player = state.get("player", state)
        return self._safe_float(player.get("block", player.get("currentBlock", 0.0)))

    def _get_monsters(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        monsters = state.get("monsters", state.get("enemies", []))
        if isinstance(monsters, list):
            return monsters
        return []

    def _get_hand(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        hand = state.get("hand", state.get("hand_cards", state.get("handCards", [])))
        if isinstance(hand, list):
            return hand
        return []

    def _get_powers(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        powers = entity.get("powers", [])
        if isinstance(powers, list):
            return powers
        return []

    def _get_power_amount(self, entity: Dict[str, Any], power_name: str) -> float:
        power_name_low = power_name.lower()
        total = 0.0

        for p in self._get_powers(entity):
            pid = str(p.get("id", p.get("name", ""))).lower()
            if pid == power_name_low or power_name_low in pid:
                total += self._safe_float(p.get("amount", p.get("stack", 0.0)))
        return total

    def _get_player_power(self, state: Dict[str, Any], power_name: str) -> float:
        player = state.get("player", state)
        return self._get_power_amount(player, power_name)

    def _has_player_power(self, state: Dict[str, Any], power_name: str) -> float:
        return 1.0 if self._get_player_power(state, power_name) > 0.0 else 0.0

    def _sum_enemy_power(self, state: Dict[str, Any], power_name: str) -> float:
        total = 0.0
        for m in self._get_monsters(state):
            if self._is_enemy_dead(m):
                continue
            total += self._get_power_amount(m, power_name)
        return total

    def _get_meta(self, state: Dict[str, Any], key: str, default: float = 0.0) -> float:
        meta = state.get("combat_meta", {})
        if not isinstance(meta, dict):
            return default
        return self._safe_float(meta.get(key, default), default)

    # =====================================================
    # Enemy / terminal utilities
    # =====================================================

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

    def _get_enemy_effective_hp(self, monster: Dict[str, Any]) -> float:
        hp = self._safe_float(
            monster.get("current_hp", monster.get("currentHealth", monster.get("hp", 0.0)))
        )
        block = self._safe_float(monster.get("block", monster.get("currentBlock", 0.0)))

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

        return count

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
        if self._get_player_hp(state) <= 0.0:
            return True
        if state.get("game_over", False):
            return True
        if state.get("is_terminal", False) and not self._is_combat_won(state):
            return self._get_player_hp(state) <= 0.0
        return False

    def _count_alive_enemies(self, state: Dict[str, Any]) -> int:
        return sum(1 for m in self._get_monsters(state) if not self._is_enemy_dead(m))

    def _min_alive_enemy_effective_hp(self, state: Dict[str, Any]) -> Optional[float]:
        values = [
            self._get_enemy_effective_hp(m)
            for m in self._get_monsters(state)
            if not self._is_enemy_dead(m)
        ]
        if not values:
            return None
        return min(values)

    # =====================================================
    # Hand / cards / attacks
    # =====================================================

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

        count = 0
        for card in self._get_hand(state):
            if isinstance(card, dict) and str(card.get("id", "")) in bad_ids:
                count += 1
        return count

    def _get_card_cost(self, card: Dict[str, Any]) -> float:
        return self._safe_float(
            card.get("cost_for_turn", card.get("costForTurn", card.get("cost", 99.0))),
            99.0,
        )

    def _is_card_playable(self, state: Dict[str, Any], card: Dict[str, Any]) -> bool:
        explicit = card.get("is_playable", card.get("playable", None))
        if explicit is not None:
            return bool(explicit)

        cost = self._get_card_cost(card)
        if cost == -1:
            return True

        energy = self._get_player_energy(state)
        return cost <= energy

    def _card_type(self, card: Dict[str, Any]) -> str:
        return str(card.get("type", "")).upper()

    def _count_playable_cards(self, state: Dict[str, Any]) -> int:
        count = 0
        for card in self._get_hand(state):
            if not isinstance(card, dict):
                continue
            if self._is_card_playable(state, card):
                count += 1
        return count

    def _count_playable_attacks(self, state: Dict[str, Any]) -> int:
        count = 0
        for card in self._get_hand(state):
            if not isinstance(card, dict):
                continue
            if self._card_type(card) == "ATTACK" and self._is_card_playable(state, card):
                count += 1
        return count

    def _count_playable_skills(self, state: Dict[str, Any]) -> int:
        count = 0
        for card in self._get_hand(state):
            if not isinstance(card, dict):
                continue
            if self._card_type(card) == "SKILL" and self._is_card_playable(state, card):
                count += 1
        return count

    def _estimate_best_attack_damage(self, state: Dict[str, Any]) -> float:
        strength = self._get_player_power(state, "Strength")
        best = 0.0

        for card in self._get_hand(state):
            if not isinstance(card, dict):
                continue
            if self._card_type(card) != "ATTACK":
                continue
            if not self._is_card_playable(state, card):
                continue

            base = self._safe_float(card.get("damage", 0.0))
            hits = max(1.0, self._safe_float(card.get("hits", 1.0)))

            if card.get("damage_from_block", False):
                est = self._get_player_block(state)
            elif card.get("strength_mult") is not None:
                est = base + strength * self._safe_float(card.get("strength_mult", 1.0))
            else:
                est = (base + strength) * hits

            best = max(best, est)

        return best

    def _estimate_self_hp_loss_risk(self, state: Dict[str, Any]) -> float:
        hand = self._get_hand(state)
        risk = 0.0

        if self._get_player_power(state, "Combust") > 0:
            risk += 1.0
        if self._get_player_power(state, "Brutality") > 0:
            risk += 1.0
        for card in hand:
            cid = str(card.get("id", ""))
            if cid in {"Hemokinesis", "Offering", "Bloodletting"}:
                risk += 1.0
        return risk

    # =====================================================
    # Threat / intents / potions
    # =====================================================

    def _extract_enemy_intent_hits(self, monster: Dict[str, Any]) -> float:
        for key in ["intent_hits", "intentHits", "move_hits", "hits", "multi"]:
            if key in monster:
                hits = self._safe_float(monster.get(key, 1.0), 1.0)
                return max(1.0, hits)
        return 1.0

    def _estimate_incoming_damage(self, state: Dict[str, Any]) -> float:
        total = 0.0
        for m in self._get_monsters(state):
            if self._is_enemy_dead(m):
                continue

            intent = str(m.get("intent", "")).upper()
            dmg = self._safe_float(
                m.get("intent_base_damage", m.get("intentBaseDmg", m.get("move_base_damage", 0.0)))
            )
            hits = self._extract_enemy_intent_hits(m)

            if intent in {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}:
                total += max(0.0, dmg) * hits

        return total

    def _estimate_enemy_threat(self, monster: Dict[str, Any]) -> float:
        if self._is_enemy_dead(monster):
            return 0.0

        intent = str(monster.get("intent", "")).upper()
        dmg = self._safe_float(
            monster.get("intent_base_damage", monster.get("intentBaseDmg", monster.get("move_base_damage", 0.0)))
        )
        hits = self._extract_enemy_intent_hits(monster)
        hp = self._safe_float(
            monster.get("current_hp", monster.get("currentHealth", monster.get("hp", 0.0)))
        )
        block = self._safe_float(monster.get("block", monster.get("currentBlock", 0.0)))

        strength = self._get_power_amount(monster, "Strength")
        ritual = self._get_power_amount(monster, "Ritual")
        metallicize = self._get_power_amount(monster, "Metallicize")
        plated = self._get_power_amount(monster, "Plated Armor")
        artifact = self._get_power_amount(monster, "Artifact")

        threat = 0.0

        if intent in {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}:
            threat += max(0.0, dmg) * hits

        threat += 0.8 * strength
        threat += 2.0 * ritual
        threat += 0.5 * metallicize
        threat += 0.5 * plated
        threat += 0.4 * artifact

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

    def _count_near_lethal_enemies(self, state: Dict[str, Any], threshold: float) -> int:
        count = 0
        for m in self._get_monsters(state):
            if self._is_enemy_dead(m):
                continue
            if self._get_enemy_effective_hp(m) <= threshold:
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
        "energy": 3,
        "player": {
            "current_hp": 70,
            "max_hp": 80,
            "block": 0,
            "powers": [{"id": "Strength", "amount": 0}],
        },
        "hand": [
            {"id": "Inflame", "cost": 1, "type": "POWER"},
            {"id": "Strike_R", "cost": 1, "type": "ATTACK", "damage": 6},
            {"id": "Bash", "cost": 2, "type": "ATTACK", "damage": 8},
        ],
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 20,
                "max_hp": 40,
                "block": 0,
                "intent": "ATTACK",
                "intent_base_damage": 11,
                "isDead": False,
                "powers": [],
            },
            {
                "name": "Cultist",
                "current_hp": 45,
                "max_hp": 48,
                "block": 0,
                "intent": "BUFF",
                "intent_base_damage": 0,
                "isDead": False,
                "powers": [{"id": "Ritual", "amount": 3}],
            },
        ],
        "potions": [
            {"name": "Fire Potion", "empty": False},
            {"name": "Dexterity Potion", "empty": False},
        ],
        "combat_meta": {
            "cards_played_this_turn": 0,
            "attacks_played_this_turn": 0,
            "double_tap_charges": 0,
        },
    }

    next_state = {
        "energy": 2,
        "player": {
            "current_hp": 70,
            "max_hp": 80,
            "block": 0,
            "powers": [{"id": "Strength", "amount": 2}],
        },
        "hand": [
            {"id": "Strike_R", "cost": 1, "type": "ATTACK", "damage": 6},
            {"id": "Bash", "cost": 2, "type": "ATTACK", "damage": 8},
        ],
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 20,
                "max_hp": 40,
                "block": 0,
                "intent": "ATTACK",
                "intent_base_damage": 11,
                "isDead": False,
                "powers": [],
            },
            {
                "name": "Cultist",
                "current_hp": 45,
                "max_hp": 48,
                "block": 0,
                "intent": "BUFF",
                "intent_base_damage": 0,
                "isDead": False,
                "powers": [{"id": "Ritual", "amount": 3}],
            },
        ],
        "potions": [
            {"name": "Fire Potion", "empty": False},
            {"name": "Dexterity Potion", "empty": False},
        ],
        "combat_meta": {
            "cards_played_this_turn": 1,
            "attacks_played_this_turn": 0,
            "double_tap_charges": 0,
        },
    }

    out = calc.compute(
        prev_state=prev_state,
        next_state=next_state,
        action_info={
            "command_type": "play_card",
            "card_name": "Inflame",
            "illegal_action": False,
        },
    )

    print("=== Combat reward debug ===")
    for k, v in out.to_dict().items():
        print(f"{k}: {v}")
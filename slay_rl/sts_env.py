
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from slay_rl.config import Config, get_default_config
from slay_rl.agents.combat_agent import CombatCommand, encode_command_to_action_index
from slay_rl.rewards.combat_reward import CombatRewardCalculator, CombatRewardOutput


@dataclass
class StepInfo:
    action_index: int
    command: Dict[str, Any]
    reward_breakdown: Dict[str, Any]
    illegal_action: bool
    combat_won: bool
    combat_lost: bool
    turn: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_index": self.action_index,
            "command": self.command,
            "reward_breakdown": self.reward_breakdown,
            "illegal_action": self.illegal_action,
            "combat_won": self.combat_won,
            "combat_lost": self.combat_lost,
            "turn": self.turn,
        }


CARD_POOL_ATTACKS = [
    "Strike_R", "Bash", "Anger", "Body Slam", "Clash", "Cleave", "Clothesline",
    "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike", "Pommel Strike",
    "Sword Boomerang", "Thunderclap", "Twin Strike", "Wild Strike", "Blood for Blood",
    "Carnage", "Dropkick", "Hemokinesis", "Pummel", "Rampage", "Reckless Charge",
    "Searing Blow", "Sever Soul", "Uppercut", "Whirlwind", "Bludgeon", "Feed",
    "Fiend Fire", "Immolate", "Reaper"
]

CARD_DEFS: Dict[str, Dict[str, Any]] = {
    "Strike_R": {"type": "ATTACK", "cost": 1, "damage": 6, "targeted": True},
    "Defend_R": {"type": "SKILL", "cost": 1, "block": 5},
    "Bash": {"type": "ATTACK", "cost": 2, "damage": 8, "targeted": True, "apply_vulnerable": 2},
    "Anger": {"type": "ATTACK", "cost": 0, "damage": 6, "targeted": True, "add_copy_to_discard": "Anger"},
    "Armaments": {"type": "SKILL", "cost": 1, "block": 5, "upgrade_hand_all": False, "upgrade_hand_one": True},
    "Body Slam": {"type": "ATTACK", "cost": 1, "damage_from_block": True, "targeted": True},
    "Clash": {"type": "ATTACK", "cost": 0, "damage": 14, "targeted": True, "requires_all_attack": True},
    "Cleave": {"type": "ATTACK", "cost": 1, "aoe_damage": 8},
    "Clothesline": {"type": "ATTACK", "cost": 2, "damage": 12, "targeted": True, "apply_weak": 2},
    "Flex": {"type": "SKILL", "cost": 0, "gain_strength_temp": 2},
    "Havoc": {"type": "SKILL", "cost": 1, "play_top_draw_exhaust": True},
    "Headbutt": {"type": "ATTACK", "cost": 1, "damage": 9, "targeted": True, "move_discard_to_top": True},
    "Heavy Blade": {"type": "ATTACK", "cost": 2, "damage": 14, "targeted": True, "strength_mult": 3},
    "Iron Wave": {"type": "ATTACK", "cost": 1, "damage": 5, "targeted": True, "block": 5},
    "Perfected Strike": {"type": "ATTACK", "cost": 2, "damage": 6, "targeted": True, "bonus_per_strike": 2},
    "Pommel Strike": {"type": "ATTACK", "cost": 1, "damage": 9, "targeted": True, "draw": 1},
    "Shrug It Off": {"type": "SKILL", "cost": 1, "block": 8, "draw": 1},
    "Sword Boomerang": {"type": "ATTACK", "cost": 1, "random_damage": 3, "hits": 3},
    "Thunderclap": {"type": "ATTACK", "cost": 1, "aoe_damage": 4, "apply_vulnerable_all": 1},
    "True Grit": {"type": "SKILL", "cost": 1, "block": 7, "exhaust_random_other": True},
    "Twin Strike": {"type": "ATTACK", "cost": 1, "damage": 5, "hits": 2, "targeted": True},
    "Warcry": {"type": "SKILL", "cost": 0, "draw": 1, "put_one_hand_top_draw": True, "exhaust_self": True},
    "Wild Strike": {"type": "ATTACK", "cost": 1, "damage": 12, "targeted": True, "shuffle_status_draw": "Wound"},
    "Battle Trance": {"type": "SKILL", "cost": 0, "draw": 3, "cannot_draw_more_this_turn": True},
    "Blood for Blood": {"type": "ATTACK", "cost": 4, "damage": 18, "targeted": True},
    "Bloodletting": {"type": "SKILL", "cost": 0, "lose_hp": 3, "gain_energy": 2},
    "Burning Pact": {"type": "SKILL","cost": 1,"exhaust_choose_other": True,"draw": 2,},
    "Carnage": {"type": "ATTACK", "cost": 2, "damage": 20, "targeted": True, "ethereal": True},
    "Combust": {"type": "POWER", "cost": 1, "power": ("Combust", 5), "combust_self_loss": 1},
    "Dark Embrace": {"type": "POWER", "cost": 2, "power": ("Dark Embrace", 1)},
    "Disarm": {"type": "SKILL", "cost": 1, "targeted": True, "reduce_enemy_strength": 2, "exhaust_self": True},
    "Dropkick": {"type": "ATTACK", "cost": 1, "damage": 5, "targeted": True, "bonus_if_vulnerable": {"gain_energy": 1, "draw": 1}},
    "Dual Wield": {"type": "SKILL", "cost": 1, "copy_attack_or_power": 1},
    "Entrench": {"type": "SKILL", "cost": 2, "double_block": True},
    "Evolve": {"type": "POWER", "cost": 1, "power": ("Evolve", 1)},
    "Feel No Pain": {"type": "POWER", "cost": 1, "power": ("Feel No Pain", 3)},
    "Fire Breathing": {"type": "POWER", "cost": 1, "power": ("Fire Breathing", 6)},
    "Flame Barrier": {"type": "SKILL", "cost": 2, "block": 12, "power": ("Flame Barrier", 4)},
    "Ghostly Armor": {"type": "SKILL", "cost": 1, "block": 10, "ethereal": True},
    "Hemokinesis": {"type": "ATTACK", "cost": 1, "damage": 15, "targeted": True, "lose_hp": 2},
    "Infernal Blade": {"type": "SKILL", "cost": 1, "add_random_attack_hand_cost0": True, "exhaust_self": True},
    "Inflame": {"type": "POWER", "cost": 1, "gain_strength": 2},
    "Intimidate": {"type": "SKILL", "cost": 0, "apply_weak_all": 1, "exhaust_self": True},
    "Metallicize": {"type": "POWER", "cost": 1, "power": ("Metallicize", 3)},
    "Power Through": {"type": "SKILL", "cost": 1, "block": 15, "add_status_hand": ("Wound", 2)},
    "Pummel": {"type": "ATTACK", "cost": 1, "damage": 2, "hits": 4, "targeted": True, "exhaust_self": True},
    "Rage": {"type": "SKILL", "cost": 0, "power_until_end_turn": ("Rage", 3)},
    "Rampage": {"type": "ATTACK", "cost": 1, "damage": 8, "targeted": True, "self_damage_bonus_each_use": 5},
    "Reckless Charge": {"type": "ATTACK", "cost": 0, "damage": 7, "targeted": True, "shuffle_status_draw": "Dazed"},
    "Rupture": {"type": "POWER", "cost": 1, "power": ("Rupture", 1)},
    "Searing Blow": {"type": "ATTACK", "cost": 2, "damage": 12, "targeted": True, "multi_upgrade_scaling": True},
    "Second Wind": {"type": "SKILL","cost": 1,"exhaust_all_non_attacks_for_block": 5,"exhaust_self": True,},
    "Seeing Red": {"type": "SKILL", "cost": 1, "gain_energy": 2, "exhaust_self": True},
    "Sentinel": {"type": "SKILL", "cost": 1, "block": 5, "on_exhaust_gain_energy": 2},
    "Sever Soul": {"type": "ATTACK", "cost": 2, "damage": 16, "targeted": True, "exhaust_all_non_attacks": True},
    "Shockwave": {"type": "SKILL", "cost": 2, "apply_weak_all": 3, "apply_vulnerable_all": 3, "exhaust_self": True},
    "Spot Weakness": {"type": "SKILL", "cost": 1, "targeted": True, "gain_strength_if_target_attacks": 3},
    "Uppercut": {"type": "ATTACK", "cost": 2, "damage": 13, "targeted": True, "apply_weak": 1, "apply_vulnerable": 1},
    "Whirlwind": {"type": "ATTACK", "cost": "X", "x_aoe_damage": 5},
    "Barricade": {"type": "POWER", "cost": 3, "power": ("Barricade", 1)},
    "Berserk": {"type": "POWER", "cost": 0, "apply_self_vulnerable": 2, "power": ("Berserk", 1)},
    "Bludgeon": {"type": "ATTACK", "cost": 3, "damage": 32, "targeted": True},
    "Brutality": {"type": "POWER", "cost": 0, "power": ("Brutality", 1)},
    "Corruption": {"type": "POWER", "cost": 3, "power": ("Corruption", 1)},
    "Demon Form": {"type": "POWER", "cost": 3, "power": ("Demon Form", 2)},
    "Double Tap": {"type": "SKILL", "cost": 1, "power_until_end_turn": ("Double Tap", 1)},
    "Exhume": {"type": "SKILL", "cost": 1, "move_exhaust_to_hand": True, "exhaust_self": True},
    "Feed": {"type": "ATTACK", "cost": 1, "damage": 10, "targeted": True, "max_hp_on_kill": 3, "exhaust_self": True},
    "Fiend Fire": {"type": "ATTACK", "cost": 2, "damage_per_exhausted_card": 7, "targeted": True, "exhaust_self": True},
    "Immolate": {"type": "ATTACK", "cost": 2, "aoe_damage": 21, "add_status_discard": ("Burn", 1)},
    "Impervious": {"type": "SKILL", "cost": 2, "block": 30, "exhaust_self": True},
    "Juggernaut": {"type": "POWER", "cost": 2, "power": ("Juggernaut", 5)},
    "Limit Break": {"type": "SKILL", "cost": 1, "double_strength": True, "exhaust_self": True},
    "Offering": {"type": "SKILL", "cost": 0, "lose_hp": 6, "gain_energy": 2, "draw": 3, "exhaust_self": True},
    "Reaper": {"type": "ATTACK", "cost": 2, "aoe_damage": 4, "heal_from_unblocked_damage": True, "exhaust_self": True},
    "Wound": {"type": "STATUS", "cost": 99, "unplayable": True},
    "Dazed": {"type": "STATUS", "cost": 99, "unplayable": True, "ethereal": True},
    "Burn": {"type": "STATUS", "cost": 99, "unplayable": True, "end_turn_burn_loss": 2},
    "Slimed": {"type": "STATUS", "cost": 1, "unplayable": False, "exhaust_self": True},
    "Void": {"type": "STATUS", "cost": 99, "unplayable": True, "on_draw_lose_energy": 1, "ethereal": True},

    "AscendersBane": {"type": "CURSE", "cost": 99, "unplayable": True, "ethereal": True},
    "Clumsy": {"type": "CURSE", "cost": 99, "unplayable": True, "ethereal": True},
    "Curse of the Bell": {"type": "CURSE", "cost": 99, "unplayable": True},
    "Decay": {"type": "CURSE", "cost": 99, "unplayable": True, "end_turn_hp_loss": 2},
    "Doubt": {"type": "CURSE", "cost": 99, "unplayable": True, "end_turn_apply_weak_self": 1},
    "Injury": {"type": "CURSE", "cost": 99, "unplayable": True},
    "Normality": {"type": "CURSE", "cost": 99, "unplayable": True, "max_cards_per_turn": 3},
    "Pain": {"type": "CURSE", "cost": 99, "unplayable": True, "on_other_card_play_hp_loss": 1},
    "Parasite": {"type": "CURSE", "cost": 99, "unplayable": True},
    "Pride": {"type": "CURSE", "cost": 99, "unplayable": True, "ethereal": True},
    "Regret": {"type": "CURSE", "cost": 99, "unplayable": True, "end_turn_hp_loss_per_card_in_hand": 1},
    "Shame": {"type": "CURSE", "cost": 99, "unplayable": True, "end_turn_apply_frail_self": 1},
    "Writhe": {"type": "CURSE", "cost": 99, "unplayable": True, "innate": True},
}

UPGRADE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "Strike_R": {"damage": 9},
    "Defend_R": {"block": 8},
    "Bash": {"damage": 10, "apply_vulnerable": 3},
    "Anger": {"damage": 8},
    "Armaments": {"upgrade_hand_all": True, "upgrade_hand_one": False},
    "Body Slam": {"cost": 0},
    "Clash": {"damage": 18},
    "Cleave": {"aoe_damage": 11},
    "Clothesline": {"damage": 14, "apply_weak": 3},
    "Flex": {"gain_strength_temp": 4},
    "Headbutt": {"damage": 12},
    "Heavy Blade": {"strength_mult": 5},
    "Iron Wave": {"damage": 7, "block": 7},
    "Perfected Strike": {"damage": 8, "bonus_per_strike": 3},
    "Pommel Strike": {"damage": 10, "draw": 2},
    "Shrug It Off": {"block": 11},
    "Sword Boomerang": {"hits": 4},
    "Thunderclap": {"aoe_damage": 7},
    "Twin Strike": {"damage": 7},
    "Warcry": {"draw": 2},
    "Wild Strike": {"damage": 17},
    "Battle Trance": {"draw": 4},
    "Blood for Blood": {"damage": 22},
    "Bloodletting": {"gain_energy": 3},
    "Burning Pact": {"draw": 3},
    "Carnage": {"damage": 28},
    "Combust": {"power": ("Combust", 7)},
    "Disarm": {"reduce_enemy_strength": 3},
    "Dropkick": {"damage": 8},
    "Dual Wield": {"copy_attack_or_power": 2},
    "Entrench": {"cost": 1},
    "Evolve": {"power": ("Evolve", 2)},
    "Feel No Pain": {"power": ("Feel No Pain", 4)},
    "Fire Breathing": {"power": ("Fire Breathing", 10)},
    "Flame Barrier": {"block": 16},
    "Ghostly Armor": {"block": 13},
    "Hemokinesis": {"damage": 20},
    "Inflame": {"gain_strength": 3},
    "Intimidate": {"apply_weak_all": 2},
    "Metallicize": {"power": ("Metallicize", 4)},
    "Power Through": {"block": 20},
    "Pummel": {"damage": 2, "hits": 5},
    "Rage": {"power_until_end_turn": ("Rage", 5)},
    "Rampage": {"damage": 8, "self_damage_bonus_each_use": 8},
    "Reckless Charge": {"damage": 10},
    "Rupture": {"power": ("Rupture", 2)},
    "Second Wind": {"exhaust_all_non_attacks_for_block": 7},
    "Seeing Red": {"cost": 0},
    "Sentinel": {"block": 8, "on_exhaust_gain_energy": 3},
    "Sever Soul": {"damage": 20},
    "Shockwave": {"apply_weak_all": 5, "apply_vulnerable_all": 5},
    "Spot Weakness": {"gain_strength_if_target_attacks": 4},
    "Uppercut": {"apply_weak": 2, "apply_vulnerable": 2},
    "Whirlwind": {"x_aoe_damage": 8},
    "Barricade": {"cost": 2},
    "Berserk": {"apply_self_vulnerable": 1},
    "Bludgeon": {"damage": 42},
    "Brutality": {"cost": 0},
    "Corruption": {"cost": 2},
    "Demon Form": {"power": ("Demon Form", 3)},
    "Double Tap": {"cost": 0},
    "Feed": {"damage": 12, "max_hp_on_kill": 4},
    "Fiend Fire": {"damage_per_exhausted_card": 10},
    "Immolate": {"aoe_damage": 28},
    "Impervious": {"block": 40},
    "Juggernaut": {"power": ("Juggernaut", 7)},
    "Limit Break": {"exhaust_self": False},
    "Offering": {"lose_hp": 6, "gain_energy": 2, "draw": 5},
    "Reaper": {"aoe_damage": 5},
    "Burn": {"end_turn_burn_loss": 4},
    "Havoc": {"cost": 0},
    "True Grit": {"block": 9, "exhaust_random_other": False, "exhaust_choose_other": True},
    "Dark Embrace": {"cost": 1},
    "Infernal Blade": {"cost": 0},
    "Exhume": {"cost": 0},
}

def make_card(card_id: str, upgraded: bool = False, cost_override: Optional[int] = None) -> Dict[str, Any]:
    base = dict(CARD_DEFS[card_id])

    if upgraded and card_id in UPGRADE_OVERRIDES:
        base.update(UPGRADE_OVERRIDES[card_id])

    cost_value = base["cost"] if cost_override is None else cost_override

    card = {
        "id": card_id,
        "type": base["type"],
        "cost": cost_value,
        "upgraded": upgraded,
        "ethereal": bool(base.get("ethereal", False)),
        "exhaust": bool(base.get("exhaust_self", False)),
    }

    if card["cost"] == "X":
        card["cost"] = -1

    return card


ENEMY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Jaw Worm": {
        "hp_range": (40, 44),
        "powers": [],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 11},
        "combat_state": {},
    },
    "Cultist": {
        "hp_range": (48, 54),
        "powers": [],
        "initial_intent": {"intent": "BUFF", "intent_base_damage": 0},
        "combat_state": {"opened": False},
    },
    "Red Louse": {
        "hp_range": (10, 15),
        "powers": "ROLL_CURL_UP",
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 7},
        "combat_state": {"bite_damage_roll": "ROLL_RED_LOUSE_BITE"},
    },
    "Green Louse": {
        "hp_range": (11, 17),
        "powers": "ROLL_CURL_UP",
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 6},
        "combat_state": {"bite_damage_roll": "ROLL_GREEN_LOUSE_BITE"},
    },
    "Fungi Beast": {
        "hp_range": (22, 28),
        "powers": [{"id": "Spore Cloud", "amount": 2}],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 6},
        "combat_state": {},
    },
    "Acid Slime (S)": {
        "hp_range": (8, 12),
        "powers": [],
        "initial_intent": {"intent": "DEBUFF", "intent_base_damage": 0},
        "combat_state": {},
    },
    "Acid Slime (M)": {
        "hp_range": (28, 34),
        "powers": [],
        "initial_intent": {"intent": "ATTACK_DEBUFF", "intent_base_damage": 7},
        "combat_state": {},
    },
    "Spike Slime (S)": {
        "hp_range": (10, 14),
        "powers": [],
        "initial_intent": {"intent": "DEBUFF", "intent_base_damage": 0},
        "combat_state": {},
    },
    "Spike Slime (M)": {
        "hp_range": (28, 32),
        "powers": [],
        "initial_intent": {"intent": "DEBUFF", "intent_base_damage": 0},
        "combat_state": {},
    },
    "Acid Slime (L)": {
        "hp_range": (65, 69),
        "powers": [{"id": "Split", "amount": 1}],
        "initial_intent": {"intent": "ATTACK_DEBUFF", "intent_base_damage": 11},
        "combat_state": {},
    },
    "Spike Slime (L)": {
        "hp_range": (64, 70),
        "powers": [{"id": "Split", "amount": 1}],
        "initial_intent": {"intent": "DEBUFF", "intent_base_damage": 0},
        "combat_state": {},
    },
    "Blue Slaver": {
        "hp_range": (46, 50),
        "powers": [],
        "initial_intent": {"intent": "ATTACK_DEBUFF", "intent_base_damage": 13},
        "combat_state": {},
    },
    "Red Slaver": {
        "hp_range": (46, 50),
        "powers": [],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 13},
        "combat_state": {"turns_taken": 0, "used_entangle": False},
    },
    "Looter": {
        "hp_range": (44, 48),
        "powers": [],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 11},
        "combat_state": {"thief_turn": 0, "gold_stolen": 0, "escaped": False, "smoked": False},
    },
    "Mad Gremlin": {
        "hp_range": (20, 24),
        "powers": [{"id": "Angry", "amount": 1}],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 4},
        "combat_state": {},
    },
    "Sneaky Gremlin": {
        "hp_range": (10, 14),
        "powers": [],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 9},
        "combat_state": {},
    },
    "Fat Gremlin": {
        "hp_range": (13, 17),
        "powers": [],
        "initial_intent": {"intent": "DEBUFF", "intent_base_damage": 0},
        "combat_state": {},
    },
    "Shield Gremlin": {
        "hp_range": (12, 15),
        "powers": [],
        "initial_intent": {"intent": "DEFEND_ALLY", "intent_base_damage": 0, "intent_block": 7},
        "combat_state": {},
    },
    "Gremlin Wizard": {
        "hp_range": (23, 25),
        "powers": [],
        "initial_intent": {"intent": "CHARGE", "intent_base_damage": 0},
        "combat_state": {"charge_turns": 0},
    },
    "Gremlin Nob": {
        "hp_range": (82, 86),
        "powers": [{"id": "Enrage", "amount": 2}],
        "initial_intent": {"intent": "BUFF", "intent_base_damage": 0, "intent_strength_gain": 2},
        "combat_state": {"opened": False, "last_move": None},
    },
    "Lagavulin": {
        "hp_range": (109, 111),
        "powers": [{"id": "Metallicize", "amount": 8}, {"id": "Asleep", "amount": 1}],
        "initial_intent": {"intent": "CHARGE", "intent_base_damage": 0},
        "combat_state": {"asleep_turns": 0, "awakened": False, "pattern_idx": 0},
    },
    "Sentry": {
        "hp_range": (38, 42),
        "powers": [{"id": "Artifact", "amount": 1}],
        "initial_intent": {"intent": "DEBUFF", "intent_base_damage": 0, "intent_add_status": ("Dazed", 2)},
        "combat_state": {"pattern_idx": 0},
    },
    "The Guardian": {
        "hp_range": (240, 250),
        "powers": [{"id": "Mode Shift", "amount": 30}],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 9},
        "combat_state": {
            "pattern_idx": 0,
            "defensive_mode": False,
            "mode_shift_threshold": 30,
        },
    },
    "Slime Boss": {
        "hp_range": (140, 150),
        "powers": [{"id": "Split", "amount": 1}],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 35},
        "combat_state": {
            "pattern_idx": 0,
            "split_done": False,
        },
    },
    "Hexaghost": {
        "hp_range": (250, 264),
        "powers": [],
        "initial_intent": {"intent": "ATTACK", "intent_base_damage": 0},
        "combat_state": {
            "pattern_idx": 0,
            "first_turn_done": False,
        },
    },
}


ACT_1_FIRST_3_WEIGHTS: List[Tuple[float, str]] = [
    (25.0, "cultist"),
    (25.0, "jaw_worm"),
    (25.0, "two_louses"),
    (25.0, "small_slimes"),
]

ACT_1_REMAINING_WEIGHTS: List[Tuple[float, str]] = [
    (6.25, "gremlin_gang"),
    (12.5, "large_slime"),
    (6.25, "lots_of_slimes"),
    (12.5, "blue_slaver"),
    (6.25, "red_slaver"),
    (12.5, "three_louses"),
    (12.5, "two_fungi_beasts"),
    (9.375, "exordium_thugs"),
    (9.375, "exordium_wildlife"),
    (12.5, "looter"),
]

ACT_1_ELITE_WEIGHTS: List[Tuple[float, str]] = [
    (34.0, "gremlin_nob"),
    (33.0, "lagavulin"),
    (33.0, "three_sentries"),
]

ACT_1_ELITE_CHANCE = 0.10

ACT_1_BOSS_WEIGHTS: List[Tuple[float, str]] = [
    (34.0, "the_guardian"),
    (33.0, "slime_boss"),
    (33.0, "hexaghost"),
]

ACT_1_BOSS_CHANCE = 0.04

POTION_DEFS: Dict[str, Dict[str, Any]] = {
    "Fire Potion": {
        "rarity": "Common",
        "requires_target": True,
        "effect_type": "target_damage",
        "amount": 20,
    },
    "Fear Potion": {
        "rarity": "Common",
        "requires_target": True,
        "effect_type": "target_vulnerable",
        "amount": 3,
    },
    "Weak Potion": {
        "rarity": "Common",
        "requires_target": True,
        "effect_type": "target_weak",
        "amount": 3,
    },
    "Explosive Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "aoe_damage",
        "amount": 10,
    },
    "Block Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "self_block",
        "amount": 12,
    },
    "Strength Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "self_strength",
        "amount": 2,
    },
    "Dexterity Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "self_dexterity",
        "amount": 2,
    },
    "Energy Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "self_energy",
        "amount": 2,
    },
    "Swift Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "draw_cards",
        "amount": 3,
    },
    "Blood Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "heal_percent_max_hp",
        "amount": 0.20,
    },
    "Fruit Juice": {
        "rarity": "Rare",
        "requires_target": False,
        "effect_type": "gain_max_hp",
        "amount": 5,
    },
    "Regen Potion": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "self_regen",
        "amount": 5,
    },
    "Speed Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "temp_dexterity",
        "amount": 5,
    },
    "Flex Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "temp_strength",
        "amount": 5,
    },
    "Essence of Steel": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "self_plated_armor",
        "amount": 4,
    },
    "Liquid Bronze": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "self_thorns",
        "amount": 3,
    },
    "Heart of Iron": {
        "rarity": "Rare",
        "requires_target": False,
        "effect_type": "self_metallicize",
        "amount": 6,
    },
    "Ancient Potion": {
        "rarity": "Rare",
        "requires_target": False,
        "effect_type": "self_artifact",
        "amount": 1,
    },
    "Distilled Chaos": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "play_top_draw",
        "amount": 3,
    },
    "Duplication Potion": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "next_cards_twice",
        "amount": 1,
    },
    "Elixir": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "exhaust_hand",
        "amount": 999,
    },
    "Gambler's Brew": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "discard_redraw_hand",
        "amount": 999,
    },
    "Liquid Memories": {
        "rarity": "Uncommon",
        "requires_target": False,
        "effect_type": "return_discard_to_hand_cost0",
        "amount": 1,
    },
    "Attack Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "add_random_attack_cost0",
        "amount": 1,
    },
    "Skill Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "add_random_skill_cost0",
        "amount": 1,
    },
    "Power Potion": {
        "rarity": "Common",
        "requires_target": False,
        "effect_type": "add_random_power_cost0",
        "amount": 1,
    },
    "Ghost in a Jar": {
    "rarity": "Rare",
    "requires_target": False,
    "effect_type": "self_intangible",
    "amount": 1,
    },
}

RELIC_ALIASES: Dict[str, str] = {
    "Paper Frog": "Paper Phrog",
    "Paper Phrog": "Paper Phrog",
    "Cultist Mask": "Cultist Headpiece",
    "Gremlin Mask": "Gremlin Visage",
    "Wing Boots": "Winged Greaves",
    "Captain wheel": "Captain's Wheel",
    "Nloth's Gift": "N'loth's Gift",
    "Sling": "Sling of Courage",
    "Boot": "The Boot",
}

RELIC_DEFS: Dict[str, Dict[str, Any]] = {
    # Ironclad
    "Burning Blood": {
        "hooks": ["combat_end_win"],
        "heal_amount": 6,
    },
    "Black Blood": {
        "hooks": ["combat_end_win"],
        "heal_amount": 12,
    },
    "Red Skull": {
        "hooks": ["damage_bonus"],
        "strength_below_half_hp": 3,
    },
    "Paper Phrog": {
        "hooks": ["enemy_vulnerable_multiplier"],
        "vulnerable_multiplier": 1.75,
    },
    "Champion Belt": {
        "hooks": ["on_apply_vulnerable"],
        "extra_weak_on_vulnerable": 1,
    },
    "Self-Forming Clay": {
        "hooks": ["on_player_hp_loss", "start_turn"],
        "block_next_turn_on_hp_loss": 3,
    },
    "Runic Cube": {
        "hooks": ["on_player_hp_loss"],
        "draw_on_hp_loss": 1,
    },
    "Anchor": {
        "hooks": ["combat_start"],
        "start_combat_block": 10,
    },
    "Bag of Marbles": {
        "hooks": ["combat_start"],
        "apply_vulnerable_all": 1,
    },
    "Bag of Preparation": {
        "hooks": ["combat_start"],
        "draw_on_combat_start": 2,
    },
    "Blood Vial": {
        "hooks": ["combat_start"],
        "heal_on_combat_start": 2,
    },
    "Bronze Scales": {
        "hooks": ["combat_start"],
        "gain_thorns": 3,
    },
    "Lantern": {
        "hooks": ["combat_start"],
        "energy_on_combat_start": 1,
    },
    "Oddly Smooth Stone": {
        "hooks": ["combat_start"],
        "gain_dexterity": 1,
    },
    "Orichalcum": {
        "hooks": ["end_turn_before_enemies"],
        "block_if_none": 6,
    },
    "Vajra": {
        "hooks": ["combat_start"],
        "gain_strength": 1,
    },
    "Akabeko": {
        "hooks": ["first_attack_bonus"],
        "first_attack_bonus_damage": 8,
    },
    "Pen Nib": {
        "hooks": ["every_10_attacks_double"],
    },
    "Nunchaku": {
        "hooks": ["every_10_attacks_energy"],
    },
    "Kunai": {
        "hooks": ["every_3_attacks"],
        "gain_dexterity": 1,
    },
    "Shuriken": {
        "hooks": ["every_3_attacks"],
        "gain_strength": 1,
    },
    "Horn Cleat": {
        "hooks": ["turn_2_block"],
        "block_amount": 14,
    },
    "Incense Burner": {
        "hooks": ["every_6_turns_intangible"],
    },
    "Thread and Needle": {
        "hooks": ["combat_start"],
        "gain_plated_armor": 4,
    },
    "Calipers": {
        "hooks": ["retain_block"],
    },
    "Torii": {
        "hooks": ["small_damage_reduce"],
    },
    "Tungsten Rod": {
        "hooks": ["reduce_all_hp_loss"],
    },
    "Mercury Hourglass": {
        "hooks": ["start_turn_damage_all"],
        "damage": 3,
    },
    "Stone Calendar": {
        "hooks": ["turn_7_damage_all"],
        "damage": 52,
    },
    "Gambling Chip": {
        "hooks": ["combat_start_discard_redraw"],
    },
    "Lizard Tail": {
        "hooks": ["revive_once"],
    },
    "Preserved Insect": {
        "hooks": ["elite_combat_start"],
        "elite_hp_multiplier": 0.75,
    },
    "Sling of Courage": {
        "hooks": ["elite_combat_start"],
        "elite_gain_strength": 2,
    },
    "Black Star": {
        "hooks": ["elite_reward"],
        "extra_relics_on_elite_win": 1,
    },
    "Slaver's Collar": {
        "hooks": ["elite_combat_start", "boss_combat_start"],
        "elite_energy_on_combat_start": 1,
        "boss_energy_on_combat_start": 1,
    },
    "Philosopher's Stone": {
        "hooks": ["boss_combat_start"],
        "boss_energy_on_combat_start": 1,
        "enemy_strength_all": 1,
    },
    "Sacred Bark": {
        "hooks": ["double_potion_effect"],
        "potion_multiplier": 2.0,
    },
    "Velvet Choker": {
        "hooks": ["card_play_limit"],
        "max_cards_per_turn": 6,
    },
}

class MockGameBackend:
    """
    Full-ish mock combat backend for Ironclad.
    Honest note: this is a simplified simulation, not a pixel-perfect clone.
    """

    def __init__(self, cfg: Optional[Config] = None, seed: int = 42):
        self.cfg = cfg or get_default_config()
        self.rng = random.Random(seed)
        self.state: Optional[Dict[str, Any]] = None
        self.turn: int = 0
        self.act1_normal_fight_index: int = 0
        self.current_update: int = 0

        self.base_draw_pool = [
            make_card("Strike_R"), make_card("Strike_R"), make_card("Strike_R"), make_card("Strike_R"),
            make_card("Defend_R"), make_card("Defend_R"), make_card("Defend_R"), make_card("Defend_R"),
            make_card("Bash"), make_card("Pommel Strike"), make_card("Shrug It Off"),
            make_card("Anger"), make_card("Inflame"), make_card("Twin Strike"),
        ]

    def set_training_progress(self, update_idx: int) -> None:
        self.current_update = int(update_idx)

    def _get_curriculum_phase(self) -> Dict[str, float]:
        curriculum = self.cfg.combat_curriculum

        if not curriculum.enabled or len(curriculum.phases) == 0:
            return {
                "elite_chance": ACT_1_ELITE_CHANCE,
                "boss_chance": ACT_1_BOSS_CHANCE,
            }

        for phase in curriculum.phases:
            if self.current_update <= phase.until_update:
                return {
                    "elite_chance": float(phase.elite_chance),
                    "boss_chance": float(phase.boss_chance),
                }

        last = curriculum.phases[-1]
        return {
            "elite_chance": float(last.elite_chance),
            "boss_chance": float(last.boss_chance),
        }

    def _normalize_relic_name(self, name: str) -> str:
        raw = str(name or "").strip()
        return RELIC_ALIASES.get(raw, raw)

    def _get_player_relics(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        player = state.get("player", {})
        relics = player.get("relics", [])
        if isinstance(relics, list):
            return relics
        return []

    def _get_player_relic_names(self, state: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for relic in self._get_player_relics(state):
            raw_name = relic.get("name", relic.get("id", ""))
            names.append(self._normalize_relic_name(str(raw_name)))
        return names

    def _player_has_relic(self, state: Dict[str, Any], relic_name: str) -> bool:
        target = self._normalize_relic_name(relic_name)
        return target in self._get_player_relic_names(state)

    def _get_relic_def(self, relic_name: str) -> Dict[str, Any]:
        canon = self._normalize_relic_name(relic_name)
        return RELIC_DEFS.get(canon, {})

    def _combat_relic_runtime(self, state: Dict[str, Any]) -> Dict[str, Any]:
        meta = state.setdefault("combat_meta", {})
        runtime = meta.setdefault("relic_runtime", {})
        return runtime

    def _apply_power_to_monster(self, state: Dict[str, Any], monster: Dict[str, Any], power_name: str, amount: int) -> None:
        amount = int(amount)
        if amount == 0 or self._monster_dead(monster):
            return

        self._add_power(monster, power_name, amount)

        if power_name == "Vulnerable" and amount > 0 and self._player_has_relic(state, "Champion Belt"):
            belt = self._get_relic_def("Champion Belt")
            extra_weak = int(belt.get("extra_weak_on_vulnerable", 1))
            if extra_weak > 0:
                self._add_power(monster, "Weak", extra_weak)

    def _apply_relics_on_combat_start(self, state: Dict[str, Any]) -> None:
        player = state["player"]
        runtime = self._combat_relic_runtime(state)

        runtime["self_forming_clay_pending"] = 0

        for relic_name in self._get_player_relic_names(state):
            spec = self._get_relic_def(relic_name)
            hooks = set(spec.get("hooks", []))

            if "combat_start" not in hooks:
                continue

            if spec.get("heal_on_combat_start", 0) > 0:
                self._heal_player(player, int(spec["heal_on_combat_start"]))

            if spec.get("start_combat_block", 0) > 0:
                player["block"] = int(player.get("block", 0)) + int(spec["start_combat_block"])

            if spec.get("gain_strength", 0) > 0:
                self._add_power(player, "Strength", int(spec["gain_strength"]))

            if spec.get("gain_dexterity", 0) > 0:
                self._add_power(player, "Dexterity", int(spec["gain_dexterity"]))

            if spec.get("gain_plated_armor", 0) > 0:
                self._add_power(player, "Plated Armor", int(spec["gain_plated_armor"]))

            if spec.get("gain_thorns", 0) > 0:
                self._add_power(player, "Thorns", int(spec["gain_thorns"]))

            if spec.get("energy_on_combat_start", 0) > 0:
                state["energy"] = int(state.get("energy", 0)) + int(spec["energy_on_combat_start"])

            if spec.get("apply_vulnerable_all", 0) > 0:
                for monster in state.get("monsters", []):
                    if not self._monster_dead(monster):
                        self._apply_power_to_monster(
                            state,
                            monster,
                            "Vulnerable",
                            int(spec["apply_vulnerable_all"]),
                        )

            if spec.get("draw_on_combat_start", 0) > 0:
                self._draw_cards(state, int(spec["draw_on_combat_start"]))

        if self._player_has_relic(state, "Gambling Chip"):
            hand = state["hand"]
            while hand:
                state["discard_pile"].append(hand.pop())

            self._draw_cards(state, 5)

        is_elite = bool(state.get("combat_meta", {}).get("is_elite", False))
        is_boss = bool(state.get("combat_meta", {}).get("is_boss", False))

        if is_elite and self._player_has_relic(state, "Preserved Insect"):
            insect = self._get_relic_def("Preserved Insect")
            hp_mult = float(insect.get("elite_hp_multiplier", 0.75))
            for monster in state.get("monsters", []):
                if self._monster_dead(monster):
                    continue
                new_hp = max(1, int(round(int(monster.get("current_hp", 1)) * hp_mult)))
                monster["current_hp"] = new_hp

        if is_elite and self._player_has_relic(state, "Sling of Courage"):
            sling = self._get_relic_def("Sling of Courage")
            self._add_power(player, "Strength", int(sling.get("elite_gain_strength", 2)))

        if is_elite and self._player_has_relic(state, "Slaver's Collar"):
            collar = self._get_relic_def("Slaver's Collar")
            state["energy"] = int(state.get("energy", 0)) + int(collar.get("elite_energy_on_combat_start", 1))

        if is_boss and self._player_has_relic(state, "Slaver's Collar"):
            collar = self._get_relic_def("Slaver's Collar")
            state["energy"] = int(state.get("energy", 0)) + int(collar.get("boss_energy_on_combat_start", 1))

        if is_elite and self._player_has_relic(state, "Black Star"):
            star = self._get_relic_def("Black Star")
            state["combat_meta"]["elite_relic_reward_count"] = 1 + int(star.get("extra_relics_on_elite_win", 1))

        if is_boss and self._player_has_relic(state, "Philosopher's Stone"):
            stone = self._get_relic_def("Philosopher's Stone")
            state["energy"] = int(state.get("energy", 0)) + int(stone.get("boss_energy_on_combat_start", 1))
            enemy_str = int(stone.get("enemy_strength_all", 1))
            for monster in state.get("monsters", []):
                if not self._monster_dead(monster):
                    self._add_power(monster, "Strength", enemy_str)

    def _apply_relics_on_player_hp_loss(self, state: Dict[str, Any], amount: int, source: str) -> None:
        if amount <= 0:
            return

        runtime = self._combat_relic_runtime(state)

        if self._player_has_relic(state, "Self-Forming Clay"):
            clay = self._get_relic_def("Self-Forming Clay")
            runtime["self_forming_clay_pending"] = int(runtime.get("self_forming_clay_pending", 0)) + int(
                clay.get("block_next_turn_on_hp_loss", 3)
            )

        if self._player_has_relic(state, "Runic Cube"):
            cube = self._get_relic_def("Runic Cube")
            draw_n = int(cube.get("draw_on_hp_loss", 1))
            self._draw_cards(state, draw_n)

    def _apply_relics_at_start_of_turn(self, state: Dict[str, Any]) -> None:
        runtime = self._combat_relic_runtime(state)

        # reset attack counter for Kunai / Shuriken
        state["combat_meta"]["attacks_this_turn"] = 0

        pending = int(runtime.get("self_forming_clay_pending", 0))
        if pending > 0:
            self._gain_block(state, pending)
            runtime["self_forming_clay_pending"] = 0

        # Incense Burner
        if self._player_has_relic(state, "Incense Burner"):
            runtime = self._combat_relic_runtime(state)
            counter = int(runtime.get("incense_counter", 0)) + 1
            runtime["incense_counter"] = counter

            if counter >= 6:
                self._add_power(state["player"], "Intangible", 1)
                runtime["incense_counter"] = 0

    def _apply_relics_after_attack(self, state: Dict[str, Any]) -> None:
        meta = state["combat_meta"]

        meta["attack_counter"] = int(meta.get("attack_counter", 0)) + 1

        # Nunchaku
        if self._player_has_relic(state, "Nunchaku"):
            if meta["attack_counter"] % 10 == 0:
                state["energy"] = int(state.get("energy", 0)) + 1

        # Pen Nib
        if self._player_has_relic(state, "Pen Nib"):
            if meta["attack_counter"] % 10 == 0:
                meta["next_attack_double"] = True

        # Kunai / Shuriken
        meta["attacks_this_turn"] = int(meta.get("attacks_this_turn", 0)) + 1
        if meta["attacks_this_turn"] == 3:
            if self._player_has_relic(state, "Kunai"):
                self._add_power(state["player"], "Dexterity", 1)
            if self._player_has_relic(state, "Shuriken"):
                self._add_power(state["player"], "Strength", 1)

    def _apply_relics_at_end_turn_before_enemies(self, state: Dict[str, Any]) -> None:
        player = state["player"]

        if self._player_has_relic(state, "Orichalcum"):
            spec = self._get_relic_def("Orichalcum")
            if int(player.get("block", 0)) == 0:
                player["block"] = int(player.get("block", 0)) + int(spec.get("block_if_none", 6))

    def _apply_relics_on_combat_end_win(self, state: Dict[str, Any]) -> None:
        player = state["player"]

        for relic_name in self._get_player_relic_names(state):
            spec = self._get_relic_def(relic_name)
            if "combat_end_win" not in set(spec.get("hooks", [])):
                continue

            heal_amount = int(spec.get("heal_amount", 0))
            if heal_amount > 0:
                self._heal_player(player, heal_amount)

    def _random_louse_name(self) -> str:
        return "Red Louse" if self.rng.random() < 0.5 else "Green Louse"

    def _random_small_slime_name(self) -> str:
        return "Acid Slime (S)" if self.rng.random() < 0.5 else "Spike Slime (S)"

    def _random_medium_slime_name(self) -> str:
        return "Acid Slime (M)" if self.rng.random() < 0.5 else "Spike Slime (M)"

    def _sample_gremlin_gang_names(self) -> List[str]:
        pool = ["Mad Gremlin", "Mad Gremlin", "Sneaky Gremlin", "Sneaky Gremlin",
                "Fat Gremlin", "Fat Gremlin", "Gremlin Wizard", "Shield Gremlin"]
        self.rng.shuffle(pool)
        return pool[:4]

    def _weighted_choice(self, weighted_entries: List[Tuple[float, str]]) -> str:
        total = sum(weight for weight, _ in weighted_entries)
        r = self.rng.random() * total
        acc = 0.0
        for weight, value in weighted_entries:
            acc += weight
            if r <= acc:
                return value
        return weighted_entries[-1][1]

    def _sample_exordium_thugs_names(self) -> List[str]:
        first = self.rng.choice([
            self._random_louse_name(),
            self._random_medium_slime_name(),
        ])
        second = self.rng.choice([
            "Looter",
            "Cultist",
            "Blue Slaver",
            "Red Slaver",
        ])
        return [first, second]

    def _sample_exordium_wildlife_names(self) -> List[str]:
        first = self.rng.choice(["Fungi Beast", "Jaw Worm"])
        second = self.rng.choice([
            self._random_louse_name(),
            self._random_medium_slime_name(),
        ])
        return [first, second]

    def _resolve_act1_encounter_names(self, encounter_key: str) -> List[str]:
        if encounter_key == "cultist":
            return ["Cultist"]
        if encounter_key == "jaw_worm":
            return ["Jaw Worm"]
        if encounter_key == "two_louses":
            return [self._random_louse_name(), self._random_louse_name()]
        if encounter_key == "small_slimes":
            return [self._random_medium_slime_name(), self._random_small_slime_name()]

        if encounter_key == "gremlin_gang":
            return self._sample_gremlin_gang_names()
        if encounter_key == "large_slime":
            return [self.rng.choice(["Acid Slime (L)", "Spike Slime (L)"])]
        if encounter_key == "lots_of_slimes":
            return ["Spike Slime (S)", "Spike Slime (S)", "Spike Slime (S)", "Acid Slime (S)", "Acid Slime (S)"]
        if encounter_key == "blue_slaver":
            return ["Blue Slaver"]
        if encounter_key == "red_slaver":
            return ["Red Slaver"]
        if encounter_key == "three_louses":
            return [self._random_louse_name(), self._random_louse_name(), self._random_louse_name()]
        if encounter_key == "two_fungi_beasts":
            return ["Fungi Beast", "Fungi Beast"]
        if encounter_key == "exordium_thugs":
            return self._sample_exordium_thugs_names()
        if encounter_key == "exordium_wildlife":
            return self._sample_exordium_wildlife_names()
        if encounter_key == "looter":
            return ["Looter"]

        raise ValueError(f"Unknown Act 1 encounter key: {encounter_key}")

    def _resolve_act1_elite_names(self, elite_key: str) -> List[str]:
        if elite_key == "gremlin_nob":
            return ["Gremlin Nob"]
        if elite_key == "lagavulin":
            return ["Lagavulin"]
        if elite_key == "three_sentries":
            return ["Sentry", "Sentry", "Sentry"]
        raise ValueError(f"Unknown Act 1 elite key: {elite_key}")

    def _resolve_act1_boss_names(self, boss_key: str) -> List[str]:
        if boss_key == "the_guardian":
            return ["The Guardian"]
        if boss_key == "slime_boss":
            return ["Slime Boss"]
        if boss_key == "hexaghost":
            return ["Hexaghost"]
        raise ValueError(f"Unknown Act 1 boss key: {boss_key}")

    def _sample_enemy_group_with_meta(self) -> Tuple[List[Dict[str, Any]], bool, bool]:
        phase = self._get_curriculum_phase()
        boss_chance = phase["boss_chance"]
        elite_chance = phase["elite_chance"]

        boss_roll = self.rng.random()
        if boss_roll < boss_chance:
            boss_key = self._weighted_choice(ACT_1_BOSS_WEIGHTS)
            names = self._resolve_act1_boss_names(boss_key)
            return [self._make_enemy(name) for name in names], False, True

        elite_roll = self.rng.random()
        if elite_roll < elite_chance:
            elite_key = self._weighted_choice(ACT_1_ELITE_WEIGHTS)
            names = self._resolve_act1_elite_names(elite_key)
            return [self._make_enemy(name) for name in names], True, False

        if self.act1_normal_fight_index < 3:
            encounter_key = self._weighted_choice(ACT_1_FIRST_3_WEIGHTS)
        else:
            encounter_key = self._weighted_choice(ACT_1_REMAINING_WEIGHTS)

        self.act1_normal_fight_index += 1
        names = self._resolve_act1_encounter_names(encounter_key)
        return [self._make_enemy(name) for name in names], False, False

    def _starter_deck_pool(self) -> List[Dict[str, Any]]:
        return [
            make_card("Strike_R"), make_card("Strike_R"), make_card("Strike_R"), make_card("Strike_R"),
            make_card("Defend_R"), make_card("Defend_R"), make_card("Defend_R"), make_card("Defend_R"),
            make_card("Bash"),
        ]

    def _sample_bonus_cards(self, k: int = 5) -> List[Dict[str, Any]]:
        bonus_ids = [
            # Attacks / basics
            "Pommel Strike", "Anger", "Twin Strike", "Headbutt", "Iron Wave",
            "Cleave", "Clothesline", "Uppercut", "Carnage", "Pummel",
            "Whirlwind", "Sword Boomerang", "Perfected Strike", "Heavy Blade",
            "Wild Strike", "Thunderclap", "Dropkick", "Hemokinesis", "Rampage",

            # Skills / block / setup
            "Shrug It Off", "True Grit", "Warcry", "Armaments", "Battle Trance",
            "Ghostly Armor", "Power Through", "Flame Barrier", "Disarm",
            "Second Wind", "Entrench", "Seeing Red", "Bloodletting", "Burning Pact",

            # Powers / scaling
            "Inflame", "Metallicize", "Combust", "Feel No Pain",
            "Dark Embrace", "Rupture", "Fire Breathing", "Brutality",

            # High impact / rarer-feeling but already coded
            "Shockwave", "Offering", "Immolate", "Fiend Fire", "Reaper",
        ]

        picked = self.rng.sample(bonus_ids, k=min(k, len(bonus_ids)))
        cards = []
        for cid in picked:
            upgraded = self.rng.random() < 0.20
            cards.append(make_card(cid, upgraded=upgraded))
        return cards

    def _build_random_player_deck(self, bonus_cards: int = 4) -> List[Dict[str, Any]]:
        deck = self._starter_deck_pool() + self._sample_bonus_cards(k=bonus_cards)
        self.rng.shuffle(deck)
        return deck

    def _roll_template_powers(self, raw_powers: Any) -> List[Dict[str, Any]]:
        if raw_powers == "ROLL_CURL_UP":
            return [{"id": "Curl Up", "amount": self.rng.randint(3, 7)}]
        if raw_powers is None:
            return []
        return copy.deepcopy(raw_powers)

    def _roll_template_combat_state(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        state = copy.deepcopy(raw_state)

        for key, value in list(state.items()):
            if value == "ROLL_RED_LOUSE_BITE":
                state[key] = self.rng.randint(5, 7)
            elif value == "ROLL_GREEN_LOUSE_BITE":
                state[key] = self.rng.randint(6, 8)

        return state

    def _make_enemy(self, name: str) -> Dict[str, Any]:
        template = ENEMY_TEMPLATES.get(name)
        if template is None:
            raise ValueError(f"Unknown enemy: {name}")

        hp_min, hp_max = template["hp_range"]
        hp = self.rng.randint(hp_min, hp_max)

        enemy = {
            "name": name,
            "current_hp": hp,
            "max_hp": hp,
            "block": 0,
            "powers": self._roll_template_powers(template.get("powers")),
            "isDead": False,
            "move_history": [],
            "last_move": None,
            "combat_state": self._roll_template_combat_state(template.get("combat_state", {})),
        }

        enemy.update(copy.deepcopy(template.get("initial_intent", {})))
        return enemy

    def _sample_enemy_group(self) -> List[Dict[str, Any]]:
        if self.act1_normal_fight_index < 3:
            encounter_key = self._weighted_choice(ACT_1_FIRST_3_WEIGHTS)
        else:
            encounter_key = self._weighted_choice(ACT_1_REMAINING_WEIGHTS)

        self.act1_normal_fight_index += 1
        names = self._resolve_act1_encounter_names(encounter_key)
        return [self._make_enemy(name) for name in names]

    def _roll_random_potion_name(self) -> str:
        rarity_roll = self.rng.random()
        if rarity_roll < 0.65:
            rarity = "Common"
        elif rarity_roll < 0.90:
            rarity = "Uncommon"
        else:
            rarity = "Rare"

        candidates = [
            name for name, spec in POTION_DEFS.items()
            if spec.get("rarity") == rarity
        ]
        return self.rng.choice(candidates)

    def _make_potion(self, name: str) -> Dict[str, Any]:
        spec = POTION_DEFS[name]
        return {
            "name": name,
            "usable": True,
            "empty": False,
            "requires_target": bool(spec.get("requires_target", False)),
            "rarity": spec.get("rarity", "Common"),
        }

    def _build_starting_potions(self) -> List[Dict[str, Any]]:
        potions: List[Dict[str, Any]] = []
        num_real_potions = 3

        for _ in range(num_real_potions):
            potions.append(self._make_potion(self._roll_random_potion_name()))

        while len(potions) < 5:
            potions.append({
                "name": "Empty Slot",
                "usable": False,
                "empty": True,
                "requires_target": False,
            })

        return potions

    def _build_starting_relics(self, extra_relics: int = 1) -> List[Dict[str, Any]]:
        relics = [{"name": "Burning Blood"}]

        pool = [name for name in RELIC_DEFS.keys() if name != "Burning Blood"]
        extra = self.rng.sample(pool, extra_relics)

        for r in extra:
            relics.append({"name": r})

        return relics

    def reset(self) -> Dict[str, Any]:
        self.turn = 1
        self.act1_normal_fight_index = 0

        monsters, is_elite, is_boss = self._sample_enemy_group_with_meta()

        if is_boss:
            bonus_cards = 6
        elif is_elite:
            bonus_cards = 5
        else:
            bonus_cards = 4

        if is_boss:
            extra_relics = 3
        elif is_elite:
            extra_relics = 2
        else:
            extra_relics = 1

        draw_pile = self._build_random_player_deck(bonus_cards)

        self.rng.shuffle(draw_pile)
        hand = draw_pile[:5]
        draw_pile = draw_pile[5:]

        self.state = {
            "in_combat": True,
            "combat_over": False,
            "game_over": False,
            "turn": self.turn,
            "energy": 3,
            "player": {
                "current_hp": 80,
                "max_hp": 80,
                "block": 0,
                "powers": [],
                "relics": self._build_starting_relics(extra_relics),
                "gold": 99,
            },
            "hand": hand,
            "draw_pile": draw_pile,
            "discard_pile": [],
            "exhaust_pile": [],
            "potions": self._build_starting_potions(),
            "combat_meta": {
                "hp_loss_count": 0,
                "cards_played_this_turn": 0,
                "attacks_played_this_turn": 0,
                "cannot_draw_more_this_turn": False,
                "double_tap_charges": 0,
                "strength_down_at_end_turn": 0,
                "dexterity_down_at_end_turn": 0,
                "pending_max_hp_gain": 0,
                "last_x_energy_spent": 0,
                "relic_runtime": {},
                "attack_counter": 0,
                "attacks_this_turn": 0,
                "next_attack_double": False,
                "first_attack_done": False,
                "is_elite": is_elite,
                "is_boss": is_boss,
                "elite_relic_reward_count": 1,
                "boss_relic_reward_count": 1,
            },
            "hp_loss_breakdown": {
                "enemy": 0,
                "burn": 0,
                "pain": 0,
                "decay": 0,
                "other": 0,
            },
            "monsters": monsters,
        }

        self._apply_relics_on_combat_start(self.state)
        self._update_enemy_intents(self.state)
        return copy.deepcopy(self.state)

    def get_state(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        return copy.deepcopy(self.state)

    def _consume_potion_slot(self, state: Dict[str, Any], potion_index: int) -> None:
        potions = state.get("potions", [])
        if 0 <= potion_index < len(potions):
            potions[potion_index] = {
                "name": "Empty Slot",
                "usable": False,
                "empty": True,
                "requires_target": False,
            }

    def _add_card_to_hand_cost0(self, state: Dict[str, Any], card_id: str) -> None:
        state["hand"].append(make_card(card_id, cost_override=0))

    def _random_card_id_by_type(self, card_type: str) -> Optional[str]:
        candidates = []
        for cid, spec in CARD_DEFS.items():
            if spec.get("type") != card_type:
                continue
            if spec.get("unplayable", False):
                continue
            candidates.append(cid)

        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _heal_player_flat(self, state: Dict[str, Any], amount: int) -> None:
        player = state["player"]
        player["current_hp"] = min(
            int(player.get("max_hp", 0)),
            int(player.get("current_hp", 0)) + int(amount),
        )

    def _discard_hand_and_redraw_same_count(self, state: Dict[str, Any]) -> None:
        n = len(state["hand"])
        while state["hand"]:
            state["discard_pile"].append(state["hand"].pop())
        self._draw_cards(state, n)

    def _return_random_discard_to_hand_cost0(self, state: Dict[str, Any]) -> bool:
        if not state["discard_pile"]:
            return False
        idx = self.rng.randrange(len(state["discard_pile"]))
        card = state["discard_pile"].pop(idx)
        card["cost"] = 0
        state["hand"].append(card)
        return True

    def _play_top_draw_cards(self, state: Dict[str, Any], n: int) -> None:
        for _ in range(int(n)):
            if not state["draw_pile"]:
                break
            card = state["draw_pile"].pop(0)
            state["hand"].append(card)
            cmd = CombatCommand(
                command_type="play_card",
                hand_index=len(state["hand"]) - 1,
                target_index=self._first_alive_monster_index(state),
            )
            state, _ = self._apply_play_card(state, cmd)

    def _apply_use_potion(
            self,
            state: Dict[str, Any],
            potion_index: int,
            target_index: Optional[int] = None,
    ) -> bool:
        potions = state.get("potions", [])
        if potion_index < 0 or potion_index >= len(potions):
            return False

        potion = potions[potion_index]
        if potion.get("empty", False) or not potion.get("usable", True):
            return False

        name = potion.get("name", "")
        spec = POTION_DEFS.get(name)
        if spec is None:
            print(f"[potion-invalid] unknown potion: {name}")
            return False

        effect_type = spec["effect_type"]
        amount = spec["amount"]

        if self._player_has_relic(state, "Sacred Bark"):
            bark = self._get_relic_def("Sacred Bark")
            mult = float(bark.get("potion_multiplier", 2.0))
            if isinstance(amount, (int, float)):
                amount = amount * mult

        player = state["player"]

        if spec.get("requires_target", False):
            if target_index is None:
                print(f"[potion-invalid] {name} requires target but none provided")
                return False
            if not (0 <= target_index < len(state["monsters"])):
                return False
            target = state["monsters"][target_index]
            if self._monster_dead(target):
                return False
        else:
            target = None

        if effect_type == "target_damage":
            self._deal_damage_to_monster(
                state,
                target,
                int(amount),
                source_card={"id": name, "type": "POTION"},
            )
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "target_vulnerable":
            # print(f"[potion] {name} used on {target.get('name', '?')} (slot={potion_index}) -> +{amount} Vulnerable")
            self._apply_power_to_monster(state, target, "Vulnerable", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "target_weak":
            # print(f"[potion] {name} used on {target.get('name', '?')} (slot={potion_index}) -> +{amount} Weak")
            self._apply_power_to_monster(state, target, "Weak", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "aoe_damage":
            #print(f"[potion] {name} used (slot={potion_index}) -> {amount} AOE dmg")
            for m in state["monsters"]:
                if not self._monster_dead(m):
                    self._deal_damage_to_monster(
                        state,
                        m,
                        int(amount),
                        source_card={"id": name, "type": "POTION"},
                    )
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_block":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} block")
            player["block"] = int(player.get("block", 0)) + int(amount)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_strength":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Strength")
            self._add_power(player, "Strength", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_dexterity":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Dexterity")
            self._add_power(player, "Dexterity", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_energy":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Energy")
            state["energy"] = int(state.get("energy", 0)) + int(amount)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "draw_cards":
            #print(f"[potion] {name} used (slot={potion_index}) -> draw {amount}")
            self._draw_cards(state, int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "heal_percent_max_hp":
            heal_amt = max(1, int(int(player.get("max_hp", 0)) * float(amount)))
            #print(f"[potion] {name} used (slot={potion_index}) -> heal {heal_amt}")
            self._heal_player_flat(state, heal_amt)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "gain_max_hp":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} max HP")
            player["max_hp"] = int(player.get("max_hp", 0)) + int(amount)
            player["current_hp"] = min(int(player["max_hp"]), int(player.get("current_hp", 0)) + int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_regen":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Regen")
            self._add_power(player, "Regeneration", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "temp_dexterity":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} temporary Dexterity")
            self._add_power(player, "Dexterity", int(amount))
            state["combat_meta"]["dexterity_down_at_end_turn"] = int(
                state["combat_meta"].get("dexterity_down_at_end_turn", 0)
            ) + int(amount)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "temp_strength":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} temporary Strength")
            self._add_power(player, "Strength", int(amount))
            state["combat_meta"]["strength_down_at_end_turn"] = int(
                state["combat_meta"].get("strength_down_at_end_turn", 0)
            ) + int(amount)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_plated_armor":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Plated Armor")
            self._add_power(player, "Plated Armor", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_thorns":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Thorns")
            self._add_power(player, "Thorns", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_metallicize":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Metallicize")
            self._add_power(player, "Metallicize", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_artifact":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Artifact")
            self._add_power(player, "Artifact", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "self_intangible":
            #print(f"[potion] {name} used (slot={potion_index}) -> +{amount} Intangible")
            self._add_power(player, "Intangible", int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "play_top_draw":
            #print(f"[potion] {name} used (slot={potion_index}) -> play top {amount} draw cards")
            self._play_top_draw_cards(state, int(amount))
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "next_cards_twice":
            #print(f"[potion] {name} used (slot={potion_index}) -> next card played twice")
            state["combat_meta"]["double_tap_charges"] = int(
                state["combat_meta"].get("double_tap_charges", 0)
            ) + int(amount)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "exhaust_hand":
            #print(f"[potion] {name} used (slot={potion_index}) -> exhaust hand")
            while state["hand"]:
                card = state["hand"].pop()
                state["exhaust_pile"].append(card)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "discard_redraw_hand":
            #print(f"[potion] {name} used (slot={potion_index}) -> discard hand and redraw")
            self._discard_hand_and_redraw_same_count(state)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "return_discard_to_hand_cost0":
            #print(f"[potion] {name} used (slot={potion_index}) -> return discard to hand cost 0")
            success = self._return_random_discard_to_hand_cost0(state)
            if success:
                self._consume_potion_slot(state, potion_index)
            return success

        if effect_type == "add_random_attack_cost0":
            picked = self._random_card_id_by_type("ATTACK")
            if picked is None:
                return False
            #print(f"[potion] {name} used (slot={potion_index}) -> add {picked} cost 0")
            self._add_card_to_hand_cost0(state, picked)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "add_random_skill_cost0":
            picked = self._random_card_id_by_type("SKILL")
            if picked is None:
                return False
            #print(f"[potion] {name} used (slot={potion_index}) -> add {picked} cost 0")
            self._add_card_to_hand_cost0(state, picked)
            self._consume_potion_slot(state, potion_index)
            return True

        if effect_type == "add_random_power_cost0":
            picked = self._random_card_id_by_type("POWER")
            if picked is None:
                return False
            #print(f"[potion] {name} used (slot={potion_index}) -> add {picked} cost 0")
            self._add_card_to_hand_cost0(state, picked)
            self._consume_potion_slot(state, potion_index)
            return True

        print(f"[potion-invalid] unsupported potion effect: {name} / {effect_type}")
        return False

    def _first_alive_monster_index(self, state: Dict[str, Any]) -> Optional[int]:
        for i, monster in enumerate(state.get("monsters", [])):
            if not self._monster_dead(monster):
                return i
        return None

    def step(self, command: CombatCommand) -> Tuple[Dict[str, Any], bool]:
        if self.state is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        state = copy.deepcopy(self.state)
        illegal_action = False
        was_combat_over = bool(state.get("combat_over", False))

        state["hp_loss_breakdown"] = {
            "enemy": 0,
            "burn": 0,
            "pain": 0,
            "decay": 0,
            "other": 0,
        }

        if state.get("combat_over", False) or state.get("game_over", False):
            return state, False

        if command.command_type == "play_card":
            state, illegal_action = self._apply_play_card(state, command)

        elif command.command_type == "end_turn":
            state = self._apply_end_turn(state)

        elif command.command_type == "use_potion":
            potion_index = command.potion_index
            success = self._apply_use_potion(
                state,
                potion_index=potion_index if potion_index is not None else -1,
                target_index=command.target_index,
            )
            if not success:
                illegal_action = True

        elif command.command_type in {
            "choose_hand_card",
            "choose_option",
            "choose_discard_target",
            "choose_exhaust_target",
        }:
            success = self._apply_pending_choice(state, command)
            if not success:
                illegal_action = True

        else:
            illegal_action = True

        self._refresh_terminal_flags(state)

        player_dead = int(state["player"].get("current_hp", 0)) <= 0
        all_dead = all(self._monster_dead(m) for m in state["monsters"])

        if state.get("combat_over", False) and not was_combat_over and all_dead and not player_dead:
            self._apply_relics_on_combat_end_win(state)

        self.state = state
        return copy.deepcopy(self.state), illegal_action

    def _set_pending_choice(
        self,
        state: Dict[str, Any],
        choice_type: str,
        valid_hand_indices: Optional[List[int]] = None,
        options: Optional[List[Any]] = None,
        source_card_id: Optional[str] = None,
    ) -> None:
        state["pending_choice"] = {
            "choice_type": choice_type,
            "valid_hand_indices": list(valid_hand_indices or []),
            "options": list(options or []),
            "source_card_id": source_card_id,
        }

    def _clear_pending_choice(self, state: Dict[str, Any]) -> None:
        state.pop("pending_choice", None)

    def _apply_pending_choice(self, state: Dict[str, Any], command: CombatCommand) -> bool:
        choice = state.get("pending_choice")
        if not isinstance(choice, dict):
            return False

        choice_type = str(choice.get("choice_type", "")).strip().lower()
        valid_hand_indices = choice.get("valid_hand_indices", []) or []
        options = choice.get("options", []) or []
        source_card_id = choice.get("source_card_id")

        if choice_type == "choose_option":
            idx = command.target_index
            if idx is None or not (0 <= idx < len(options)):
                return False

            chosen = options[idx]
            if isinstance(chosen, dict):
                chosen_card = copy.deepcopy(chosen)
            else:
                chosen_card = make_card(str(chosen))

            state["hand"].append(chosen_card)
            self._clear_pending_choice(state)
            return True

        if choice_type == "choose_hand_card":
            idx = command.hand_index
            if idx is None or idx not in valid_hand_indices or not (0 <= idx < len(state["hand"])):
                return False

            card = state["hand"].pop(idx)
            state["draw_pile"].insert(0, card)
            self._clear_pending_choice(state)
            return True

        if choice_type == "choose_discard_target":
            idx = command.hand_index
            if idx is None or idx not in valid_hand_indices or not (0 <= idx < len(state["hand"])):
                return False

            card = state["hand"].pop(idx)
            state["discard_pile"].append(card)
            self._clear_pending_choice(state)
            return True

        if choice_type == "choose_exhaust_target":
            idx = command.hand_index
            if idx is None or idx not in valid_hand_indices or not (0 <= idx < len(state["hand"])):
                return False

            card = state["hand"].pop(idx)
            self._exhaust_card_object(state, card)

            if source_card_id == "Burning Pact":
                self._draw_cards(state, 2)

            self._clear_pending_choice(state)
            return True

        return False

    def _get_effective_card_def(self, card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        card_id = card.get("id")
        base = CARD_DEFS.get(card_id)
        if base is None:
            return None

        effective = dict(base)
        if card.get("upgraded", False) and card_id in UPGRADE_OVERRIDES:
            effective.update(UPGRADE_OVERRIDES[card_id])

        return effective

    def _apply_play_card(self, state: Dict[str, Any], command: CombatCommand) -> Tuple[Dict[str, Any], bool]:
        hand = state["hand"]
        if command.hand_index is None or not (0 <= command.hand_index < len(hand)):
            return state, True

        card = hand[command.hand_index]
        card_id = card["id"]

        # Curse: Normality -> max 3 cartes jouées par tour
        if self._has_card_in_hand_or_drawn(state, "Normality"):
            if int(state["combat_meta"].get("cards_played_this_turn", 0)) >= 3:
                return state, True

        if self._player_has_relic(state, "Velvet Choker"):
            choker = self._get_relic_def("Velvet Choker")
            max_cards = int(choker.get("max_cards_per_turn", 6))
            if int(state["combat_meta"].get("cards_played_this_turn", 0)) >= max_cards:
                return state, True

        card_def = self._get_effective_card_def(card)
        if card_def is None or card_def.get("unplayable", False):
            return state, True

        energy = int(state.get("energy", 0))
        cost = self._card_cost_for_state(card, state)
        if cost > energy:
            return state, True

        if card_id == "Clash" and not self._all_other_cards_are_attack(hand, command.hand_index):
            return state, True

        target = None
        if card_def.get("targeted", False):
            if command.target_index is None or command.target_index < 0 or command.target_index >= len(state["monsters"]):
                return state, True
            target = state["monsters"][command.target_index]
            if self._monster_dead(target):
                return state, True

        remove_from_hand = True

        x_energy_spent = None
        if card_id == "Whirlwind":
            x_energy_spent = energy

        state["energy"] = max(0, energy - cost)

        if x_energy_spent is not None:
            state["combat_meta"]["last_x_energy_spent"] = x_energy_spent
        else:
            state["combat_meta"]["last_x_energy_spent"] = 0

        self._on_card_played_before_resolution(state, card)

        repeat_count = 1
        if card_def["type"] == "ATTACK" and state["combat_meta"].get("double_tap_charges", 0) > 0:
            repeat_count = 2
            state["combat_meta"]["double_tap_charges"] -= 1

        for _ in range(repeat_count):
            self._resolve_card_effect(state, card, card_def, command.target_index)

        if remove_from_hand:
            if card_def.get("exhaust_self", False) or card.get("exhaust", False):
                self._move_card_from_hand_to_exhaust(state, command.hand_index)
            else:
                self._move_card_from_hand_to_discard(state, command.hand_index)

        # Curse: Pain -> lose 1 HP whenever another card is played
        if card_id != "Pain" and self._has_card_in_hand_or_drawn(state, "Pain"):
            self._lose_hp(state["player"], state, 1, source="pain")

        # Ethereal leftovers are handled at end turn, not here.
        self._refresh_terminal_flags(state)
        return state, False

    def _resolve_card_effect(self, state: Dict[str, Any], card: Dict[str, Any], card_def: Dict[str, Any], target_index: Optional[int]) -> None:
        player = state["player"]
        meta = state["combat_meta"]

        if card_def.get("lose_hp"):
            self._lose_hp_from_card(player, state, int(card_def["lose_hp"]))

        if card_def["type"] == "ATTACK":
            self._apply_relics_after_attack(state)

        if card_def.get("gain_energy"):
            state["energy"] = int(state.get("energy", 0)) + int(card_def["gain_energy"])

        if card_def.get("gain_strength"):
            self._add_power(player, "Strength", int(card_def["gain_strength"]))

        if card_def.get("gain_strength_temp"):
            gain = int(card_def["gain_strength_temp"])
            self._add_power(player, "Strength", gain)
            meta["strength_down_at_end_turn"] = int(meta.get("strength_down_at_end_turn", 0)) + gain

        if card_def.get("double_strength"):
            self._set_power(player, "Strength", self._get_power_amount(player, "Strength") * 2)

        if card_def.get("double_block"):
            old = int(player.get("block", 0))
            self._gain_block(state, old)

        if card_def.get("power"):
            power_name, amount = card_def["power"]
            self._add_power(player, power_name, int(amount))

        if card_def.get("power_until_end_turn"):
            power_name, amount = card_def["power_until_end_turn"]
            amount = int(amount)

            if power_name == "Double Tap":
                meta["double_tap_charges"] = int(meta.get("double_tap_charges", 0)) + amount
            else:
                self._add_power(player, power_name, amount)

        if card_def.get("apply_self_vulnerable"):
            self._add_power(player, "Vulnerable", int(card_def["apply_self_vulnerable"]))

        if card_def.get("block"):
            self._gain_block(state, int(card_def["block"]))

        if card_def.get("draw") and card["id"] != "Burning Pact":
            self._draw_cards(state, int(card_def["draw"]))

        if card_def.get("cannot_draw_more_this_turn"):
            meta["cannot_draw_more_this_turn"] = True

        if card_def.get("upgrade_hand_one") and state["hand"]:
            for c in state["hand"]:
                if c["id"] != card["id"]:
                    c["upgraded"] = True
                    break

        if card_def.get("upgrade_hand_all"):
            for c in state["hand"]:
                c["upgraded"] = True

        if card_def.get("put_one_hand_top_draw"):
            valid = [i for i, c in enumerate(state["hand"]) if c["id"] != card["id"]]
            if valid:
                self._set_pending_choice(
                    state,
                    choice_type="choose_hand_card",
                    valid_hand_indices=valid,
                    source_card_id=card["id"],
                )

        if card_def.get("move_discard_to_top"):
            self._move_random_discard_to_top_draw(state)

        if card_def.get("play_top_draw_exhaust"):
            self._play_top_draw_and_exhaust(state)

        if card_def.get("copy_attack_or_power"):
            options = []
            for c in state["hand"]:
                cdef = CARD_DEFS.get(c["id"], {})
                if cdef.get("type") in {"ATTACK", "POWER"}:
                    options.append(copy.deepcopy(c))
            if options:
                self._set_pending_choice(
                    state,
                    choice_type="choose_option",
                    options=options[: self.cfg.combat_action.max_choose_option_actions],
                    source_card_id=card["id"],
                )

        if card_def.get("move_exhaust_to_hand"):
            self._move_random_exhaust_to_hand(state)

        if card_def.get("add_random_attack_hand_cost0"):
            picked = self.rng.choice(CARD_POOL_ATTACKS)
            state["hand"].append(make_card(picked, cost_override=0))

        if card_def.get("add_copy_to_discard"):
            state["discard_pile"].append(make_card(card_def["add_copy_to_discard"]))

        if card_def.get("add_status_discard"):
            status_name, n = card_def["add_status_discard"]
            for _ in range(int(n)):
                state["discard_pile"].append(make_card(status_name))

        if card_def.get("add_status_hand"):
            status_name, n = card_def["add_status_hand"]
            for _ in range(int(n)):
                state["hand"].append(make_card(status_name))

        if card_def.get("shuffle_status_draw"):
            state["draw_pile"].append(make_card(card_def["shuffle_status_draw"]))
            self.rng.shuffle(state["draw_pile"])

        if card_def.get("exhaust_choose_other"):
            valid = [i for i, c in enumerate(state["hand"]) if c["id"] != card["id"]]
            if valid:
                self._set_pending_choice(
                    state,
                    choice_type="choose_exhaust_target",
                    valid_hand_indices=valid,
                    source_card_id=card["id"],
                )

        elif card_def.get("exhaust_random_other"):
            self._exhaust_random_other_card_from_hand(
                state,
                exclude_card_id=card["id"],
            )

        if card_def.get("exhaust_all_non_attacks_for_block"):
            per_card_block = int(card_def["exhaust_all_non_attacks_for_block"])

            to_exhaust = []
            for i, c in enumerate(state["hand"]):
                if c["id"] == card["id"]:
                    continue
                cdef = CARD_DEFS.get(c["id"], {})
                if cdef.get("type") != "ATTACK":
                    to_exhaust.append(i)

            exhausted_count = 0
            for i in reversed(to_exhaust):
                card_obj = state["hand"].pop(i)
                self._exhaust_card_object(state, card_obj)
                exhausted_count += 1

            if exhausted_count > 0:
                self._gain_block(state, exhausted_count * per_card_block)

        damage_done = 0

        # Targeted / attack resolution
        if card_def.get("targeted"):
            target = state["monsters"][target_index]
            if not self._monster_dead(target):
                damage_done += self._deal_targeted_card_damage(state, card, card_def, target)
                if card_def.get("apply_vulnerable"):
                    self._apply_power_to_monster(state, target, "Vulnerable", int(card_def["apply_vulnerable"]))
                if card_def.get("apply_weak"):
                    self._apply_power_to_monster(state, target, "Weak", int(card_def["apply_weak"]))
                if card_def.get("reduce_enemy_strength"):
                    self._add_power(target, "Strength", -int(card_def["reduce_enemy_strength"]))
                if card_def.get("gain_strength_if_target_attacks") and target.get("intent") == "ATTACK":
                    self._add_power(player, "Strength", int(card_def["gain_strength_if_target_attacks"]))
                if card_def.get("bonus_if_vulnerable") and self._get_power_amount(target, "Vulnerable") > 0:
                    bonus = card_def["bonus_if_vulnerable"]
                    if bonus.get("gain_energy"):
                        state["energy"] += int(bonus["gain_energy"])
                    if bonus.get("draw"):
                        self._draw_cards(state, int(bonus["draw"]))

        if card_def.get("aoe_damage"):
            for m in state["monsters"]:
                if not self._monster_dead(m):
                    damage_done += self._deal_damage_to_monster(
                        state,
                        m,
                        int(card_def["aoe_damage"]),
                        source_card=card,
                    )

        if card_def.get("apply_vulnerable_all"):
            for m in state["monsters"]:
                if not self._monster_dead(m):
                    self._apply_power_to_monster(
                        state,
                        m,
                        "Vulnerable",
                        int(card_def["apply_vulnerable_all"]),
                    )

        if card_def.get("apply_weak_all"):
            for m in state["monsters"]:
                if not self._monster_dead(m):
                    self._apply_power_to_monster(
                        state,
                        m,
                        "Weak",
                        int(card_def["apply_weak_all"]),
                    )

        if card_def.get("x_aoe_damage"):
            base = int(card_def.get("x_aoe_damage", 0))
            hits = int(meta.get("last_x_energy_spent", 0))

            player = state["player"]
            strength = self._get_power_amount(player, "Strength")
            damage = max(0, base + strength)

            for monster in state["monsters"]:
                if monster.get("isDead"):
                    continue
                for _ in range(hits):
                    self._deal_damage_to_monster(state, monster, damage)

        if card_def.get("random_damage") and card_def.get("hits"):
            for _ in range(int(card_def["hits"])):
                m = self._choose_random_alive_monster(state)
                if m is not None:
                    damage_done += self._deal_damage_to_monster(state, m, int(card_def["random_damage"]), source_card=card)

        if card_def.get("hits") and card_def.get("damage") and card_def.get("targeted"):
            target = state["monsters"][target_index]
            for _ in range(int(card_def["hits"]) - 1):
                if not self._monster_dead(target):
                    damage_done += self._deal_damage_to_monster(state, target, self._compute_base_damage(card, card_def, state), source_card=card)

        if card_def.get("heal_from_unblocked_damage") and damage_done > 0:
            self._heal_player(player, damage_done)

        if card_def.get("max_hp_on_kill"):
            target = state["monsters"][target_index]
            if self._monster_dead(target):
                player["max_hp"] = int(player.get("max_hp", 0)) + int(card_def["max_hp_on_kill"])
                player["current_hp"] = min(int(player["max_hp"]), int(player["current_hp"]) + int(card_def["max_hp_on_kill"]))

        if card["id"] == "Fiend Fire":
            if target_index is not None:
                target = state["monsters"][target_index]
                if not self._monster_dead(target):
                    exhausted_now = self._exhaust_all_cards_except_index(state, keep_index=None)
                    per = int(card_def["damage_per_exhausted_card"])
                    self._deal_damage_to_monster(
                        state,
                        target,
                        per * exhausted_now,
                        source_card=card,
                    )

    def _deal_targeted_card_damage(self, state: Dict[str, Any], card: Dict[str, Any], card_def: Dict[str, Any], target: Dict[str, Any]) -> int:
        if card_def.get("damage_from_block"):
            damage = int(state["player"].get("block", 0))
            return self._deal_damage_to_monster(state, target, damage, source_card=card)

        if card_def.get("damage_per_exhausted_card"):
            return 0

        damage = self._compute_base_damage(card, card_def, state)
        return self._deal_damage_to_monster(state, target, damage, source_card=card)

    def _compute_base_damage(self, card: Dict[str, Any], card_def: Dict[str, Any], state: Dict[str, Any]) -> int:
        player = state["player"]
        strength = self._get_power_amount(player, "Strength")

        if self._player_has_relic(state, "Red Skull"):
            cur_hp = int(player.get("current_hp", 0))
            max_hp = max(1, int(player.get("max_hp", 1)))
            if cur_hp <= (max_hp // 2):
                bonus = int(self._get_relic_def("Red Skull").get("strength_below_half_hp", 3))
                strength += bonus

        base = int(card_def.get("damage", 0))

        meta = state["combat_meta"]

        if self._player_has_relic(state, "Akabeko"):
            if meta.get("first_attack_done") is not True:
                base += int(self._get_relic_def("Akabeko").get("first_attack_bonus_damage", 8))
                meta["first_attack_done"] = True

        if meta.get("next_attack_double"):
            base *= 2
            meta["next_attack_double"] = False

        if card["id"] == "Perfected Strike":
            strike_count = 0
            for pile_name in ["hand", "draw_pile", "discard_pile", "exhaust_pile"]:
                for c in state.get(pile_name, []):
                    if "Strike" in c.get("id", ""):
                        strike_count += 1
            base += strike_count * int(card_def.get("bonus_per_strike", 2))

        elif card["id"] == "Heavy Blade":
            base += strength * int(card_def.get("strength_mult", 3))
            strength = 0

        elif card["id"] == "Blood for Blood":
            base += max(0, int(state["combat_meta"].get("hp_loss_count", 0)))

        elif card["id"] == "Rampage":
            bonus = int(card.get("combat_bonus_damage", 0))
            base += bonus
            card["combat_bonus_damage"] = bonus + int(card_def.get("self_damage_bonus_each_use", 5))

        elif card["id"] == "Searing Blow":
            if card.get("upgraded"):
                base += 4

        total = base + strength
        return max(0, total)

    def _execute_enemy_intent(self, state: Dict[str, Any], monster: Dict[str, Any]) -> None:
        if self._monster_dead(monster):
            return

        intent = monster.get("intent", "ATTACK")

        if intent == "ATTACK":
            hits = int(monster.get("intent_hits", 1))
            base_damage = int(monster.get("intent_base_damage", 0))

            for _ in range(max(1, hits)):
                if int(state["player"].get("current_hp", 0)) <= 0:
                    break
                self._deal_monster_attack_to_player(state, monster, base_damage)

            monster["intent_hits"] = 1

            if monster.get("name") == "Looter":
                player = state["player"]
                stolen_now = min(15, int(player.get("gold", 99)))
                player["gold"] = int(player.get("gold", 99)) - stolen_now
                monster.setdefault("combat_state", {})["gold_stolen"] = (
                        int(monster.setdefault("combat_state", {}).get("gold_stolen", 0)) + stolen_now
                )

            return

        if intent == "ATTACK_DEFEND":
            self._deal_monster_attack_to_player(state, monster, int(monster.get("intent_base_damage", 0)))
            monster["block"] = int(monster.get("block", 0)) + int(monster.get("intent_block", 0))
            return

        if intent == "BUFF":
            if monster.get("name") == "Cultist":
                self._add_power(monster, "Ritual", 3)
            else:
                gain = int(monster.get("intent_strength_gain", 3))
                self._add_power(monster, "Strength", gain)
            return

        if intent == "BUFF_DEFEND":
            monster["block"] = int(monster.get("block", 0)) + int(monster.get("intent_block", 0))
            self._add_power(monster, "Strength", int(monster.get("intent_strength_gain", 0)))
            return

        if intent == "DEFEND":
            monster["block"] = int(monster.get("block", 0)) + int(monster.get("intent_block", 0))
            return

        if intent == "DEBUFF":
            if monster.get("intent_apply_weak", 0) > 0:
                self._add_power(state["player"], "Weak", int(monster["intent_apply_weak"]))
            if monster.get("intent_apply_vulnerable", 0) > 0:
                self._add_power(state["player"], "Vulnerable", int(monster["intent_apply_vulnerable"]))
            if monster.get("intent_apply_frail", 0) > 0:
                self._add_power(state["player"], "Frail", int(monster["intent_apply_frail"]))
            if monster.get("intent_entangle", False):
                self._add_power(state["player"], "Entangled", 1)
            return

        if intent == "ATTACK_DEBUFF":
            self._deal_monster_attack_to_player(state, monster, int(monster.get("intent_base_damage", 0)))

            if monster.get("intent_apply_weak", 0) > 0:
                self._add_power(state["player"], "Weak", int(monster["intent_apply_weak"]))
            if monster.get("intent_apply_vulnerable", 0) > 0:
                self._add_power(state["player"], "Vulnerable", int(monster["intent_apply_vulnerable"]))
            if monster.get("intent_apply_frail", 0) > 0:
                self._add_power(state["player"], "Frail", int(monster["intent_apply_frail"]))

            add_status = monster.get("intent_add_status")
            if add_status:
                status_id, n = add_status
                for _ in range(int(n)):
                    state["discard_pile"].append(make_card(status_id))
            return

        if intent == "DEFEND_ALLY":
            target = self._choose_lowest_hp_ally(
                state,
                exclude_dead=True,
                exclude_monster=monster,
            )
            if target is not None:
                target["block"] = int(target.get("block", 0)) + int(monster.get("intent_block", 0))
            else:
                self._deal_monster_attack_to_player(state, monster, 6)
            return

        if intent == "ESCAPE":
            monster["isDead"] = True
            monster.setdefault("combat_state", {})["escaped"] = True
            return

        if intent == "CHARGE":
            return

    def _apply_end_turn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        player = state["player"]

        self._handle_ethereal_and_burn(state)

        # End-turn player powers
        combust = self._get_power_amount(player, "Combust")
        if combust > 0:
            self._lose_hp_from_card(player, state, 1)
            for m in state["monsters"]:
                if not self._monster_dead(m):
                    self._deal_damage_to_monster(state, m, combust, source_card={"id": "Combust"})

        metallicize = self._get_power_amount(player, "Metallicize")
        if metallicize > 0:
            self._gain_block(state, metallicize)

        self._apply_relics_at_end_turn_before_enemies(state)

        # Enemies act
        for monster in state["monsters"]:
            if self._monster_dead(monster):
                continue
            self._execute_enemy_intent(state, monster)

        # End-turn monster powers (e.g. Cultist Ritual)
        for monster in state["monsters"]:
            if self._monster_dead(monster):
                continue

            ritual = self._get_power_amount(monster, "Ritual")
            if ritual > 0:
                self._add_power(monster, "Strength", ritual)

        # End turn temporary strength loss
        str_down = int(state["combat_meta"].get("strength_down_at_end_turn", 0))
        if str_down > 0:
            self._add_power(player, "Strength", -str_down)
        state["combat_meta"]["strength_down_at_end_turn"] = 0

        # Intangible decays by 1 at end of turn
        intang = self._get_power_amount(player, "Intangible")
        if intang > 0:
            self._add_power(player, "Intangible", -1)

        if self._get_power_amount(player, "Barricade") <= 0:
            if self._player_has_relic(state, "Calipers"):
                player["block"] = max(0, int(player.get("block", 0)) - 15)
            else:
                player["block"] = 0

        self.turn += 1
        state["turn"] = self.turn

        energy = 3
        if self._get_power_amount(player, "Berserk") > 0:
            energy += self._get_power_amount(player, "Berserk")
        state["energy"] = energy

        self._start_turn_powers(state)
        self._draw_new_hand(state, target_hand_size=5)
        state["combat_meta"]["cards_played_this_turn"] = 0
        state["combat_meta"]["attacks_played_this_turn"] = 0
        state["combat_meta"]["cannot_draw_more_this_turn"] = False
        self._update_enemy_intents(state)
        return state

    def _choose_lowest_hp_ally(
            self,
            state: Dict[str, Any],
            exclude_dead: bool = True,
            exclude_monster: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        alive = []
        for m in state["monsters"]:
            if exclude_dead and self._monster_dead(m):
                continue
            if exclude_monster is not None and m is exclude_monster:
                continue
            alive.append(m)

        if not alive:
            return None

        alive.sort(key=lambda m: int(m.get("current_hp", 0)))
        return alive[0]

    def _start_turn_powers(self, state: Dict[str, Any]) -> None:
        player = state["player"]

        if self._player_has_relic(state, "Mercury Hourglass"):
            dmg = int(self._get_relic_def("Mercury Hourglass").get("damage", 3))
            for m in state["monsters"]:
                if not self._monster_dead(m):
                    self._deal_damage_to_monster(state, m, dmg)

        if self._player_has_relic(state, "Horn Cleat"):
            if int(state.get("turn", 1)) == 2:
                block = int(self._get_relic_def("Horn Cleat").get("block_amount", 14))
                self._gain_block(state, block)

        if self._player_has_relic(state, "Stone Calendar"):
            if int(state.get("turn", 1)) == 7:
                dmg = int(self._get_relic_def("Stone Calendar").get("damage", 52))
                for m in state["monsters"]:
                    if not self._monster_dead(m):
                        self._deal_damage_to_monster(state, m, dmg)

        self._apply_relics_at_start_of_turn(state)

        if self._get_power_amount(player, "Brutality") > 0:
            self._lose_hp_from_card(player, state, 1)
            self._draw_cards(state, 1)

        demon = self._get_power_amount(player, "Demon Form")
        if demon > 0:
            self._add_power(player, "Strength", demon)

    def _handle_ethereal_and_burn(self, state: Dict[str, Any]) -> None:
        new_hand = []
        for card in state["hand"]:
            if card.get("ethereal"):
                self._exhaust_card_object(state, card)
            else:
                new_hand.append(card)
        state["hand"] = new_hand

        hand_snapshot = list(state["hand"])

        for card in hand_snapshot:
            card_id = card["id"]

            if card_id == "Burn":
                self._lose_hp(
                    state["player"],
                    state,
                    4 if card.get("upgraded", False) else 2,
                    source="burn",
                )

            elif card_id == "Decay":
                self._lose_hp(state["player"], state, 2, source="decay")

            elif card_id == "Regret":
                self._lose_hp(state["player"], state, len(state["hand"]), source="other")

            elif card_id == "Doubt":
                self._add_power(state["player"], "Weak", 1)

            elif card_id == "Shame":
                self._add_power(state["player"], "Frail", 1)

    def _draw_new_hand(self, state: Dict[str, Any], target_hand_size: int = 5) -> None:
        while state["hand"]:
            state["discard_pile"].append(state["hand"].pop())

        while len(state["hand"]) < target_hand_size:
            if len(state["draw_pile"]) == 0:
                if len(state["discard_pile"]) == 0:
                    break
                state["draw_pile"] = state["discard_pile"]
                state["discard_pile"] = []
                self.rng.shuffle(state["draw_pile"])

            card = state["draw_pile"].pop(0)
            state["hand"].append(card)

            if card["id"] == "Void":
                state["energy"] = max(0, int(state.get("energy", 0)) - 1)

            if card["id"] in {"Wound", "Dazed", "Burn", "Slimed", "Void"}:
                evolve = self._get_power_amount(state["player"], "Evolve")
                fire = self._get_power_amount(state["player"], "Fire Breathing")
                if evolve > 0:
                    self._draw_cards(state, evolve)
                if fire > 0:
                    for m in state["monsters"]:
                        if not self._monster_dead(m):
                            self._deal_damage_to_monster(state, m, fire, source_card={"id": "Fire Breathing"})

    def _draw_cards(self, state: Dict[str, Any], n: int) -> None:
        if n <= 0 or state["combat_meta"].get("cannot_draw_more_this_turn", False):
            return
        for _ in range(n):
            if len(state["draw_pile"]) == 0:
                if len(state["discard_pile"]) == 0:
                    return
                state["draw_pile"] = state["discard_pile"]
                state["discard_pile"] = []
                self.rng.shuffle(state["draw_pile"])
            card = state["draw_pile"].pop(0)
            state["hand"].append(card)

            if card["id"] == "Void":
                state["energy"] = max(0, int(state.get("energy", 0)) - 1)

            if card["id"] in {"Wound", "Dazed", "Burn", "Slimed", "Void"}:
                evolve = self._get_power_amount(state["player"], "Evolve")
                fire = self._get_power_amount(state["player"], "Fire Breathing")
                if evolve > 0:
                    self._draw_cards(state, evolve)
                if fire > 0:
                    for m in state["monsters"]:
                        if not self._monster_dead(m):
                            self._deal_damage_to_monster(state, m, fire, source_card={"id": "Fire Breathing"})

    def _move_card_from_hand_to_discard(self, state: Dict[str, Any], hand_index: int) -> None:
        if 0 <= hand_index < len(state["hand"]):
            state["discard_pile"].append(state["hand"].pop(hand_index))

    def _move_card_from_hand_to_exhaust(self, state: Dict[str, Any], hand_index: int) -> None:
        if 0 <= hand_index < len(state["hand"]):
            card = state["hand"].pop(hand_index)
            self._exhaust_card_object(state, card)

    def _exhaust_card_object(self, state: Dict[str, Any], card: Dict[str, Any]) -> None:
        state["exhaust_pile"].append(card)
        player = state["player"]
        feel = self._get_power_amount(player, "Feel No Pain")
        dark = self._get_power_amount(player, "Dark Embrace")
        if feel > 0:
            self._gain_block(state, feel)
        if dark > 0:
            self._draw_cards(state, dark)
        if card["id"] == "Sentinel":
            state["energy"] += 2

    def _gain_block(self, state: Dict[str, Any], amount: int) -> None:
        if amount <= 0:
            return
        state["player"]["block"] = int(state["player"].get("block", 0)) + amount
        jug = self._get_power_amount(state["player"], "Juggernaut")
        if jug > 0:
            target = self._choose_random_alive_monster(state)
            if target is not None:
                self._deal_damage_to_monster(state, target, jug, source_card={"id": "Juggernaut"})

    def _deal_damage_to_monster(self, state: Dict[str, Any], monster: Dict[str, Any], damage: int, source_card: Optional[Dict[str, Any]] = None) -> int:

        if self._monster_dead(monster):
            return 0

        damage = max(0, int(damage))
        if damage <= 0:
            return 0

        curl = self._get_power_amount(monster, "Curl Up")
        combat_state = monster.setdefault("combat_state", {})
        if curl > 0 and not combat_state.get("curl_up_used", False):
            monster["block"] = int(monster.get("block", 0)) + curl
            combat_state["curl_up_used"] = True

        if self._get_power_amount(monster, "Vulnerable") > 0:
            vuln_mult = 1.5
            if self._player_has_relic(state, "Paper Phrog"):
                vuln_mult = float(self._get_relic_def("Paper Phrog").get("vulnerable_multiplier", 1.75))
            damage = int(round(damage * vuln_mult))

        block = int(monster.get("block", 0))
        absorbed = min(block, damage)
        monster["block"] = block - absorbed
        hp_damage = damage - absorbed
        monster["current_hp"] = max(0, int(monster.get("current_hp", 0)) - hp_damage)

        if monster.get("name") == "Lagavulin":
            cstate = monster.setdefault("combat_state", {})
            if not cstate.get("awakened", False) and hp_damage > 0:
                cstate["awakened"] = True
                self._add_power(monster, "Asleep", -999)
                self._add_power(monster, "Metallicize", -999)

        if monster.get("name") == "The Guardian":
            cstate = monster.setdefault("combat_state", {})
            if not cstate.get("defensive_mode", False) and hp_damage > 0:
                threshold = int(cstate.get("mode_shift_threshold", 30))
                threshold -= hp_damage
                cstate["mode_shift_threshold"] = threshold
                if threshold <= 0:
                    cstate["defensive_mode"] = True
                    cstate["pattern_idx"] = 0
                    monster["block"] = int(monster.get("block", 0)) + 20
                    cstate["mode_shift_threshold"] = 30

        if monster.get("name") == "Slime Boss":
            cstate = monster.setdefault("combat_state", {})
            max_hp = int(monster.get("max_hp", 1))
            cur_hp = int(monster.get("current_hp", 0))

            if not cstate.get("split_done", False) and cur_hp > 0 and cur_hp <= max_hp // 2:
                cstate["split_done"] = True
                split_hp = max(1, cur_hp // 2)

                medium_a = self._make_enemy("Acid Slime (M)")
                medium_b = self._make_enemy("Spike Slime (M)")
                medium_a["current_hp"] = min(split_hp, int(medium_a["max_hp"]))
                medium_b["current_hp"] = min(split_hp, int(medium_b["max_hp"]))

                monster["isDead"] = True
                monster["current_hp"] = 0
                state["monsters"].append(medium_a)
                state["monsters"].append(medium_b)

        if int(monster["current_hp"]) <= 0:
            monster["isDead"] = True

            if monster.get("name") == "Fungi Beast":
                spore = self._get_power_amount(monster, "Spore Cloud")
                if spore > 0:
                    self._add_power(state["player"], "Vulnerable", spore)

        return hp_damage

    def _deal_monster_attack_to_player(self, state: Dict[str, Any], monster: Dict[str, Any], damage: int) -> None:
        player = state["player"]
        if self._get_power_amount(player, "Weak") > 0:
            pass

        damage = max(0, int(damage))

        # Intangible: all incoming damage becomes 1
        if damage > 0 and self._get_power_amount(player, "Intangible") > 0:
            damage = 1

        block = int(player.get("block", 0))
        absorbed = min(block, damage)
        player["block"] = block - absorbed
        hp_damage = damage - absorbed

        if hp_damage > 0:
            self._lose_hp(player, state, hp_damage, source="enemy")

        flame = self._get_power_amount(player, "Flame Barrier")
        if flame > 0:
            self._deal_damage_to_monster(state, monster, flame, source_card={"id": "Flame Barrier"})

        thorns = self._get_power_amount(player, "Thorns")
        if thorns > 0 and not self._monster_dead(monster):
            self._deal_damage_to_monster(state, monster, thorns, source_card={"id": "Thorns"})

    def _heal_player(self, player: Dict[str, Any], amount: int) -> None:
        player["current_hp"] = min(int(player["max_hp"]), int(player["current_hp"]) + max(0, int(amount)))

    def _lose_hp(self, player: Dict[str, Any], state: Dict[str, Any], amount: int, source: str = "other") -> None:
        amount = max(0, int(amount))
        if amount <= 0:
            return

        # Intangible: all HP loss becomes 1
        if self._get_power_amount(player, "Intangible") > 0:
            amount = 1

        if self._player_has_relic(state, "Tungsten Rod"):
            amount = max(0, amount - 1)

        if self._player_has_relic(state, "Torii"):
            if 0 < amount <= 5:
                amount = 1

        player["current_hp"] = max(0, int(player["current_hp"]) - amount)
        state["combat_meta"]["hp_loss_count"] = int(state["combat_meta"].get("hp_loss_count", 0)) + 1

        breakdown = state.setdefault(
            "hp_loss_breakdown",
            {"enemy": 0, "burn": 0, "pain": 0, "decay": 0, "other": 0},
        )
        if source not in breakdown:
            source = "other"
        breakdown[source] += amount

        rupture = self._get_power_amount(player, "Rupture")
        if rupture > 0:
            self._add_power(player, "Strength", rupture)

        self._apply_relics_on_player_hp_loss(state, amount, source)

    def _lose_hp_from_card(self, player: Dict[str, Any], state: Dict[str, Any], amount: int) -> None:
        self._lose_hp(player, state, amount, source="other")

    def _choose_best_card_to_topdeck_index(self, state: Dict[str, Any], exclude_card_id: Optional[str] = None) -> Optional[int]:
        candidates = [
            (i, c) for i, c in enumerate(state["hand"])
            if c.get("id") != exclude_card_id
        ]
        if not candidates:
            return None

        # préfère remettre une attaque chère / utile au-dessus
        candidates.sort(
            key=lambda x: (
                CARD_DEFS.get(x[1]["id"], {}).get("type") != "ATTACK",
                x[1].get("cost", 0),
            )
        )
        return candidates[-1][0]

    def _choose_best_card_to_exhaust_index(self, state: Dict[str, Any], exclude_card_id: Optional[str] = None) -> Optional[int]:
        bad_ids = {
            "Wound", "Dazed", "Burn", "Slimed", "Void",
            "AscendersBane", "Clumsy", "Curse of the Bell", "Decay", "Doubt",
            "Injury", "Normality", "Pain", "Parasite", "Pride", "Regret",
            "Shame", "Writhe",
        }

        candidates = [
            (i, c) for i, c in enumerate(state["hand"])
            if c.get("id") != exclude_card_id
        ]
        if not candidates:
            return None

        # priorité aux status/curse, sinon aux Defend
        candidates.sort(
            key=lambda x: (
                x[1].get("id") not in bad_ids,
                x[1].get("id") != "Defend_R",
                x[1].get("cost", 0),
            )
        )
        return candidates[0][0]

    def _choose_best_attack_or_power_from_hand(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = [
            c for c in state["hand"]
            if CARD_DEFS.get(c["id"], {}).get("type") in {"ATTACK", "POWER"}
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda c: (
                CARD_DEFS.get(c["id"], {}).get("type") != "POWER",
                c.get("cost", 0),
            )
        )
        return copy.deepcopy(candidates[-1])

    def _put_random_hand_card_on_top_of_draw(self, state: Dict[str, Any], exclude_card_id: Optional[str] = None) -> None:
        idx = self._choose_best_card_to_topdeck_index(state, exclude_card_id=exclude_card_id)
        if idx is None:
            return
        state["draw_pile"].insert(0, state["hand"].pop(idx))

    def _move_random_discard_to_top_draw(self, state: Dict[str, Any]) -> None:
        if not state["discard_pile"]:
            return
        idx = self.rng.randrange(len(state["discard_pile"]))
        card = state["discard_pile"].pop(idx)
        state["draw_pile"].insert(0, card)

    def _play_top_draw_and_exhaust(self, state: Dict[str, Any]) -> None:
        if not state["draw_pile"]:
            return
        card = state["draw_pile"].pop(0)
        tmp_hand_index = len(state["hand"])
        state["hand"].append(card)
        fake_target = self._first_alive_monster_index(state)
        if CARD_DEFS[card["id"]].get("targeted", False) and fake_target is None:
            return
        command = CombatCommand(command_type="play_card", hand_index=tmp_hand_index, target_index=fake_target)
        self._apply_play_card(state, command)
        # make sure card ends exhausted
        for pile in ["discard_pile"]:
            for i, c in enumerate(state[pile]):
                if c is card:
                    state[pile].pop(i)
                    self._exhaust_card_object(state, c)
                    return

    def _copy_attack_or_power_from_hand(self, state: Dict[str, Any], n: int) -> None:
        picked = self._choose_best_attack_or_power_from_hand(state)
        if picked is None:
            return
        for _ in range(n):
            state["hand"].append(copy.deepcopy(picked))

    def _move_random_exhaust_to_hand(self, state: Dict[str, Any]) -> None:
        if not state["exhaust_pile"]:
            return
        idx = self.rng.randrange(len(state["exhaust_pile"]))
        state["hand"].append(state["exhaust_pile"].pop(idx))

    def _exhaust_random_other_card_from_hand(self, state: Dict[str, Any], exclude_card_id: Optional[str] = None) -> None:
        idx = self._choose_best_card_to_exhaust_index(state, exclude_card_id=exclude_card_id)
        if idx is None:
            return
        self._move_card_from_hand_to_exhaust(state, idx)

    def _exhaust_all_non_attacks_from_hand(self, state: Dict[str, Any]) -> int:
        kept = []
        exhausted = 0
        for card in state["hand"]:
            if CARD_DEFS.get(card["id"], {}).get("type") == "ATTACK":
                kept.append(card)
            else:
                self._exhaust_card_object(state, card)
                exhausted += 1
        state["hand"] = kept
        return exhausted

    def _exhaust_all_cards_except_index(self, state: Dict[str, Any], keep_index: Optional[int]) -> int:
        kept = []
        exhausted = 0
        for i, card in enumerate(state["hand"]):
            if keep_index is not None and i == keep_index:
                kept.append(card)
            else:
                self._exhaust_card_object(state, card)
                exhausted += 1
        state["hand"] = kept
        return exhausted

    def _on_card_played_before_resolution(self, state: Dict[str, Any], card: Dict[str, Any]) -> None:
        meta = state["combat_meta"]
        card_type = CARD_DEFS[card["id"]]["type"]

        meta["cards_played_this_turn"] = int(meta.get("cards_played_this_turn", 0)) + 1

        if card_type == "ATTACK":
            meta["attacks_played_this_turn"] = int(meta.get("attacks_played_this_turn", 0)) + 1
            rage = self._get_power_amount(state["player"], "Rage")
            if rage > 0:
                self._gain_block(state, rage)

        if card_type == "SKILL":
            for monster in state.get("monsters", []):
                if self._monster_dead(monster):
                    continue
                if monster.get("name") == "Gremlin Nob":
                    enrage = self._get_power_amount(monster, "Enrage")
                    if enrage > 0:
                        self._add_power(monster, "Strength", enrage)

        if self._get_power_amount(state["player"], "Corruption") > 0 and card_type == "SKILL":
            card["exhaust"] = True
            card["cost"] = 0

    def _card_cost_for_state(self, card: Dict[str, Any], state: Dict[str, Any]) -> int:
        effective_def = self._get_effective_card_def(card)

        cost = card.get("cost", None)
        if cost is None and effective_def is not None:
            cost = effective_def.get("cost", 0)

        if effective_def is not None and card.get("upgraded", False):
            base_cost = effective_def.get("cost", cost)
            if card.get("cost", base_cost) == CARD_DEFS.get(card["id"], {}).get("cost"):
                cost = base_cost

        cost = int(cost)

        if cost < 0:
            return int(state.get("energy", 0))

        return cost

    def _all_other_cards_are_attack(self, hand: List[Dict[str, Any]], exclude_index: int) -> bool:
        for i, c in enumerate(hand):
            if i == exclude_index:
                continue
            if CARD_DEFS.get(c["id"], {}).get("type") != "ATTACK":
                return False
        return True

    def _choose_random_alive_monster(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        alive = [m for m in state["monsters"] if not self._monster_dead(m)]
        return self.rng.choice(alive) if alive else None

    def _clear_enemy_intent_fields(self, monster: Dict[str, Any]) -> None:
        for key in [
            "intent",
            "intent_base_damage",
            "intent_block",
            "intent_strength_gain",
            "intent_apply_weak",
            "intent_apply_vulnerable",
            "intent_apply_frail",
            "intent_add_status",
            "intent_entangle",
            "intent_escape",
        ]:
            if key in monster:
                del monster[key]

    def _set_enemy_intent(
            self,
            monster: Dict[str, Any],
            *,
            intent: str,
            base_damage: int = 0,
            block: int = 0,
            strength_gain: int = 0,
            apply_weak: int = 0,
            apply_vulnerable: int = 0,
            apply_frail: int = 0,
            add_status: Optional[Tuple[str, int]] = None,
            entangle: bool = False,
    ) -> None:
        self._clear_enemy_intent_fields(monster)
        monster["intent"] = intent
        monster["intent_base_damage"] = int(base_damage)

        if block:
            monster["intent_block"] = int(block)
        if strength_gain:
            monster["intent_strength_gain"] = int(strength_gain)
        if apply_weak:
            monster["intent_apply_weak"] = int(apply_weak)
        if apply_vulnerable:
            monster["intent_apply_vulnerable"] = int(apply_vulnerable)
        if apply_frail:
            monster["intent_apply_frail"] = int(apply_frail)
        if add_status is not None:
            monster["intent_add_status"] = add_status
        if entangle:
            monster["intent_entangle"] = True

    def _record_enemy_move(self, monster: Dict[str, Any], move: Optional[str]) -> None:
        if not move:
            return
        monster["last_move"] = move
        history = monster.setdefault("move_history", [])
        history.append(move)
        if len(history) > 3:
            history.pop(0)

    def _update_single_enemy_intent(self, monster: Dict[str, Any], state: Dict[str, Any]) -> None:
        name = monster.get("name", "")
        strength = self._get_power_amount(monster, "Strength")
        prev = monster.get("last_move")
        hist = monster.get("move_history", [])
        cstate = monster.setdefault("combat_state", {})

        if name == "Jaw Worm":
            if self.turn == 1:
                move = "CHOMP"
            else:
                rolls: List[str] = []
                if prev != "BELLOW":
                    rolls += ["BELLOW"] * 45
                if not (len(hist) >= 2 and hist[-1] == "THRASH" and hist[-2] == "THRASH"):
                    rolls += ["THRASH"] * 30
                if prev != "CHOMP":
                    rolls += ["CHOMP"] * 25
                move = self.rng.choice(rolls) if rolls else "CHOMP"

            if move == "CHOMP":
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=11 + strength)
            elif move == "THRASH":
                self._set_enemy_intent(monster, intent="ATTACK_DEFEND", base_damage=7 + strength, block=5)
            else:
                self._set_enemy_intent(monster, intent="BUFF_DEFEND", block=6, strength_gain=3)
            self._record_enemy_move(monster, move)
            return

        if name == "Cultist":
            if not cstate.get("opened", False):
                self._set_enemy_intent(monster, intent="BUFF")
                cstate["opened"] = True
                self._record_enemy_move(monster, "INCANTATION")
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=6 + strength)
                self._record_enemy_move(monster, "DARK_STRIKE")
            return

        if name == "Red Louse":
            if self.rng.random() < 0.25:
                self._set_enemy_intent(monster, intent="BUFF", strength_gain=3)
                self._record_enemy_move(monster, "GROW")
            else:
                dmg = int(cstate.get("bite_damage_roll", 6)) + strength
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=dmg)
                self._record_enemy_move(monster, "BITE")
            return

        if name == "Green Louse":
            if self.rng.random() < 0.25:
                self._set_enemy_intent(monster, intent="DEBUFF", apply_weak=2)
                self._record_enemy_move(monster, "SPIT_WEB")
            else:
                dmg = int(cstate.get("bite_damage_roll", 7)) + strength
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=dmg)
                self._record_enemy_move(monster, "BITE")
            return

        if name == "Fungi Beast":
            if self.rng.random() < 0.40:
                self._set_enemy_intent(monster, intent="BUFF", strength_gain=4)
                self._record_enemy_move(monster, "GROW")
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=6 + strength)
                self._record_enemy_move(monster, "BITE")
            return

        if name == "Acid Slime (S)":
            if self.rng.random() < 0.5:
                self._set_enemy_intent(monster, intent="DEBUFF", apply_weak=1)
                self._record_enemy_move(monster, "LICK")
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=3 + strength)
                self._record_enemy_move(monster, "TACKLE")
            return

        if name == "Acid Slime (M)":
            move = self.rng.choice(["SPIT", "TACKLE", "LICK"])
            if move == "SPIT":
                self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=7 + strength,
                                       add_status=("Slimed", 1))
            elif move == "TACKLE":
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=10 + strength)
            else:
                self._set_enemy_intent(monster, intent="DEBUFF", apply_weak=1)
            self._record_enemy_move(monster, move)
            return

        if name == "Spike Slime (S)":
            if self.rng.random() < 0.4:
                self._set_enemy_intent(monster, intent="DEBUFF", apply_frail=1)
                self._record_enemy_move(monster, "LICK")
            else:
                self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=5 + strength,
                                       add_status=("Slimed", 1))
                self._record_enemy_move(monster, "FLAME_TACKLE")
            return

        if name == "Spike Slime (M)":
            if self.rng.random() < 0.4:
                self._set_enemy_intent(monster, intent="DEBUFF", apply_frail=1)
                self._record_enemy_move(monster, "LICK")
            else:
                self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=8 + strength,
                                       add_status=("Slimed", 1))
                self._record_enemy_move(monster, "FLAME_TACKLE")
            return

        if name == "Blue Slaver":
            move = "RAKE" if self.rng.random() < 0.4 else "STAB"
            if move == "RAKE":
                self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=8 + strength, apply_weak=2)
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=12 + strength)
            self._record_enemy_move(monster, move)
            return

        if name == "Red Slaver":
            turns_taken = int(cstate.get("turns_taken", 0))
            used_entangle = bool(cstate.get("used_entangle", False))

            if turns_taken == 0:
                move = "STAB"
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=13 + strength)
            elif not used_entangle and turns_taken >= 2 and self.rng.random() < 0.30:
                move = "ENTANGLE"
                self._set_enemy_intent(monster, intent="DEBUFF", entangle=True)
                cstate["used_entangle"] = True
            else:
                move = "SCRAPE" if self.rng.random() < 0.40 else "STAB"
                if move == "SCRAPE":
                    self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=8 + strength,
                                           apply_vulnerable=1)
                else:
                    self._set_enemy_intent(monster, intent="ATTACK", base_damage=13 + strength)

            cstate["turns_taken"] = turns_taken + 1
            self._record_enemy_move(monster, move)
            return

        if name == "Looter":
            thief_turn = int(cstate.get("thief_turn", 0))
            smoked = bool(cstate.get("smoked", False))

            if smoked:
                self._set_enemy_intent(monster, intent="ESCAPE")
                self._record_enemy_move(monster, "ESCAPE")
                return

            if thief_turn < 2:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=11 + strength)
                self._record_enemy_move(monster, "MUG")
            elif thief_turn == 2:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=14 + strength)
                self._record_enemy_move(monster, "LUNGE")
            else:
                self._set_enemy_intent(monster, intent="DEFEND", block=6)
                self._record_enemy_move(monster, "SMOKE_BOMB")
                cstate["smoked"] = True

            cstate["thief_turn"] = thief_turn + 1
            return

        if name == "Mad Gremlin":
            self._set_enemy_intent(monster, intent="ATTACK", base_damage=4 + strength)
            self._record_enemy_move(monster, "SCRATCH")
            return

        if name == "Sneaky Gremlin":
            self._set_enemy_intent(monster, intent="ATTACK", base_damage=9 + strength)
            self._record_enemy_move(monster, "PUNCTURE")
            return

        if name == "Fat Gremlin":
            self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=4 + strength, apply_weak=1)
            self._record_enemy_move(monster, "SMASH")
            return

        if name == "Shield Gremlin":
            ally = self._choose_lowest_hp_ally(state, exclude_dead=True, exclude_monster=monster)
            if ally is not None:
                self._set_enemy_intent(monster, intent="DEFEND_ALLY", block=7)
                self._record_enemy_move(monster, "PROTECT")
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=6 + strength)
                self._record_enemy_move(monster, "SHIELD_BASH")
            return

        if name == "Gremlin Wizard":
            charge_turns = int(cstate.get("charge_turns", 0))
            if charge_turns < 2:
                self._set_enemy_intent(monster, intent="CHARGE")
                cstate["charge_turns"] = charge_turns + 1
                self._record_enemy_move(monster, "CHARGING")
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=25 + strength)
                cstate["charge_turns"] = 0
                self._record_enemy_move(monster, "ULTIMATE_BLAST")
            return

        if name == "Gremlin Nob":
            opened = bool(cstate.get("opened", False))
            last_move = cstate.get("last_move", None)

            if not opened:
                self._set_enemy_intent(monster, intent="BUFF", strength_gain=2)
                cstate["opened"] = True
                self._record_enemy_move(monster, "BELLOW")
                return

            if last_move == "SKULL_BASH":
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=16 + strength)
                self._record_enemy_move(monster, "RUSH")
            else:
                self._set_enemy_intent(monster, intent="ATTACK_DEBUFF", base_damage=8 + strength, apply_vulnerable=2)
                self._record_enemy_move(monster, "SKULL_BASH")
            return

        if name == "Lagavulin":
            asleep_turns = int(cstate.get("asleep_turns", 0))
            awakened = bool(cstate.get("awakened", False))
            pattern_idx = int(cstate.get("pattern_idx", 0))

            if not awakened:
                if asleep_turns >= 2:
                    cstate["awakened"] = True
                    self._add_power(monster, "Asleep", -999)
                    self._add_power(monster, "Metallicize", -999)
                    self._set_enemy_intent(monster, intent="ATTACK", base_damage=18 + strength)
                    cstate["pattern_idx"] = 1
                    self._record_enemy_move(monster, "AWAKE_SMASH")
                else:
                    self._set_enemy_intent(monster, intent="CHARGE")
                    cstate["asleep_turns"] = asleep_turns + 1
                    self._record_enemy_move(monster, "SLEEP")
                return

            if pattern_idx in (0, 1):
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=18 + strength)
                cstate["pattern_idx"] = pattern_idx + 1
                self._record_enemy_move(monster, "HEAVY_SMASH")
            else:
                self._set_enemy_intent(monster, intent="DEBUFF", apply_weak=1, apply_frail=1)
                cstate["pattern_idx"] = 0
                self._record_enemy_move(monster, "SIPHON_SOUL")
            return

        if name == "Sentry":
            pattern_idx = int(cstate.get("pattern_idx", 0))

            if pattern_idx % 2 == 0:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=9 + strength)
                self._record_enemy_move(monster, "BEAM")
            else:
                self._set_enemy_intent(monster, intent="DEBUFF", add_status=("Dazed", 2))
                self._record_enemy_move(monster, "DAZED")
            cstate["pattern_idx"] = pattern_idx + 1
            return

        if name == "The Guardian":
            pattern_idx = int(cstate.get("pattern_idx", 0))
            defensive_mode = bool(cstate.get("defensive_mode", False))

            if defensive_mode:
                if pattern_idx % 2 == 0:
                    self._set_enemy_intent(monster, intent="DEFEND", block=20)
                    self._record_enemy_move(monster, "DEFENSIVE_BLOCK")
                else:
                    self._set_enemy_intent(monster, intent="ATTACK", base_damage=8 + strength)
                    self._record_enemy_move(monster, "ROLL_ATTACK")
                cstate["pattern_idx"] = pattern_idx + 1
            else:
                phase = pattern_idx % 3
                if phase == 0:
                    self._set_enemy_intent(monster, intent="ATTACK", base_damage=9 + strength)
                    self._record_enemy_move(monster, "CHARGING_UP")
                elif phase == 1:
                    self._set_enemy_intent(monster, intent="ATTACK", base_damage=32 + strength)
                    self._record_enemy_move(monster, "FIERCE_BASH")
                else:
                    self._set_enemy_intent(monster, intent="DEFEND", block=12)
                    self._record_enemy_move(monster, "VENT_STEAM")
                cstate["pattern_idx"] = pattern_idx + 1
            return

        if name == "Slime Boss":
            pattern_idx = int(cstate.get("pattern_idx", 0))
            if cstate.get("split_done", False):
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=16 + strength)
                self._record_enemy_move(monster, "POST_SPLIT_SLAM")
            else:
                if pattern_idx % 2 == 0:
                    self._set_enemy_intent(monster, intent="ATTACK", base_damage=35 + strength)
                    self._record_enemy_move(monster, "GOOP_SPRAY")
                else:
                    self._set_enemy_intent(monster, intent="BUFF")
                    self._record_enemy_move(monster, "PREPARING")
                cstate["pattern_idx"] = pattern_idx + 1
            return

        if name == "Hexaghost":
            pattern_idx = int(cstate.get("pattern_idx", 0))
            first_turn_done = bool(cstate.get("first_turn_done", False))

            if not first_turn_done:
                missing_hp = max(0, int(state["player"]["max_hp"]) - int(state["player"]["current_hp"]))
                per_hit = max(1, min(6, missing_hp // 12 + 1))
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=per_hit)
                monster["intent_hits"] = 6
                cstate["first_turn_done"] = True
                self._record_enemy_move(monster, "INFERNAL")
                return

            phase = pattern_idx % 3
            if phase == 0:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=6 + strength)
                monster["intent_hits"] = 2
                self._record_enemy_move(monster, "SEAR")
            elif phase == 1:
                self._set_enemy_intent(monster, intent="DEBUFF", add_status=("Burn", 2))
                self._record_enemy_move(monster, "SEAR_BURN")
            else:
                self._set_enemy_intent(monster, intent="ATTACK", base_damage=12 + strength)
                self._record_enemy_move(monster, "TACKLE")
            cstate["pattern_idx"] = pattern_idx + 1
            return

        self._set_enemy_intent(monster, intent="ATTACK", base_damage=5 + strength)
        self._record_enemy_move(monster, "DEFAULT_ATTACK")

    def _update_enemy_intents(self, state: Dict[str, Any]) -> None:
        for monster in state["monsters"]:
            if self._monster_dead(monster):
                continue
            self._update_single_enemy_intent(monster, state)

    def _has_card_in_hand_or_drawn(self, state: Dict[str, Any], card_id: str) -> bool:
        for card in state.get("hand", []):
            if card.get("id") == card_id:
                return True
        return False

    def _refresh_terminal_flags(self, state: Dict[str, Any]) -> None:
        player = state["player"]
        player_dead = int(player.get("current_hp", 0)) <= 0

        if player_dead and self._player_has_relic(state, "Lizard Tail"):
            runtime = self._combat_relic_runtime(state)

            if not runtime.get("lizard_tail_used", False):
                runtime["lizard_tail_used"] = True
                player["current_hp"] = max(1, int(player.get("max_hp", 0) * 0.5))

                player_dead = False

        all_dead = all(self._monster_dead(m) for m in state["monsters"])
        state["combat_over"] = bool(player_dead or all_dead)
        state["game_over"] = bool(player_dead)

    def _monster_dead(self, monster: Dict[str, Any]) -> bool:
        return bool(monster.get("isDead", False)) or int(monster.get("current_hp", 0)) <= 0

    def _add_power(self, entity: Dict[str, Any], power_name: str, amount: int) -> None:
        powers = entity.setdefault("powers", [])

        for i, p in enumerate(powers):
            if p.get("id") == power_name:
                new_amount = int(p.get("amount", 0)) + int(amount)
                if new_amount <= 0:
                    powers.pop(i)
                else:
                    p["amount"] = new_amount
                return

        if int(amount) > 0:
            powers.append({"id": power_name, "amount": int(amount)})

    def _set_power(self, entity: Dict[str, Any], power_name: str, value: int) -> None:
        powers = entity.setdefault("powers", [])
        for p in powers:
            if p.get("id") == power_name:
                p["amount"] = int(value)
                return
        powers.append({"id": power_name, "amount": int(value)})

    def _get_power_amount(self, entity: Dict[str, Any], power_name: str) -> int:
        for p in entity.get("powers", []):
            if p.get("id") == power_name:
                return int(p.get("amount", 0))
        return 0


class LiveSpireBackend:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_default_config()
        self._last_state: Optional[Dict[str, Any]] = None

    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError("LiveSpireBackend not wired to CommunicationMod yet.")

    def get_state(self) -> Dict[str, Any]:
        if self._last_state is None:
            raise RuntimeError("No live state available yet.")
        return copy.deepcopy(self._last_state)

    def step(self, command: CombatCommand) -> Tuple[Dict[str, Any], bool]:
        raise NotImplementedError("LiveSpireBackend.step() not wired yet.")


class STSEnv:
    def __init__(self, cfg: Optional[Config] = None, mode: str = "mock", seed: int = 42):
        self.cfg = cfg or get_default_config()
        self.mode = mode
        self.seed = seed
        self.reward_calculator = CombatRewardCalculator(self.cfg)

        if mode == "mock":
            self.backend = MockGameBackend(self.cfg, seed=seed)
        elif mode == "live":
            self.backend = LiveSpireBackend(self.cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.state: Optional[Dict[str, Any]] = None
        self.turn: int = 0

    def reset(self) -> Dict[str, Any]:
        self.state = self.backend.reset()
        self.turn = int(self.state.get("turn", 1))
        return copy.deepcopy(self.state)

    def set_training_progress(self, update_idx: int) -> None:
        if hasattr(self.backend, "set_training_progress"):
            self.backend.set_training_progress(update_idx)

    def current_state(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return copy.deepcopy(self.state)

    def step(self, action_index: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        command = self.decode_action_index(action_index, self.state)
        return self.step_command(command, forced_action_index=action_index)

    def step_command(self, command: CombatCommand, forced_action_index: Optional[int] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        prev_state = copy.deepcopy(self.state)
        next_state, illegal_action = self.backend.step(command)

        played_card_name = ""
        played_card = None

        if command.command_type == "play_card" and command.hand_index is not None:
            hand = prev_state.get("hand", [])
            if isinstance(hand, list) and 0 <= command.hand_index < len(hand):
                candidate = hand[command.hand_index]
                if isinstance(candidate, dict):
                    played_card = copy.deepcopy(candidate)
                    played_card_name = str(
                        candidate.get(
                            "id",
                            candidate.get(
                                "card_id",
                                candidate.get("name", "")
                            )
                        )
                    )

        reward_out: CombatRewardOutput = self.reward_calculator.compute(
            prev_state=prev_state,
            next_state=next_state,
            action_info={
                "illegal_action": illegal_action,
                "command_type": command.command_type,
                "action_type": command.command_type,
                "played_card_name": played_card_name,
                "played_card": played_card,
                "hand_index": command.hand_index,
                "target_index": command.target_index,
                "potion_index": command.potion_index,
            },
        )

        self.state = copy.deepcopy(next_state)
        self.turn = int(self.state.get("turn", self.turn))
        done = reward_out.combat_won or reward_out.combat_lost
        action_index = forced_action_index if forced_action_index is not None else encode_command_to_action_index(command, self.cfg)

        info = StepInfo(
            action_index=action_index,
            command=command.to_dict(),
            reward_breakdown=reward_out.to_dict(),
            illegal_action=illegal_action,
            combat_won=reward_out.combat_won,
            combat_lost=reward_out.combat_lost,
            turn=self.turn,
        ).to_dict()

        info["hp_loss_breakdown"] = copy.deepcopy(self.state.get("hp_loss_breakdown", {}))

        return copy.deepcopy(self.state), reward_out.total_reward, done, info

    def decode_action_index(self, action_index: int, state: Dict[str, Any]) -> CombatCommand:
        max_hand_cards = self.cfg.combat_obs.max_hand_cards
        max_enemies = self.cfg.combat_obs.max_enemies
        max_potions = self.cfg.combat_obs.max_potions

        targeted_base = max_hand_cards
        targeted_size = max_hand_cards * max_enemies
        end_turn_idx = targeted_base + targeted_size
        potion_base = end_turn_idx + 1
        potion_target_base = potion_base + max_potions
        potion_target_size = max_potions * max_enemies

        max_choose_hand = self.cfg.combat_action.max_choose_hand_actions
        max_choose_option = self.cfg.combat_action.max_choose_option_actions
        max_choose_discard = self.cfg.combat_action.max_choose_discard_actions
        max_choose_exhaust = self.cfg.combat_action.max_choose_exhaust_actions

        choose_hand_base = potion_target_base + potion_target_size
        choose_option_base = choose_hand_base + max_choose_hand
        choose_discard_base = choose_option_base + max_choose_option
        choose_exhaust_base = choose_discard_base + max_choose_discard

        if 0 <= action_index < max_hand_cards:
            return CombatCommand(command_type="play_card", hand_index=action_index, target_index=None)

        if targeted_base <= action_index < targeted_base + targeted_size:
            relative = action_index - targeted_base
            hand_index = relative // max_enemies
            target_index = relative % max_enemies
            return CombatCommand(command_type="play_card", hand_index=hand_index, target_index=target_index)

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
            potion_index = relative // max_enemies
            target_index = relative % max_enemies
            return CombatCommand(
                command_type="use_potion",
                potion_index=potion_index,
                target_index=target_index,
            )

        if choose_hand_base <= action_index < choose_option_base:
            return CombatCommand(
                command_type="choose_hand_card",
                hand_index=action_index - choose_hand_base,
            )

        if choose_option_base <= action_index < choose_discard_base:
            return CombatCommand(
                command_type="choose_option",
                target_index=action_index - choose_option_base,
            )

        if choose_discard_base <= action_index < choose_exhaust_base:
            return CombatCommand(
                command_type="choose_discard_target",
                hand_index=action_index - choose_discard_base,
            )

        if choose_exhaust_base <= action_index < choose_exhaust_base + max_choose_exhaust:
            return CombatCommand(
                command_type="choose_exhaust_target",
                hand_index=action_index - choose_exhaust_base,
            )

        return CombatCommand(command_type="wait")

    def render_text(self) -> None:
        if self.state is None:
            print("No state.")
            return

        from typing import Dict, Any
        s: Dict[str, Any] = self.state
        p: Dict[str, Any] = s.get("player", {})

        print("=" * 60)
        print(
            f"TURN {s.get('turn', '?')} | "
            f"HP {p.get('current_hp', '?')}/{p.get('max_hp', '?')} | "
            f"Block {p.get('block', 0)} | "
            f"Energy {s.get('energy', 0)}"
        )

        print("\nPOWERS:", p.get("powers", []))
        print("RELICS:", [r.get("name", r.get("id", "?")) for r in p.get("relics", [])])

        print("\nHAND:")
        for i, card in enumerate(s.get("hand", [])):
            print(f"  [{i}] {card.get('id')} (cost={card.get('cost')}, up={card.get('upgraded', False)})")

        print("\nENEMIES:")
        for i, m in enumerate(s.get("monsters", [])):
            status = "DEAD" if int(m.get("current_hp", 0)) <= 0 or m.get("isDead", False) else "ALIVE"
            print(
                f"  [{i}] {m.get('name')} | "
                f"HP {m.get('current_hp')}/{m.get('max_hp')} | "
                f"Block {m.get('block', 0)} | "
                f"Intent {m.get('intent')} {m.get('intent_base_damage', '')} | "
                f"Powers {m.get('powers', [])} | {status}"
            )
        print("=" * 60)

from __future__ import annotations

import math
import random

from slay_rl.features.combat_encoder import CombatEncoder
from slay_rl.rewards.combat_reward import CombatRewardCalculator


CARD_POOL = [
    "Strike_R",
    "Defend_R",
    "Bash",
    "Inflame",
    "Pommel Strike",
    "Twin Strike",
    "Shrug It Off",
    "Wound",
    "Burn",
]

POTION_POOL = [
    {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
    {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
    {"name": "Block Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Common"},
    {"name": "Strength Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Common"},
]


def _random_card(make_state, rng: random.Random):
    return make_state.card(rng.choice(CARD_POOL))


def _random_enemy(make_state, rng: random.Random, idx: int):
    hp = rng.randint(0, 45)
    intent = rng.choice(["ATTACK", "BUFF", "DEFEND", "DEBUFF"])
    intent_base_damage = rng.choice([0, 0, 6, 8, 10, 12])
    enemy = make_state.enemy(
        name=f"Enemy{idx}",
        hp=hp,
        block=rng.randint(0, 12),
        intent=intent,
        intent_base_damage=intent_base_damage,
    )
    if hp <= 0:
        enemy["isDead"] = True
    return enemy


def _random_potions(rng: random.Random):
    potions = []
    usable_count = rng.randint(0, 2)
    for i in range(5):
        if i < usable_count:
            potions.append(dict(rng.choice(POTION_POOL)))
        else:
            potions.append(
                {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False}
            )
    return potions


def test_fuzz_action_mask_never_targets_dead_or_missing_enemies(make_state):
    rng = random.Random(1337)
    encoder = CombatEncoder()

    max_hand = encoder.max_hand_cards
    max_enemies = encoder.max_enemies
    targeted_base = max_hand

    for _ in range(80):
        hand_size = rng.randint(0, 5)
        enemy_count = rng.randint(1, 4)

        state = make_state(
            hand=[_random_card(make_state, rng) for _ in range(hand_size)],
            monsters=[_random_enemy(make_state, rng, i) for i in range(enemy_count)],
            energy=rng.randint(0, 4),
            potions=_random_potions(rng),
        )

        encoded = encoder.encode(state)
        mask = encoded["valid_action_mask"].tolist()
        enemy_mask = encoded["enemy_mask"].tolist()

        for hand_idx in range(max_hand):
            for target_idx in range(max_enemies):
                action_idx = targeted_base + hand_idx * max_enemies + target_idx

                if hand_idx >= hand_size or target_idx >= enemy_count or enemy_mask[target_idx] == 0.0:
                    assert mask[action_idx] == 0.0


def test_fuzz_non_target_play_slots_outside_hand_are_always_zero(make_state):
    rng = random.Random(2026)
    encoder = CombatEncoder()

    for _ in range(80):
        hand_size = rng.randint(0, 5)

        state = make_state(
            hand=[_random_card(make_state, rng) for _ in range(hand_size)],
            monsters=[_random_enemy(make_state, rng, 0)],
            energy=rng.randint(0, 4),
        )

        encoded = encoder.encode(state)
        mask = encoded["valid_action_mask"].tolist()

        for hand_idx in range(hand_size, encoder.max_hand_cards):
            assert mask[hand_idx] == 0.0


def test_fuzz_targeted_potion_actions_only_exist_for_living_targets(make_state):
    rng = random.Random(7)
    encoder = CombatEncoder()

    end_turn_idx = encoder.max_hand_cards + encoder.max_hand_cards * encoder.max_enemies
    potion_base = end_turn_idx + 1
    potion_target_base = potion_base + encoder.max_potions

    for _ in range(60):
        enemy_count = rng.randint(1, 4)
        enemies = [_random_enemy(make_state, rng, i) for i in range(enemy_count)]
        potions = [
            {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Block Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Common"},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ]

        state = make_state(
            hand=[],
            monsters=enemies,
            potions=potions,
        )

        encoded = encoder.encode(state)
        mask = encoded["valid_action_mask"].tolist()
        enemy_mask = encoded["enemy_mask"].tolist()

        for potion_idx in (0, 1):
            assert mask[potion_base + potion_idx] == 0.0
            for target_idx in range(encoder.max_enemies):
                action_idx = potion_target_base + potion_idx * encoder.max_enemies + target_idx
                if target_idx >= enemy_count or enemy_mask[target_idx] == 0.0:
                    assert mask[action_idx] == 0.0


def test_fuzz_reward_compute_stays_finite_on_random_transitions(make_state):
    rng = random.Random(99)
    calc = CombatRewardCalculator()

    for _ in range(100):
        prev_enemy_hp = rng.randint(1, 40)
        next_enemy_hp = rng.randint(0, prev_enemy_hp)

        prev_state = make_state(
            hand=[_random_card(make_state, rng) for _ in range(rng.randint(0, 4))],
            monsters=[make_state.enemy(hp=prev_enemy_hp, block=rng.randint(0, 8), intent="ATTACK", intent_base_damage=rng.randint(0, 14))],
            hp=rng.randint(10, 80),
            max_hp=80,
            block=rng.randint(0, 20),
            energy=rng.randint(0, 4),
            player_powers=[],
            potions=_random_potions(rng),
            combat_meta={
                "cards_played_this_turn": rng.randint(0, 5),
                "attacks_played_this_turn": rng.randint(0, 5),
                "double_tap_charges": rng.randint(0, 1),
            },
        )

        next_enemy = make_state.enemy(
            hp=next_enemy_hp,
            block=rng.randint(0, 8),
            intent="ATTACK",
            intent_base_damage=rng.randint(0, 14),
        )
        if next_enemy_hp == 0:
            next_enemy["isDead"] = True

        next_state = make_state(
            hand=[_random_card(make_state, rng) for _ in range(rng.randint(0, 4))],
            monsters=[next_enemy],
            hp=rng.randint(0, prev_state["player"]["current_hp"] if "player" in prev_state else 80),
            max_hp=80,
            block=rng.randint(0, 20),
            energy=rng.randint(0, 4),
            player_powers=[],
            potions=_random_potions(rng),
            combat_meta={
                "cards_played_this_turn": rng.randint(0, 5),
                "attacks_played_this_turn": rng.randint(0, 5),
                "double_tap_charges": rng.randint(0, 1),
            },
        )

        if next_enemy_hp == 0 and rng.random() < 0.5:
            next_state["combat_over"] = True
            next_state["victory"] = True

        out = calc.compute(
            prev_state,
            next_state,
            action_info={
                "command_type": rng.choice(["play_card", "end_turn", "use_potion"]),
                "card_name": rng.choice(["Strike_R", "Defend_R", "Inflame", "Bash"]),
                "illegal_action": rng.choice([False, False, False, True]),
            },
        )

        assert math.isfinite(out.total_reward)
        assert math.isfinite(out.damage_dealt_reward)
        assert math.isfinite(out.damage_taken_reward)
        assert math.isfinite(out.block_reward)
        assert math.isfinite(out.buff_reward)
        assert math.isfinite(out.debuff_reward)
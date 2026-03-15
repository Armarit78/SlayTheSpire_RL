from __future__ import annotations

import pytest

from slay_rl.features.combat_encoder import CombatEncoder


def _targeted_action_index(encoder: CombatEncoder, hand_index: int, target_index: int) -> int:
    base = encoder.max_hand_cards
    return base + hand_index * encoder.max_enemies + target_index


def test_live_style_escaped_enemy_is_not_targetable(make_state):
    encoder = CombatEncoder()

    enemy = make_state.enemy(hp=30)
    enemy["escaped"] = True

    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[enemy],
        energy=2,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"]

    targeted_idx = _targeted_action_index(encoder, hand_index=0, target_index=0)

    assert encoded["enemy_mask"][0].item() == 0.0
    assert mask[targeted_idx].item() == 0.0


def test_live_style_dead_enemy_is_not_targetable(make_state):
    encoder = CombatEncoder()

    enemy = make_state.enemy(hp=0)
    enemy["isDead"] = True

    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[enemy],
        energy=2,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"]

    targeted_idx = _targeted_action_index(encoder, hand_index=0, target_index=0)

    assert encoded["enemy_mask"][0].item() == 0.0
    assert mask[targeted_idx].item() == 0.0


def test_live_style_potion_slot_is_not_usable(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
        potions=[
            {"name": "Potion Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Potion Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Potion Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Potion Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Potion Slot", "usable": False, "empty": True, "requires_target": False},
        ],
    )

    encoded = encoder.encode(state)

    assert encoded["potion_mask"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    max_hand = encoder.max_hand_cards
    max_enemies = encoder.max_enemies
    end_turn_idx = max_hand + max_hand * max_enemies
    potion_base = end_turn_idx + 1
    potion_target_base = potion_base + encoder.max_potions

    potion_non_target_mask = encoded["valid_action_mask"][potion_base:potion_target_base]
    potion_target_mask = encoded["valid_action_mask"][potion_target_base:]

    assert potion_non_target_mask.sum().item() == 0.0
    assert potion_target_mask.sum().item() == 0.0


def test_live_style_is_gone_enemy_should_not_be_targetable(make_state):
    encoder = CombatEncoder()

    enemy = make_state.enemy(hp=30)
    enemy["is_gone"] = True

    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[enemy],
        energy=2,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"]

    targeted_idx = _targeted_action_index(encoder, hand_index=0, target_index=0)

    assert encoded["enemy_mask"][0].item() == 0.0
    assert mask[targeted_idx].item() == 0.0


def test_live_style_half_dead_enemy_should_not_be_targetable(make_state):
    encoder = CombatEncoder()

    enemy = make_state.enemy(hp=30)
    enemy["half_dead"] = True

    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[enemy],
        energy=2,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"]

    targeted_idx = _targeted_action_index(encoder, hand_index=0, target_index=0)

    assert encoded["enemy_mask"][0].item() == 0.0
    assert mask[targeted_idx].item() == 0.0
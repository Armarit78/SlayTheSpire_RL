from __future__ import annotations

from slay_rl.features.combat_encoder import CombatEncoder
from slay_rl.agents.combat_agent import encode_command_to_action_index, CombatCommand



def test_action_mask_marks_playable_non_target_card_and_end_turn(make_state):
    encoder = CombatEncoder()
    state = make_state(
        hand=[make_state.card("Defend_R"), make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
        energy=1,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    end_turn_idx = 10 + 10 * 5
    assert mask[0] == 1.0
    assert mask[1] == 0.0
    assert mask[end_turn_idx] == 1.0



def test_action_mask_marks_targeted_card_only_for_living_targets(make_state):
    encoder = CombatEncoder()
    dead_enemy = make_state.enemy(hp=0)
    dead_enemy["isDead"] = True
    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40), dead_enemy],
        energy=3,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    bash_t0 = 10 + 0 * 5 + 0
    bash_t1 = 10 + 0 * 5 + 1
    assert mask[bash_t0] == 1.0
    assert mask[bash_t1] == 0.0



def test_action_mask_marks_potions_in_correct_slots(make_state):
    encoder = CombatEncoder()
    potions = [
        {"name": "Explosive Potion", "usable": True, "empty": False, "requires_target": False},
        {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(3)],
    ]
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=40), make_state.enemy(name="Cultist", hp=35)],
        potions=potions,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    end_turn_idx = 10 + 10 * 5
    potion_base = end_turn_idx + 1
    potion_target_base = potion_base + 5

    assert mask[potion_base + 0] == 1.0
    assert mask[potion_base + 1] == 0.0
    assert mask[potion_target_base + 1 * 5 + 0] == 1.0
    assert mask[potion_target_base + 1 * 5 + 1] == 1.0



def test_encode_command_to_action_index_round_trip_layout():
    non_target = encode_command_to_action_index(CombatCommand("play_card", hand_index=2, target_index=None))
    targeted = encode_command_to_action_index(CombatCommand("play_card", hand_index=3, target_index=4))
    end_turn = encode_command_to_action_index(CombatCommand("end_turn"))
    potion = encode_command_to_action_index(CombatCommand("use_potion", potion_index=1, target_index=None))
    targeted_potion = encode_command_to_action_index(CombatCommand("use_potion", potion_index=2, target_index=3))

    assert non_target == 2
    assert targeted == 10 + 3 * 5 + 4
    assert end_turn == 60
    assert potion == 62
    assert targeted_potion == 66 + 2 * 5 + 3

from __future__ import annotations

from slay_rl.agents.combat_agent import CombatCommand, encode_command_to_action_index


def test_choose_action_layout_constants():
    choose_hand_base = 66 + 5 * 5
    choose_option_base = choose_hand_base + 10
    choose_discard_base = choose_option_base + 5
    choose_exhaust_base = choose_discard_base + 10

    assert choose_hand_base == 91
    assert choose_option_base == 101
    assert choose_discard_base == 106
    assert choose_exhaust_base == 116


def test_choose_action_indices_round_trip():
    assert encode_command_to_action_index(CombatCommand("choose_hand_card", hand_index=0)) == 91
    assert encode_command_to_action_index(CombatCommand("choose_option", target_index=0)) == 101
    assert encode_command_to_action_index(CombatCommand("choose_discard_target", hand_index=0)) == 106
    assert encode_command_to_action_index(CombatCommand("choose_exhaust_target", hand_index=0)) == 116
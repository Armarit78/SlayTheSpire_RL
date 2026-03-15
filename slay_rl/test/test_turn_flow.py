from __future__ import annotations

import pytest


def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0



def test_end_turn_resets_draw_lock_and_turn_counters(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        combat_meta={
            "cards_played_this_turn": 3,
            "attacks_played_this_turn": 2,
            "cannot_draw_more_this_turn": True,
        },
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["combat_meta"]["cards_played_this_turn"] == 0
    assert next_state["combat_meta"]["attacks_played_this_turn"] == 0
    assert next_state["combat_meta"]["cannot_draw_more_this_turn"] is False



def test_block_is_cleared_without_barricade(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        block=12,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["player"]["block"] == 0



def test_block_is_kept_with_barricade(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        block=12,
        player_powers=[{"id": "Barricade", "amount": 1}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["player"]["block"] == 12



def test_flex_strength_is_removed_at_end_turn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Flex")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        energy=1,
    )
    step_helpers.set_state(state)

    after_flex, illegal = step_helpers.play_card(hand_index=0)
    assert illegal is False
    assert _power_amount(after_flex["player"], "Strength") == 2

    step_helpers.set_state(after_flex)
    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert _power_amount(next_state["player"], "Strength") == 0


def test_double_tap_duplicates_next_attack_only_once(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Double Tap"), make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=40)],
        energy=2,
    )
    step_helpers.set_state(state)

    after_dt, illegal = step_helpers.play_card(hand_index=0)
    assert illegal is False
    assert after_dt["combat_meta"]["double_tap_charges"] == 1

    step_helpers.set_state(after_dt)
    after_attack, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert after_attack["combat_meta"]["double_tap_charges"] == 0
    assert after_attack["monsters"][0]["current_hp"] == 28

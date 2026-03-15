from __future__ import annotations

import pytest

def test_whirlwind_spends_all_energy_and_hits_all_enemies_x_times(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Whirlwind")],
        monsters=[make_state.enemy(hp=40), make_state.enemy(name="Cultist", hp=35)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] == 0
    assert next_state["combat_meta"]["last_x_energy_spent"] == 3
    assert next_state["monsters"][0]["current_hp"] == 25
    assert next_state["monsters"][1]["current_hp"] == 20
    assert [c["id"] for c in next_state["discard_pile"]] == ["Whirlwind"]


def test_whirlwind_should_scale_with_strength_like_a_real_attack(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Whirlwind")],
        monsters=[make_state.enemy(hp=60), make_state.enemy(name="Cultist", hp=60)],
        energy=2,
        player_powers=[{"id": "Strength", "amount": 2}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 46
    assert next_state["monsters"][1]["current_hp"] == 46

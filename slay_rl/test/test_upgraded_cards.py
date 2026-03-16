from __future__ import annotations

import pytest


def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0


def test_bash_plus_applies_three_vulnerable(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Bash", upgraded=True)],
        monsters=[make_state.enemy(hp=40)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["energy"] == 1
    assert next_state["monsters"][0]["current_hp"] == 30
    assert _power_amount(next_state["monsters"][0], "Vulnerable") == 3


def test_pommel_strike_plus_draws_two_cards(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Pommel Strike", upgraded=True)],
        draw_pile=[
            make_state.card("Defend_R"),
            make_state.card("Strike_R"),
            make_state.card("Bash"),
        ],
        monsters=[make_state.enemy(hp=30)],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 20
    assert [c["id"] for c in next_state["hand"]] == ["Defend_R", "Strike_R"]


def test_true_grit_plus_exhausts_non_self_other_card(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("True Grit", upgraded=True),
            make_state.card("Wound"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy()],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["block"] == 9

    assert next_state["pending_choice"]["choice_type"] == "choose_exhaust_target"
    assert next_state["pending_choice"]["valid_hand_indices"] == [1, 2]

    assert next_state["exhaust_pile"] == []


def test_offering_plus_draws_five_cards(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Offering", upgraded=True)],
        draw_pile=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
            make_state.card("Pommel Strike"),
            make_state.card("Shrug It Off"),
        ],
        hp=50,
        energy=1,
        monsters=[make_state.enemy(hp=45)],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["current_hp"] == 44
    assert next_state["energy"] == 3
    assert len(next_state["hand"]) == 5


def test_feed_plus_grants_four_max_hp_on_kill(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Feed", upgraded=True)],
        monsters=[make_state.enemy(hp=12)],
        hp=60,
        max_hp=80,
        energy=2,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["isDead"] is True
    assert next_state["player"]["max_hp"] == 84
    assert next_state["player"]["current_hp"] == 70

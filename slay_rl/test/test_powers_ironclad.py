from __future__ import annotations


def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0


def test_feel_no_pain_gains_block_when_card_exhausts(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Second Wind"), make_state.card("Defend_R")],
        monsters=[make_state.enemy()],
        player_powers=[{"id": "Feel No Pain", "amount": 3}],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["block"] == 11
    assert sorted(c["id"] for c in next_state["exhaust_pile"]) == ["Defend_R", "Second Wind"]


def test_dark_embrace_draws_when_card_is_exhausted(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Ghostly Armor")],
        draw_pile=[make_state.card("Strike_R"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        player_powers=[{"id": "Dark Embrace", "amount": 1}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Ghostly Armor"]
    assert [c["id"] for c in next_state["hand"]] == ["Defend_R", "Strike_R"]


def test_evolve_draws_extra_when_status_is_drawn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        draw_pile=[make_state.card("Wound"), make_state.card("Defend_R"), make_state.card("Bash")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        player_powers=[{"id": "Evolve", "amount": 1}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    hand_ids = [c["id"] for c in next_state["hand"]]
    assert "Wound" in hand_ids
    assert "Defend_R" in hand_ids
    assert len(hand_ids) >= 3


def test_metallicize_reduces_incoming_damage_during_end_turn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(intent="ATTACK", intent_base_damage=5)],
        player_powers=[{"id": "Metallicize", "amount": 3}],
        hp=50,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["player"]["current_hp"] == 48


def test_demon_form_grants_strength_at_start_of_next_turn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        player_powers=[{"id": "Demon Form", "amount": 2}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert _power_amount(next_state["player"], "Strength") == 2


def test_fire_breathing_hits_all_enemies_when_status_is_drawn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        draw_pile=[make_state.card("Wound"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30), make_state.enemy(name="Cultist", hp=28)],
        player_powers=[{"id": "Fire Breathing", "amount": 6}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 24
    assert next_state["monsters"][1]["current_hp"] == 22

from __future__ import annotations


def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0


def test_feel_no_pain_with_second_wind_grants_extra_block_from_exhausts(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Second Wind"),
            make_state.card("Defend_R"),
            make_state.card("Wound"),
        ],
        monsters=[make_state.enemy(hp=50)],
        energy=1,
        block=0,
        player_powers=[{"id": "Feel No Pain", "amount": 3}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False

    # Second Wind donne du block pour les non-attacks exhaustées
    # Feel No Pain ajoute encore du block à chaque exhaust
    assert next_state["player"]["block"] >= 16

    exhausted_ids = [c["id"] for c in next_state["exhaust_pile"]]
    assert "Defend_R" in exhausted_ids
    assert "Wound" in exhausted_ids


def test_dark_embrace_with_second_wind_draws_after_multiple_exhausts(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Second Wind"),
            make_state.card("Defend_R"),
            make_state.card("Wound"),
            make_state.card("Strike_R"),
        ],
        draw_pile=[
            make_state.card("Bash"),
            make_state.card("Pommel Strike"),
            make_state.card("Defend_R"),
        ],
        monsters=[make_state.enemy(hp=50)],
        energy=1,
        player_powers=[{"id": "Dark Embrace", "amount": 1}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False

    # Dark Embrace doit piocher quand des cartes sont exhaustées
    # Après avoir joué Second Wind, il doit rester au moins une carte en main.
    assert len(next_state["hand"]) >= 1

    exhausted_ids = [c["id"] for c in next_state["exhaust_pile"]]
    assert "Defend_R" in exhausted_ids
    assert "Wound" in exhausted_ids


def test_corruption_causes_skill_to_exhaust_instead_of_discard(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=40)],
        energy=1,
        player_powers=[{"id": "Corruption", "amount": 1}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False

    discard_ids = [c["id"] for c in next_state["discard_pile"]]
    exhaust_ids = [c["id"] for c in next_state["exhaust_pile"]]

    assert "Defend_R" not in discard_ids
    assert "Defend_R" in exhaust_ids


def test_rage_with_twin_strike_grants_block_once_per_attack_card_not_per_hit(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Twin Strike")],
        monsters=[make_state.enemy(hp=50)],
        energy=1,
        block=0,
        player_powers=[{"id": "Rage", "amount": 3}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False

    # Rage proc une fois à la lecture de la carte attaque, pas une fois par hit
    assert next_state["player"]["block"] == 3

    # Twin Strike : 2 hits de 5 = 10 dégâts
    assert next_state["monsters"][0]["current_hp"] == 40


def test_juggernaut_triggers_when_block_is_gained_from_skill(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=20)],
        energy=1,
        block=0,
        player_powers=[{"id": "Juggernaut", "amount": 5}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False

    # Defend_R donne du block, Juggernaut doit infliger des dégâts
    assert next_state["player"]["block"] == 5
    assert next_state["monsters"][0]["current_hp"] == 15
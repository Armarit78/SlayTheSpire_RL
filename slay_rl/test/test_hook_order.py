from __future__ import annotations


def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0


def test_combust_end_turn_order_direct_backend(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=5, intent="ATTACK", intent_base_damage=10)],
        player_powers=[{"id": "Combust", "amount": 5}],
    )
    backend.state = state

    hp_before = backend.state["player"]["current_hp"]

    next_state = backend._apply_end_turn(backend.state)

    # Combust doit faire perdre exactement 1 HP au joueur
    assert next_state["player"]["current_hp"] == hp_before - 1

    # Puis tuer l'ennemi avant qu'il n'agisse
    assert next_state["monsters"][0]["current_hp"] == 0
    assert next_state["monsters"][0]["isDead"] is True

    # Donc pas de dégâts ennemis
    assert next_state["hp_loss_breakdown"]["enemy"] == 0


def test_intangible_protects_during_enemy_attack_then_decays_at_end_turn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=40, intent="ATTACK", intent_base_damage=10)],
        hp=20,
        player_powers=[{"id": "Intangible", "amount": 1}],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False

    # L'attaque ennemie du tour courant est réduite à 1
    assert next_state["player"]["current_hp"] == 19
    assert next_state["hp_loss_breakdown"]["enemy"] == 1

    # Puis Intangible décroît à 0 à la fin du cycle
    assert _power_amount(next_state["player"], "Intangible") == 0


def test_mercury_hourglass_hook_order_on_start_turn_direct_backend(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=3, intent="DEFEND", intent_base_damage=0)],
        relics=[{"name": "Burning Blood"}, {"name": "Mercury Hourglass"}],
    )
    backend.state = state

    backend._start_turn_powers(backend.state)

    assert backend.state["monsters"][0]["current_hp"] == 0
    assert backend.state["monsters"][0]["isDead"] is True


def test_stone_calendar_hook_order_on_turn_seven_direct_backend(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=52, intent="DEFEND", intent_base_damage=0)],
        relics=[{"name": "Burning Blood"}, {"name": "Stone Calendar"}],
        turn=7,
    )
    backend.state = state

    backend._start_turn_powers(backend.state)

    assert backend.state["monsters"][0]["current_hp"] == 0
    assert backend.state["monsters"][0]["isDead"] is True
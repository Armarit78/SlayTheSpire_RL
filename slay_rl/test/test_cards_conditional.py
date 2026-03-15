from __future__ import annotations



def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0



def test_dropkick_only_gets_bonus_when_target_is_vulnerable(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Dropkick")],
        monsters=[make_state.enemy(hp=40, powers=[{"id": "Vulnerable", "amount": 2}])],
        energy=2,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["energy"] == 2
    assert next_state["monsters"][0]["current_hp"] == 32



def test_entrench_doubles_current_block(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Entrench")],
        monsters=[make_state.enemy()],
        block=11,
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] == 1
    assert next_state["player"]["block"] == 22



def test_body_slam_uses_current_block_as_damage(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Body Slam")],
        monsters=[make_state.enemy(hp=40)],
        block=17,
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["energy"] == 2
    assert next_state["monsters"][0]["current_hp"] == 23



def test_heavy_blade_scales_with_strength_mult(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Heavy Blade")],
        monsters=[make_state.enemy(hp=50)],
        player_powers=[{"id": "Strength", "amount": 2}],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 30



def test_limit_break_doubles_strength_and_exhausts(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Limit Break")],
        monsters=[make_state.enemy()],
        player_powers=[{"id": "Strength", "amount": 3}],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Strength") == 6
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Limit Break"]



def test_spot_weakness_only_grants_strength_when_target_is_attacking(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Spot Weakness")],
        monsters=[make_state.enemy(intent="ATTACK", intent_base_damage=10)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Strength") == 3

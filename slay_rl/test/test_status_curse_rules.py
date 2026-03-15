from __future__ import annotations



def _power_amount(entity, power_name: str) -> int:
    for power in entity.get("powers", []):
        if power.get("id") == power_name:
            return int(power.get("amount", 0))
    return 0



def test_battle_trance_draws_then_blocks_future_draw_this_turn(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Battle Trance"), make_state.card("Pommel Strike")],
        draw_pile=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
            make_state.card("Shrug It Off"),
        ],
        monsters=[make_state.enemy(hp=40)],
        energy=3,
    )
    step_helpers.set_state(state)

    after_bt, illegal_bt = step_helpers.play_card(hand_index=0)
    assert illegal_bt is False
    assert after_bt["combat_meta"]["cannot_draw_more_this_turn"] is True
    assert [c["id"] for c in after_bt["hand"]] == [
        "Pommel Strike",
        "Strike_R",
        "Defend_R",
        "Bash",
    ]

    after_pommel, illegal_pommel = step_helpers.play_card(hand_index=0, target_index=0)
    assert illegal_pommel is False
    assert after_pommel["monsters"][0]["current_hp"] == 31
    assert [c["id"] for c in after_pommel["hand"]] == ["Strike_R", "Defend_R", "Bash"]
    assert [c["id"] for c in after_pommel["draw_pile"]] == ["Shrug It Off"]



def test_pain_triggers_hp_loss_when_another_card_is_played(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Pain"), make_state.card("Defend_R")],
        monsters=[make_state.enemy()],
        hp=50,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=1)

    assert illegal is False
    assert next_state["player"]["current_hp"] == 49
    assert next_state["hp_loss_breakdown"]["pain"] == 1
    assert next_state["player"]["block"] == 5



def test_normality_blocks_fourth_card_play_in_same_turn(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Normality"),
            make_state.card("Anger"),
            make_state.card("Anger"),
            make_state.card("Anger"),
            make_state.card("Anger"),
        ],
        monsters=[make_state.enemy(hp=100)],
        combat_meta={"cards_played_this_turn": 0, "attacks_played_this_turn": 0},
        energy=0,
    )
    step_helpers.set_state(state)

    for _ in range(3):
        state_after, illegal = step_helpers.play_card(hand_index=1, target_index=0)
        assert illegal is False
    blocked_state, illegal = step_helpers.play_card(hand_index=1, target_index=0)

    assert illegal is True
    assert blocked_state["combat_meta"]["cards_played_this_turn"] == 3
    assert blocked_state["monsters"][0]["current_hp"] == 82



def test_end_turn_status_and_curse_effects_are_applied(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Burn"),
            make_state.card("Decay"),
            make_state.card("Doubt"),
            make_state.card("Shame"),
            make_state.card("Regret"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        hp=50,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["player"]["current_hp"] == 40
    assert next_state["hp_loss_breakdown"]["burn"] == 2
    assert next_state["hp_loss_breakdown"]["decay"] == 2
    assert next_state["hp_loss_breakdown"]["other"] == 6
    assert _power_amount(next_state["player"], "Weak") == 1
    assert _power_amount(next_state["player"], "Frail") == 1

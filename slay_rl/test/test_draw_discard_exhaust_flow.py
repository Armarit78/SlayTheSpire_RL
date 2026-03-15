from __future__ import annotations



def test_draw_reshuffles_discard_when_draw_pile_empty(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Pommel Strike")],
        draw_pile=[],
        discard_pile=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30)],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert [c["id"] for c in next_state["hand"]] == ["Defend_R"]
    assert [c["id"] for c in next_state["discard_pile"]] == ["Pommel Strike"]



def test_void_draw_reduces_energy_on_end_turn_refill(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        draw_pile=[make_state.card("Void"), make_state.card("Defend_R"), make_state.card("Bash")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert any(c["id"] == "Void" for c in next_state["hand"])
    assert next_state["energy"] == 2



def test_headbutt_moves_discard_card_to_top_then_next_turn_draws_it(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Headbutt")],
        draw_pile=[make_state.card("Defend_R"), make_state.card("Strike_R")],
        discard_pile=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40, intent="DEFEND", intent_base_damage=0)],
        energy=3,
    )
    step_helpers.set_state(state)

    after_headbutt, illegal = step_helpers.play_card(hand_index=0, target_index=0)
    assert illegal is False
    assert after_headbutt["draw_pile"][0]["id"] == "Bash"

    step_helpers.set_state(after_headbutt)
    next_state, illegal = step_helpers.end_turn()
    assert illegal is False
    assert next_state["hand"][0]["id"] == "Bash"



def test_sentinel_grants_energy_when_exhausted(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("True Grit"), make_state.card("Sentinel")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Sentinel"]
    assert next_state["energy"] == 4

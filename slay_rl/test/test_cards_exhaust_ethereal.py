from __future__ import annotations



def test_true_grit_exhausts_bad_card_and_gains_block(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("True Grit"), make_state.card("Wound"), make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["block"] == 7
    assert [c["id"] for c in next_state["discard_pile"]] == ["True Grit"]
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Wound"]
    assert [c["id"] for c in next_state["hand"]] == ["Strike_R"]



def test_burning_pact_exhausts_bad_card_and_draws_two(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Burning Pact"), make_state.card("Wound"), make_state.card("Strike_R")],
        draw_pile=[make_state.card("Defend_R"), make_state.card("Bash")],
        monsters=[make_state.enemy()],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] == 2
    assert next_state["pending_choice"]["choice_type"] == "choose_exhaust_target"
    assert next_state["pending_choice"]["valid_hand_indices"] == [1, 2]



def test_second_wind_exhausts_all_non_attacks_and_converts_them_to_block(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Second Wind"),
            make_state.card("Defend_R"),
            make_state.card("Wound"),
            make_state.card("Ghostly Armor"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy()],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["block"] == 15
    assert sorted(c["id"] for c in next_state["exhaust_pile"]) == ["Defend_R", "Ghostly Armor", "Second Wind", "Wound"]
    assert [c["id"] for c in next_state["discard_pile"]] == []
    assert [c["id"] for c in next_state["hand"]] == ["Strike_R"]



def test_fiend_fire_exhausts_rest_of_hand_and_scales_damage(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Fiend Fire"),
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Wound"),
        ],
        monsters=[make_state.enemy(hp=50)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["energy"] == 1
    assert next_state["monsters"][0]["current_hp"] == 22
    assert sorted(c["id"] for c in next_state["exhaust_pile"]) == ["Defend_R", "Fiend Fire", "Strike_R", "Wound"]
    assert next_state["hand"] == []



def test_ghostly_armor_ethereal_exhausts_when_turn_ends_unplayed(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Ghostly Armor"), make_state.card("Strike_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Ghostly Armor"]
    assert all(c["id"] != "Ghostly Armor" for c in next_state["discard_pile"])



def test_carnage_ethereal_exhausts_when_turn_ends_unplayed(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Carnage"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Carnage"]
    assert all(c["id"] != "Carnage" for c in next_state["discard_pile"])

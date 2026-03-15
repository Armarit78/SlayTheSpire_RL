from __future__ import annotations


def test_bash_deals_damage_applies_vulnerable_and_updates_meta(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["energy"] == 1
    assert next_state["monsters"][0]["current_hp"] == 32
    assert next_state["monsters"][0]["powers"] == [{"id": "Vulnerable", "amount": 2}]
    assert next_state["combat_meta"]["cards_played_this_turn"] == 1
    assert next_state["combat_meta"]["attacks_played_this_turn"] == 1
    assert [c["id"] for c in next_state["discard_pile"]] == ["Bash"]
    assert next_state["hand"] == []



def test_pommel_strike_draws_one_after_hitting(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Pommel Strike")],
        draw_pile=[make_state.card("Defend_R"), make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=30)],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["energy"] == 2
    assert next_state["monsters"][0]["current_hp"] == 21
    assert [c["id"] for c in next_state["hand"]] == ["Defend_R"]
    assert [c["id"] for c in next_state["discard_pile"]] == ["Pommel Strike"]
    assert [c["id"] for c in next_state["draw_pile"]] == ["Strike_R"]



def test_power_through_adds_block_and_two_wounds_to_hand(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Power Through")],
        monsters=[make_state.enemy(hp=35)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] == 2
    assert next_state["player"]["block"] == 15
    assert sorted(c["id"] for c in next_state["hand"]) == ["Wound", "Wound"]
    assert [c["id"] for c in next_state["discard_pile"]] == ["Power Through"]



def test_offering_costs_hp_grants_energy_draw_and_exhausts_itself(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Offering")],
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
    assert next_state["combat_meta"]["hp_loss_count"] == 1
    assert next_state["hp_loss_breakdown"]["other"] == 6
    assert next_state["energy"] == 3
    assert [c["id"] for c in next_state["hand"]] == [
        "Strike_R",
        "Defend_R",
        "Bash",
    ]
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Offering"]



def test_feed_kill_increases_max_hp_and_exhausts(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Feed")],
        monsters=[make_state.enemy(hp=10)],
        hp=60,
        max_hp=80,
        energy=2,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 0
    assert next_state["monsters"][0]["isDead"] is True
    assert next_state["player"]["max_hp"] == 83
    assert next_state["player"]["current_hp"] == 69
    assert [c["id"] for c in next_state["exhaust_pile"]] == ["Feed"]

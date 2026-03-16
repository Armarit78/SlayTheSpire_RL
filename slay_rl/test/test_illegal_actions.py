from __future__ import annotations



def test_cannot_play_card_without_enough_energy(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
        energy=1,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is True
    assert next_state["energy"] == 1
    assert next_state["monsters"][0]["current_hp"] == 40
    assert next_state["hand"][0]["id"] == "Bash"



def test_targeted_card_without_target_is_illegal(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is True
    assert next_state["monsters"][0]["current_hp"] == 40
    assert next_state["discard_pile"] == []



def test_unplayable_status_card_is_illegal(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Wound")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is True
    assert next_state["hand"][0]["id"] == "Wound"
    assert next_state["discard_pile"] == []



def test_clash_is_illegal_when_hand_contains_non_attack(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Clash"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is True
    assert next_state["monsters"][0]["current_hp"] == 30
    assert next_state["hand"][0]["id"] == "Clash"



def test_using_empty_potion_slot_is_illegal(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0)

    assert illegal is True
    assert next_state["potions"][0]["empty"] is True



def test_targeted_potion_on_dead_enemy_is_illegal(make_state, step_helpers):
    potions = [
        {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    dead_enemy = make_state.enemy(hp=0)
    dead_enemy["isDead"] = True
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[dead_enemy],
        potions=potions,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0, target_index=0)

    assert illegal is True
    assert next_state["potions"][0]["name"] == "Fear Potion"

def test_non_target_card_with_target_should_remain_legal_or_be_safely_ignored(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=40)],
        energy=1,
        block=0,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert len(next_state["hand"]) == 0
    assert next_state["energy"] == 0


def test_cannot_target_enemy_out_of_range_even_if_slot_exists(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=4)

    assert illegal is True
    assert next_state["monsters"][0]["current_hp"] == 40
    assert next_state["discard_pile"] == []


def test_using_targeted_potion_without_target_is_illegal(make_state, step_helpers):
    potions = [
        {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    state = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30)],
        potions=potions,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0)

    assert illegal is True
    assert next_state["potions"][0]["name"] == "Fear Potion"


def test_cannot_play_missing_hand_slot(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=30)],
        energy=1,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=3, target_index=0)

    assert illegal is True
    assert next_state["hand"][0]["id"] == "Strike_R"
    assert next_state["monsters"][0]["current_hp"] == 30

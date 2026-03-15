from __future__ import annotations



def test_fire_potion_targets_enemy_and_consumes_slot(make_state, step_helpers):
    potions = [
        {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=35)],
        potions=potions,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0, target_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 15
    assert next_state["potions"][0]["empty"] is True
    assert next_state["potions"][0]["name"] == "Empty Slot"



def test_explosive_potion_hits_all_enemies(make_state, step_helpers):
    potions = [
        {"name": "Explosive Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=30), make_state.enemy(name="Cultist", hp=25)],
        potions=potions,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] == 20
    assert next_state["monsters"][1]["current_hp"] == 15
    assert next_state["potions"][0]["empty"] is True



def test_block_potion_grants_block_without_target(make_state, step_helpers):
    potions = [
        {"name": "Block Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
        potions=potions,
        block=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0)

    assert illegal is False
    assert next_state["player"]["block"] == 15
    assert next_state["potions"][0]["empty"] is True



def test_targeted_potion_without_target_is_illegal(make_state, step_helpers):
    potions = [
        {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=30)],
        potions=potions,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.use_potion(potion_index=0)

    assert illegal is True
    assert next_state["monsters"][0]["powers"] == []
    assert next_state["potions"][0]["name"] == "Fear Potion"

from __future__ import annotations
from slay_rl.agents.combat_agent import CombatCommand


def test_full_combat_two_strikes_kill_single_enemy(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy(hp=12, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=2,
    )

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert state["monsters"][0]["current_hp"] == 6

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert state["monsters"][0]["current_hp"] <= 0
    assert state["monsters"][0].get("isDead", False) is True


def test_full_combat_inflame_then_strike_increases_followup_damage(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Inflame"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy(hp=20, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=2,
        player_powers=[],
    )

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert any(p.get("id") == "Strength" for p in state["player"]["powers"])

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert state["monsters"][0]["current_hp"] == 12


def test_full_combat_bash_then_strike_gains_value_from_vulnerable(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Bash"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy(hp=30, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=3,
    )

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    enemy = state["monsters"][0]
    assert enemy["current_hp"] == 22
    assert any(p.get("id") == "Vulnerable" for p in enemy.get("powers", []))

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    assert state["monsters"][0]["current_hp"] < 16


def test_full_combat_whirlwind_hits_all_enemies_and_spends_energy(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Whirlwind")],
        monsters=[
            make_state.enemy(hp=30, block=0, intent="ATTACK", intent_base_damage=8),
            make_state.enemy(name="Cultist", hp=25, block=0, intent="BUFF", intent_base_damage=0),
        ],
        hp=50,
        energy=3,
    )

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert state["energy"] == 0
    assert state["monsters"][0]["current_hp"] < 30
    assert state["monsters"][1]["current_hp"] < 25


def test_full_combat_block_potion_then_defend_stacks_survivability_tools(make_state, step_helpers):
    potions = [
        {"name": "Block Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Common"},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]
    state = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=35, block=0, intent="ATTACK", intent_base_damage=18)],
        hp=18,
        block=0,
        energy=1,
        potions=potions,
    )

    step_helpers.set_state(state)
    state, illegal = step_helpers.use_potion(potion_index=0)

    assert illegal is False
    assert state["potions"][0]["empty"] is True

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert len(state["hand"]) == 0
    assert state["energy"] == 0


def test_full_combat_offering_then_strike_converts_hp_into_tempo(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Offering"),
            make_state.card("Strike_R"),
        ],
        draw_pile=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
        ],
        monsters=[make_state.enemy(hp=20, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        max_hp=80,
        energy=1,
    )

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert state["player"]["current_hp"] < 50
    assert state["energy"] > 1
    assert len(state["hand"]) >= 2

    strike_idx = next(i for i, c in enumerate(state["hand"]) if c.get("id") == "Strike_R")

    step_helpers.set_state(state)
    state, illegal = step_helpers.play_card(hand_index=strike_idx, target_index=0)

    assert illegal is False
    assert state["monsters"][0]["current_hp"] < 20

def test_full_combat_choose_discard_target_moves_card_to_discard(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
        ],
        monsters=[make_state.enemy(hp=30)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_discard_target",
        "valid_hand_indices": [1],
    }

    step_helpers.set_state(state)
    state, illegal = step_helpers.step(
        CombatCommand(command_type="choose_discard_target", hand_index=1)
    )

    assert illegal is False
    assert len(state["hand"]) == 2
    assert state["discard_pile"][-1]["id"] == "Defend_R"
    assert "pending_choice" not in state


def test_full_combat_choose_exhaust_target_moves_card_to_exhaust(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Inflame"),
        ],
        monsters=[make_state.enemy(hp=30)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_exhaust_target",
        "valid_hand_indices": [2],
    }

    step_helpers.set_state(state)
    state, illegal = step_helpers.step(
        CombatCommand(command_type="choose_exhaust_target", hand_index=2)
    )

    assert illegal is False
    assert len(state["hand"]) == 2
    assert state["exhaust_pile"][-1]["id"] == "Inflame"
    assert "pending_choice" not in state


def test_full_combat_choose_hand_card_puts_card_on_top_of_draw(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
        ],
        draw_pile=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=30)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_hand_card",
        "valid_hand_indices": [0],
    }

    step_helpers.set_state(state)
    state, illegal = step_helpers.step(
        CombatCommand(command_type="choose_hand_card", hand_index=0)
    )

    assert illegal is False
    assert len(state["hand"]) == 1
    assert state["draw_pile"][0]["id"] == "Strike_R"
    assert "pending_choice" not in state


def test_full_combat_choose_option_adds_selected_option_to_hand(make_state, step_helpers):
    state = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_option",
        "options": ["Anger", "Pommel Strike", "Shrug It Off"],
    }

    step_helpers.set_state(state)
    state, illegal = step_helpers.step(
        CombatCommand(command_type="choose_option", target_index=1)
    )

    assert illegal is False
    assert len(state["hand"]) == 1
    assert state["hand"][0]["id"] == "Pommel Strike"
    assert "pending_choice" not in state
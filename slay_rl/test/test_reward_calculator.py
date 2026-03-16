from __future__ import annotations

import pytest

from slay_rl.rewards.combat_reward import CombatRewardCalculator


def test_reward_includes_damage_kill_and_win_bonus(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(hand=[], monsters=[make_state.enemy(hp=10)], hp=50)
    next_enemy = make_state.enemy(hp=0)
    next_enemy["isDead"] = True
    next_state = make_state(hand=[], monsters=[next_enemy], hp=50)
    next_state["combat_over"] = True
    next_state["victory"] = True

    out = calc.compute(prev_state, next_state, action_info={"illegal_action": False})

    assert out.damage_dealt_reward == pytest.approx(10 * 0.07)
    assert out.kill_bonus == pytest.approx(1.25)
    assert out.win_bonus == pytest.approx(6.0)
    assert out.combat_won is True
    assert out.total_reward > 7.0


def test_reward_penalizes_player_damage_and_bad_card_hp_loss(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[make_state.card("Burn")],
        monsters=[make_state.enemy(hp=30)],
        hp=50,
    )
    next_state = make_state(
        hand=[make_state.card("Burn")],
        monsters=[make_state.enemy(hp=30)],
        hp=46,
    )
    next_state["hp_loss_breakdown"] = {
        "enemy": 0,
        "burn": 2,
        "pain": 1,
        "decay": 1,
        "other": 0,
    }

    out = calc.compute(prev_state, next_state, action_info={"illegal_action": False})

    assert out.damage_taken_reward == pytest.approx(-0.48)
    assert out.self_hp_loss_from_bad_cards_penalty == pytest.approx(-0.12)
    assert out.step_penalty == pytest.approx(-0.03)
    assert out.total_reward < 0


def test_reward_penalizes_illegal_action(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(hand=[], monsters=[make_state.enemy(hp=30)], hp=50)
    next_state = make_state(hand=[], monsters=[make_state.enemy(hp=30)], hp=50)

    out = calc.compute(prev_state, next_state, action_info={"illegal_action": True})

    assert out.illegal_action_penalty == pytest.approx(-0.5)
    assert out.total_reward < 0


def test_reward_gives_block_debuff_and_buff_components(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=10)],
        hp=50,
        block=0,
        player_powers=[],
    )
    next_state = make_state(
        hand=[],
        monsters=[
            make_state.enemy(
                hp=30,
                intent="ATTACK",
                intent_base_damage=10,
                powers=[{"id": "Vulnerable", "amount": 2}],
            )
        ],
        hp=50,
        block=6,
        player_powers=[{"id": "Strength", "amount": 2}],
    )

    out = calc.compute(prev_state, next_state, action_info={"illegal_action": False})

    assert out.block_reward > 0.0
    assert out.debuff_reward > 0.0
    assert out.buff_reward > 0.0


def test_reward_inflame_generates_setup_and_sequencing_reward(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[
            make_state.card("Inflame"),
            make_state.card("Strike_R"),
            make_state.card("Bash"),
        ],
        monsters=[make_state.enemy(hp=40)],
        hp=70,
        energy=3,
        player_powers=[],
        combat_meta={
            "cards_played_this_turn": 0,
            "attacks_played_this_turn": 0,
            "double_tap_charges": 0,
        },
    )
    next_state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Bash"),
        ],
        monsters=[make_state.enemy(hp=40)],
        hp=70,
        energy=2,
        player_powers=[{"id": "Strength", "amount": 2}],
        combat_meta={
            "cards_played_this_turn": 1,
            "attacks_played_this_turn": 0,
            "double_tap_charges": 0,
        },
    )

    out = calc.compute(
        prev_state,
        next_state,
        action_info={
            "command_type": "play_card",
            "card_name": "Inflame",
            "illegal_action": False,
        },
    )

    assert out.buff_reward > 0.0
    assert out.setup_reward > 0.0
    assert out.sequencing_reward > 0.0
    assert out.energy_reward > 0.0


def test_reward_double_tap_setup_reward_scales_with_followup_attack(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[
            make_state.card("Double Tap"),
            {"id": "Bludgeon", "cost": 3, "type": "ATTACK", "damage": 32},
        ],
        monsters=[make_state.enemy(hp=50)],
        energy=4,
        combat_meta={"double_tap_charges": 0},
    )
    next_state = make_state(
        hand=[
            {"id": "Bludgeon", "cost": 3, "type": "ATTACK", "damage": 32},
        ],
        monsters=[make_state.enemy(hp=50)],
        energy=3,
        combat_meta={"double_tap_charges": 1},
    )

    out = calc.compute(
        prev_state,
        next_state,
        action_info={"command_type": "play_card", "card_name": "Double Tap"},
    )

    assert out.setup_reward > 0.0
    assert out.sequencing_reward > 0.0


def test_reward_end_turn_with_wasted_energy_is_penalized(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
        ],
        monsters=[make_state.enemy(hp=40, intent="ATTACK", intent_base_damage=10)],
        energy=2,
        block=0,
    )
    next_state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
        ],
        monsters=[make_state.enemy(hp=40, intent="ATTACK", intent_base_damage=10)],
        energy=2,
        block=0,
    )

    out = calc.compute(
        prev_state,
        next_state,
        action_info={"command_type": "end_turn", "illegal_action": False},
    )

    assert out.step_penalty < 0.0
    assert out.energy_reward < 0.0
    assert out.total_reward < 0.0


def test_reward_potion_timing_is_better_in_emergency_than_low_threat(make_state):
    calc = CombatRewardCalculator()

    potions_before = [
        {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True},
        *[
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False}
            for _ in range(4)
        ],
    ]
    potions_after = [
        {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        *[
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False}
            for _ in range(4)
        ],
    ]

    low_prev = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="BUFF", intent_base_damage=0)],
        hp=70,
        max_hp=80,
        potions=potions_before,
    )
    low_next = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=20, intent="BUFF", intent_base_damage=0)],
        hp=70,
        max_hp=80,
        potions=potions_after,
    )

    high_prev = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=20,
        max_hp=80,
        potions=potions_before,
    )
    high_next = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=20, intent="ATTACK", intent_base_damage=18)],
        hp=20,
        max_hp=80,
        potions=potions_after,
    )

    low = calc.compute(low_prev, low_next, action_info={"command_type": "use_potion"})
    high = calc.compute(high_prev, high_next, action_info={"command_type": "use_potion"})

    assert low.potion_use_penalty < 0.0
    assert high.potion_timing_reward > 0.0
    assert high.total_reward > low.total_reward


def test_reward_lethal_reward_triggers_on_low_enemy_finish(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[],
        monsters=[
            make_state.enemy(hp=7, block=0, intent="ATTACK", intent_base_damage=8),
            make_state.enemy(name="Cultist", hp=30, block=0, intent="BUFF", intent_base_damage=0),
        ],
        hp=50,
    )

    killed_enemy = make_state.enemy(hp=0, block=0, intent="ATTACK", intent_base_damage=8)
    killed_enemy["isDead"] = True

    next_state = make_state(
        hand=[],
        monsters=[
            killed_enemy,
            make_state.enemy(name="Cultist", hp=30, block=0, intent="BUFF", intent_base_damage=0),
        ],
        hp=50,
    )

    out = calc.compute(
        prev_state,
        next_state,
        action_info={"command_type": "play_card", "card_name": "Strike_R"},
    )

    assert out.kill_bonus > 0.0
    assert out.lethal_reward >= 0.0


def test_reward_status_curse_reduction_is_rewarded(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[
            make_state.card("Wound"),
            make_state.card("Burn"),
            make_state.card("Strike_R"),
        ],
        monsters=[make_state.enemy(hp=30)],
        hp=50,
    )
    next_state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=30)],
        hp=50,
    )

    out = calc.compute(prev_state, next_state, action_info={"command_type": "play_card"})

    assert out.status_curse_hand_reward > 0.0

def test_reward_win_must_outweigh_small_setup_gain(make_state):
    calc = CombatRewardCalculator()

    prev_setup = make_state(
        hand=[make_state.card("Inflame"), make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=8, intent="ATTACK", intent_base_damage=8)],
        hp=40,
        energy=2,
        player_powers=[],
        combat_meta={"cards_played_this_turn": 0, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )
    next_setup = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=8, intent="ATTACK", intent_base_damage=8)],
        hp=40,
        energy=1,
        player_powers=[{"id": "Strength", "amount": 2}],
        combat_meta={"cards_played_this_turn": 1, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )

    prev_kill = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Inflame")],
        monsters=[make_state.enemy(hp=6, intent="ATTACK", intent_base_damage=8)],
        hp=40,
        energy=1,
        player_powers=[],
        combat_meta={"cards_played_this_turn": 0, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )
    dead_enemy = make_state.enemy(hp=0, intent="ATTACK", intent_base_damage=8)
    dead_enemy["isDead"] = True
    next_kill = make_state(
        hand=[make_state.card("Inflame")],
        monsters=[dead_enemy],
        hp=40,
        energy=0,
        player_powers=[],
        combat_meta={"cards_played_this_turn": 1, "attacks_played_this_turn": 1, "double_tap_charges": 0},
    )
    next_kill["combat_over"] = True
    next_kill["victory"] = True

    out_setup = calc.compute(
        prev_setup,
        next_setup,
        action_info={"command_type": "play_card", "card_name": "Inflame", "illegal_action": False},
    )
    out_kill = calc.compute(
        prev_kill,
        next_kill,
        action_info={"command_type": "play_card", "card_name": "Strike_R", "illegal_action": False},
    )

    assert out_setup.setup_reward > 0.0
    assert out_kill.combat_won is True
    assert out_kill.total_reward > out_setup.total_reward


def test_reward_survival_block_should_outweigh_do_nothing_end_turn(make_state):
    calc = CombatRewardCalculator()

    prev_block = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=12,
        block=0,
        energy=1,
    )
    next_block = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=12,
        block=5,
        energy=0,
    )

    prev_idle = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=12,
        block=0,
        energy=1,
    )
    next_idle = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=12,
        block=0,
        energy=1,
    )

    out_block = calc.compute(
        prev_block,
        next_block,
        action_info={"command_type": "play_card", "card_name": "Defend_R", "illegal_action": False},
    )
    out_idle = calc.compute(
        prev_idle,
        next_idle,
        action_info={"command_type": "end_turn", "illegal_action": False},
    )

    assert out_block.block_reward > 0.0
    assert out_idle.energy_reward < 0.0
    assert out_block.total_reward > out_idle.total_reward


def test_reward_should_not_prefer_stalling_over_finishing_combat(make_state):
    calc = CombatRewardCalculator()

    prev_finish = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=6, intent="BUFF", intent_base_damage=0)],
        hp=50,
        energy=1,
    )
    dead_enemy = make_state.enemy(hp=0, intent="BUFF", intent_base_damage=0)
    dead_enemy["isDead"] = True
    next_finish = make_state(
        hand=[],
        monsters=[dead_enemy],
        hp=50,
        energy=0,
    )
    next_finish["combat_over"] = True
    next_finish["victory"] = True

    prev_stall = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=6, intent="BUFF", intent_base_damage=0)],
        hp=50,
        energy=1,
        block=0,
    )
    next_stall = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=6, intent="BUFF", intent_base_damage=0)],
        hp=50,
        energy=0,
        block=5,
    )

    out_finish = calc.compute(
        prev_finish,
        next_finish,
        action_info={"command_type": "play_card", "card_name": "Strike_R", "illegal_action": False},
    )
    out_stall = calc.compute(
        prev_stall,
        next_stall,
        action_info={"command_type": "play_card", "card_name": "Defend_R", "illegal_action": False},
    )

    assert out_finish.combat_won is True
    assert out_finish.total_reward > out_stall.total_reward
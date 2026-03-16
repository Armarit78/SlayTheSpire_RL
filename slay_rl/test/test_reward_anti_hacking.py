from __future__ import annotations

from slay_rl.rewards.combat_reward import CombatRewardCalculator


def test_reward_prefers_finishing_combat_over_extra_block_when_enemy_is_already_lethal_range(make_state):
    calc = CombatRewardCalculator()

    prev_finish = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=6, block=0, intent="BUFF", intent_base_damage=0)],
        hp=50,
        block=0,
        energy=1,
    )
    dead_enemy = make_state.enemy(hp=0, block=0, intent="BUFF", intent_base_damage=0)
    dead_enemy["isDead"] = True
    next_finish = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[dead_enemy],
        hp=50,
        block=0,
        energy=0,
    )
    next_finish["combat_over"] = True
    next_finish["victory"] = True

    prev_block = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=6, block=0, intent="BUFF", intent_base_damage=0)],
        hp=50,
        block=0,
        energy=1,
    )
    next_block = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=6, block=0, intent="BUFF", intent_base_damage=0)],
        hp=50,
        block=5,
        energy=0,
    )

    out_finish = calc.compute(
        prev_finish,
        next_finish,
        action_info={"command_type": "play_card", "card_name": "Strike_R", "illegal_action": False},
    )
    out_block = calc.compute(
        prev_block,
        next_block,
        action_info={"command_type": "play_card", "card_name": "Defend_R", "illegal_action": False},
    )

    assert out_finish.combat_won is True
    assert out_finish.total_reward > out_block.total_reward


def test_reward_prefers_kill_over_pretty_setup_that_does_not_end_fight(make_state):
    calc = CombatRewardCalculator()

    prev_setup = make_state(
        hand=[make_state.card("Inflame"), make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=6, block=0, intent="ATTACK", intent_base_damage=9)],
        hp=40,
        energy=2,
        player_powers=[],
        combat_meta={"cards_played_this_turn": 0, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )
    next_setup = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=6, block=0, intent="ATTACK", intent_base_damage=9)],
        hp=40,
        energy=1,
        player_powers=[{"id": "Strength", "amount": 2}],
        combat_meta={"cards_played_this_turn": 1, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )

    prev_kill = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Inflame")],
        monsters=[make_state.enemy(hp=6, block=0, intent="ATTACK", intent_base_damage=9)],
        hp=40,
        energy=1,
        player_powers=[],
        combat_meta={"cards_played_this_turn": 0, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )
    dead_enemy = make_state.enemy(hp=0, block=0, intent="ATTACK", intent_base_damage=9)
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


def test_reward_emergency_potion_use_beats_low_value_potion_burn(make_state):
    calc = CombatRewardCalculator()

    potions_before = [
        {"name": "Block Potion", "usable": True, "empty": False, "requires_target": False},
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
        block=0,
        potions=potions_before,
    )
    low_next = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="BUFF", intent_base_damage=0)],
        hp=70,
        max_hp=80,
        block=12,
        potions=potions_after,
    )

    emergency_prev = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=12,
        max_hp=80,
        block=0,
        potions=potions_before,
    )
    emergency_next = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=18)],
        hp=12,
        max_hp=80,
        block=12,
        potions=potions_after,
    )

    low = calc.compute(low_prev, low_next, action_info={"command_type": "use_potion", "illegal_action": False})
    high = calc.compute(emergency_prev, emergency_next, action_info={"command_type": "use_potion", "illegal_action": False})

    assert low.potion_use_penalty <= 0.0
    assert high.potion_timing_reward >= 0.0
    assert high.total_reward > low.total_reward


def test_reward_losing_after_setup_is_still_bad(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(
        hand=[make_state.card("Inflame"), make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=20, intent="ATTACK", intent_base_damage=40)],
        hp=10,
        max_hp=80,
        energy=2,
        player_powers=[],
        combat_meta={"cards_played_this_turn": 0, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )
    next_state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=20, intent="ATTACK", intent_base_damage=40)],
        hp=0,
        max_hp=80,
        energy=1,
        player_powers=[{"id": "Strength", "amount": 2}],
        combat_meta={"cards_played_this_turn": 1, "attacks_played_this_turn": 0, "double_tap_charges": 0},
    )
    next_state["combat_over"] = True
    next_state["victory"] = False

    out = calc.compute(
        prev_state,
        next_state,
        action_info={"command_type": "play_card", "card_name": "Inflame", "illegal_action": False},
    )

    assert out.setup_reward > 0.0
    assert out.combat_lost is True
    assert out.total_reward < 0.0


def test_reward_illegal_action_should_not_outscore_legal_progress(make_state):
    calc = CombatRewardCalculator()

    prev_legal = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=20, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=1,
    )
    next_legal = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=14, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=0,
    )

    prev_illegal = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=20, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=1,
    )
    next_illegal = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=20, block=0, intent="ATTACK", intent_base_damage=8)],
        hp=50,
        energy=1,
    )

    legal = calc.compute(
        prev_legal,
        next_legal,
        action_info={"command_type": "play_card", "card_name": "Strike_R", "illegal_action": False},
    )
    illegal = calc.compute(
        prev_illegal,
        next_illegal,
        action_info={"command_type": "play_card", "card_name": "Bash", "illegal_action": True},
    )

    assert legal.total_reward > illegal.total_reward
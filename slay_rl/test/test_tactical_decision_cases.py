from __future__ import annotations

from slay_rl.agents.combat_agent import RuleBasedCombatAgent


def test_rule_agent_prefers_lethal_targeted_attack_over_setup(make_state):
    agent = RuleBasedCombatAgent()

    state = make_state(
        hand=[
            make_state.card("Inflame"),
            make_state.card("Strike_R"),
            make_state.card("Bash"),
        ],
        monsters=[make_state.enemy(hp=6, block=0, intent="ATTACK", intent_base_damage=8)],
        energy=2,
    )

    cmd = agent.choose_command(state)

    assert cmd.command_type == "play_card"
    assert cmd.hand_index == 1
    assert cmd.target_index == 0


def test_rule_agent_targets_lowest_hp_enemy_for_targeted_attack(make_state):
    agent = RuleBasedCombatAgent()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[
            make_state.enemy(hp=30, block=0, intent="ATTACK", intent_base_damage=8),
            make_state.enemy(name="Cultist", hp=5, block=0, intent="BUFF", intent_base_damage=0),
        ],
        energy=1,
    )

    cmd = agent.choose_command(state)

    assert cmd.command_type == "play_card"
    assert cmd.hand_index == 0
    assert cmd.target_index == 1


def test_rule_agent_plays_best_affordable_attack_before_support(make_state):
    agent = RuleBasedCombatAgent()

    state = make_state(
        hand=[
            make_state.card("Inflame"),
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
        ],
        monsters=[make_state.enemy(hp=40, block=0, intent="ATTACK", intent_base_damage=10)],
        energy=1,
    )

    cmd = agent.choose_command(state)

    assert cmd.command_type == "play_card"
    assert cmd.hand_index == 1
    assert cmd.target_index == 0


def test_rule_agent_plays_support_when_no_attack_is_playable(make_state):
    agent = RuleBasedCombatAgent()

    state = make_state(
        hand=[
            make_state.card("Inflame"),
            make_state.card("Defend_R"),
        ],
        monsters=[make_state.enemy(hp=40, block=0, intent="ATTACK", intent_base_damage=10)],
        energy=1,
    )

    cmd = agent.choose_command(state)

    assert cmd.command_type == "play_card"
    assert cmd.hand_index in (0, 1)
    assert cmd.target_index is None


def test_rule_agent_ends_turn_when_only_unplayable_cards_exist(make_state):
    agent = RuleBasedCombatAgent()

    state = make_state(
        hand=[
            make_state.card("Wound"),
            make_state.card("Burn"),
        ],
        monsters=[make_state.enemy(hp=35, block=0, intent="ATTACK", intent_base_damage=8)],
        energy=3,
    )

    cmd = agent.choose_command(state)

    assert cmd.command_type == "end_turn"


def test_rule_agent_does_not_target_dead_enemy_for_lethal_check(make_state):
    agent = RuleBasedCombatAgent()

    dead_enemy = make_state.enemy(hp=0, block=0, intent="ATTACK", intent_base_damage=8)
    dead_enemy["isDead"] = True

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[
            dead_enemy,
            make_state.enemy(name="Cultist", hp=20, block=0, intent="BUFF", intent_base_damage=0),
        ],
        energy=1,
    )

    cmd = agent.choose_command(state)

    assert cmd.command_type == "play_card"
    assert cmd.hand_index == 0
    assert cmd.target_index == 1
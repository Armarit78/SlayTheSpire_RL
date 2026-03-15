from __future__ import annotations

import pytest

from slay_rl.rewards.combat_reward import CombatRewardCalculator


def test_reward_includes_damage_and_kill_bonus(make_state):
    calc = CombatRewardCalculator()

    prev_state = make_state(hand=[], monsters=[make_state.enemy(hp=10)], hp=50)
    next_enemy = make_state.enemy(hp=0)
    next_enemy["isDead"] = True
    next_state = make_state(hand=[], monsters=[next_enemy], hp=50)

    out = calc.compute(prev_state, next_state, action_info={"illegal_action": False})

    assert out.damage_dealt_reward == pytest.approx(0.7)
    assert out.kill_bonus == pytest.approx(1.25)
    assert out.win_bonus == pytest.approx(6.0)
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


def test_reward_penalizes_illegal_action_and_potion_use(make_state):
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

    prev_state = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30)],
        hp=50,
        potions=potions_before,
    )
    next_state = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=30)],
        hp=50,
        potions=potions_after,
    )

    out = calc.compute(prev_state, next_state, action_info={"illegal_action": True})

    assert out.illegal_action_penalty == pytest.approx(-0.5)
    assert out.potion_use_penalty == pytest.approx(-0.01)
    assert out.step_penalty == pytest.approx(-0.03)
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

    assert out.block_reward == pytest.approx(0.06)
    assert out.debuff_reward == pytest.approx(0.08)
    assert out.buff_reward == pytest.approx(0.03)
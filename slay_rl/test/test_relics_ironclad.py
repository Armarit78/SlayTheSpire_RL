from __future__ import annotations

import pytest

from slay_rl.agents.combat_agent import CombatCommand


def _has_power(entity, power_id: str, amount: int | None = None) -> bool:
    for p in entity.get("powers", []):
        if p.get("id") != power_id:
            continue
        if amount is None:
            return True
        if int(p.get("amount", 0)) == int(amount):
            return True
    return False


def _get_power_amount(entity, power_id: str) -> int:
    for p in entity.get("powers", []):
        if p.get("id") == power_id:
            return int(p.get("amount", 0))
    return 0


def _refresh_with_relics(backend, state):
    backend.state = state
    backend._apply_relics_on_combat_start(backend.state)
    backend._update_enemy_intents(backend.state)
    return backend.state


def test_anchor_should_grant_block_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Anchor"}],
    )

    state = _refresh_with_relics(backend, state)

    assert state["player"]["block"] == 10


def test_lantern_should_grant_extra_energy_on_turn_one(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Lantern"}],
        energy=3,
    )

    state = _refresh_with_relics(backend, state)

    assert state["energy"] == 4


def test_bag_of_marbles_should_apply_vulnerable_to_all_enemies_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Bag of Marbles"}],
        monsters=[make_state.enemy(hp=30), make_state.enemy(name="Cultist", hp=28)],
    )

    state = _refresh_with_relics(backend, state)

    for monster in state["monsters"]:
        assert _has_power(monster, "Vulnerable", 1)


def test_bag_of_preparation_should_draw_two_extra_cards_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        draw_pile=[
            make_state.card("Defend_R"),
            make_state.card("Bash"),
            make_state.card("Pommel Strike"),
        ],
        relics=[{"name": "Burning Blood"}, {"name": "Bag of Preparation"}],
    )

    state = _refresh_with_relics(backend, state)

    assert len(state["hand"]) == 3
    assert len(state["draw_pile"]) == 1


def test_blood_vial_should_heal_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Blood Vial"}],
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80

    state = _refresh_with_relics(backend, state)

    assert state["player"]["current_hp"] == 52


def test_bronze_scales_should_grant_thorns_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Bronze Scales"}],
    )

    state = _refresh_with_relics(backend, state)

    assert _has_power(state["player"], "Thorns", 3)


def test_oddly_smooth_stone_should_grant_dexterity_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Oddly Smooth Stone"}],
    )

    state = _refresh_with_relics(backend, state)

    assert _has_power(state["player"], "Dexterity", 1)


def test_vajra_should_grant_strength_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Vajra"}],
    )

    state = _refresh_with_relics(backend, state)

    assert _has_power(state["player"], "Strength", 1)


def test_thread_and_needle_should_grant_plated_armor_at_combat_start(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Thread and Needle"}],
    )

    state = _refresh_with_relics(backend, state)

    assert _has_power(state["player"], "Plated Armor", 4)


def test_gambling_chip_should_discard_hand_and_redraw_five(make_state, backend):
    state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
        ],
        draw_pile=[
            make_state.card("Anger"),
            make_state.card("Pommel Strike"),
            make_state.card("Twin Strike"),
            make_state.card("Shrug It Off"),
            make_state.card("Inflame"),
            make_state.card("Cleave"),
        ],
        discard_pile=[],
        relics=[{"name": "Burning Blood"}, {"name": "Gambling Chip"}],
    )

    state = _refresh_with_relics(backend, state)

    assert len(state["hand"]) == 5
    assert len(state["discard_pile"]) == 3


def test_orichalcum_should_grant_block_at_end_turn_when_block_is_zero(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Orichalcum"}],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        block=0,
    )
    backend.state = state

    backend._apply_relics_at_end_turn_before_enemies(backend.state)

    assert backend.state["player"]["block"] == 6


def test_horn_cleat_should_grant_block_on_turn_two(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Horn Cleat"}],
        turn=2,
    )
    backend.state = state

    backend._start_turn_powers(backend.state)

    assert backend.state["player"]["block"] == 14


def test_mercury_hourglass_should_damage_all_enemies_at_start_of_turn(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Mercury Hourglass"}],
        monsters=[make_state.enemy(hp=30), make_state.enemy(name="Cultist", hp=28)],
    )
    backend.state = state

    hp_before = [m["current_hp"] for m in backend.state["monsters"]]
    backend._start_turn_powers(backend.state)
    hp_after = [m["current_hp"] for m in backend.state["monsters"]]

    assert hp_after == [hp_before[0] - 3, hp_before[1] - 3]


def test_stone_calendar_should_damage_all_enemies_on_turn_seven(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Stone Calendar"}],
        monsters=[make_state.enemy(hp=60), make_state.enemy(name="Cultist", hp=55)],
        turn=7,
    )
    backend.state = state

    backend._start_turn_powers(backend.state)

    assert backend.state["monsters"][0]["current_hp"] <= 8
    assert backend.state["monsters"][1]["current_hp"] <= 3


def test_self_forming_clay_should_gain_block_next_turn_after_hp_loss(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Self-Forming Clay"}],
    )
    state["player"]["current_hp"] = 60
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._lose_hp(backend.state["player"], backend.state, 5, source="enemy")
    backend._apply_relics_at_start_of_turn(backend.state)

    assert backend.state["player"]["block"] == 3


def test_runic_cube_should_draw_when_player_loses_hp(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        draw_pile=[make_state.card("Defend_R"), make_state.card("Bash")],
        relics=[{"name": "Burning Blood"}, {"name": "Runic Cube"}],
    )
    state["player"]["current_hp"] = 60
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._lose_hp(backend.state["player"], backend.state, 3, source="enemy")

    assert len(backend.state["hand"]) == 2
    assert len(backend.state["draw_pile"]) == 1


def test_red_skull_should_grant_bonus_damage_below_half_hp(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Red Skull"}],
    )
    state["player"]["current_hp"] = 40
    state["player"]["max_hp"] = 80

    backend.state = state

    card = make_state.card("Strike_R")
    card_def = backend._get_effective_card_def(card)
    dmg = backend._compute_base_damage(card, card_def, backend.state)

    assert dmg == 9


def test_paper_phrog_should_increase_damage_to_vulnerable_enemy(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Paper Phrog"}],
        monsters=[make_state.enemy(hp=40, powers=[{"id": "Vulnerable", "amount": 1}])],
    )
    backend.state = state

    dealt = backend._deal_damage_to_monster(backend.state, backend.state["monsters"][0], 10)

    assert dealt == 18


def test_champion_belt_should_apply_weak_when_vulnerable_is_applied(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Champion Belt"}],
        monsters=[make_state.enemy(hp=30)],
    )
    backend.state = state

    backend._apply_power_to_monster(backend.state, backend.state["monsters"][0], "Vulnerable", 1)

    monster = backend.state["monsters"][0]
    assert _has_power(monster, "Vulnerable", 1)
    assert _has_power(monster, "Weak", 1)


def test_akabeko_should_buff_first_attack_only(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Akabeko"}],
    )
    backend.state = state

    card1 = backend.state["hand"][0]
    def1 = backend._get_effective_card_def(card1)
    dmg1 = backend._compute_base_damage(card1, def1, backend.state)

    card2 = backend.state["hand"][1]
    def2 = backend._get_effective_card_def(card2)
    dmg2 = backend._compute_base_damage(card2, def2, backend.state)

    assert dmg1 == 14
    assert dmg2 == 6


def test_tungsten_rod_should_reduce_hp_loss_by_one(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Tungsten Rod"}],
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._lose_hp(backend.state["player"], backend.state, 5, source="enemy")

    assert backend.state["player"]["current_hp"] == 46


def test_torii_should_reduce_small_hp_loss_to_one(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Torii"}],
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._lose_hp(backend.state["player"], backend.state, 5, source="enemy")

    assert backend.state["player"]["current_hp"] == 49


def test_bronze_scales_should_damage_attacker(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Bronze Scales"}],
        monsters=[make_state.enemy(hp=30, intent="ATTACK", intent_base_damage=6)],
    )
    state = _refresh_with_relics(backend, state)

    monster_hp_before = state["monsters"][0]["current_hp"]
    backend._deal_monster_attack_to_player(state, state["monsters"][0], 6)

    assert state["monsters"][0]["current_hp"] == monster_hp_before - 3


def test_burning_blood_should_heal_after_winning_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}],
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._apply_relics_on_combat_end_win(backend.state)

    assert backend.state["player"]["current_hp"] == 56


def test_black_blood_should_heal_twelve_after_winning_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Black Blood"}],
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._apply_relics_on_combat_end_win(backend.state)

    assert backend.state["player"]["current_hp"] == 62


def test_lizard_tail_should_revive_once_instead_of_game_over(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Lizard Tail"}],
        monsters=[make_state.enemy(hp=30)],
    )
    state["player"]["current_hp"] = 1
    state["player"]["max_hp"] = 80

    backend.state = state

    backend._lose_hp(backend.state["player"], backend.state, 10, source="enemy")
    backend._refresh_terminal_flags(backend.state)

    assert backend.state["player"]["current_hp"] == 40
    assert backend.state["game_over"] is False

    backend._lose_hp(backend.state["player"], backend.state, 100, source="enemy")
    backend._refresh_terminal_flags(backend.state)

    assert backend.state["game_over"] is True


def test_calipers_should_reduce_block_by_fifteen_instead_of_zeroing_it(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Calipers"}],
        monsters=[make_state.enemy(intent="DEFEND", intent_base_damage=0)],
        block=40,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.end_turn()

    assert illegal is False
    assert next_state["player"]["block"] == 25
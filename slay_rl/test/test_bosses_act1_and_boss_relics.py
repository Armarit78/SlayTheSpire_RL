from __future__ import annotations


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


def _refresh_boss_with_relics(backend, state):
    state.setdefault("combat_meta", {})
    state["combat_meta"]["is_elite"] = False
    state["combat_meta"]["is_boss"] = True
    state["combat_meta"].setdefault("elite_relic_reward_count", 1)
    state["combat_meta"].setdefault("boss_relic_reward_count", 1)
    state["combat_meta"].setdefault("relic_runtime", {})
    state["combat_meta"].setdefault("cards_played_this_turn", 0)

    backend.state = state
    backend._apply_relics_on_combat_start(backend.state)
    backend._update_enemy_intents(backend.state)
    return backend.state


def _refresh_elite_with_relics(backend, state):
    state.setdefault("combat_meta", {})
    state["combat_meta"]["is_elite"] = True
    state["combat_meta"]["is_boss"] = False
    state["combat_meta"].setdefault("elite_relic_reward_count", 1)
    state["combat_meta"].setdefault("boss_relic_reward_count", 1)
    state["combat_meta"].setdefault("relic_runtime", {})
    state["combat_meta"].setdefault("cards_played_this_turn", 0)

    backend.state = state
    backend._apply_relics_on_combat_start(backend.state)
    backend._update_enemy_intents(backend.state)
    return backend.state


# =========================================================
# Bosses: structure / spawn helpers
# =========================================================

def test_act1_boss_name_resolver_should_return_all_three_bosses(backend):
    assert backend._resolve_act1_boss_names("the_guardian") == ["The Guardian"]
    assert backend._resolve_act1_boss_names("slime_boss") == ["Slime Boss"]
    assert backend._resolve_act1_boss_names("hexaghost") == ["Hexaghost"]


def test_make_enemy_should_build_the_guardian(backend):
    boss = backend._make_enemy("The Guardian")

    assert boss["name"] == "The Guardian"
    assert boss["max_hp"] >= 240
    assert boss["max_hp"] <= 250
    assert _has_power(boss, "Mode Shift", 30)
    assert boss["intent"] == "ATTACK"


def test_make_enemy_should_build_slime_boss(backend):
    boss = backend._make_enemy("Slime Boss")

    assert boss["name"] == "Slime Boss"
    assert boss["max_hp"] >= 140
    assert boss["max_hp"] <= 150
    assert _has_power(boss, "Split", 1)


def test_make_enemy_should_build_hexaghost(backend):
    boss = backend._make_enemy("Hexaghost")

    assert boss["name"] == "Hexaghost"
    assert boss["max_hp"] >= 250
    assert boss["max_hp"] <= 264


# =========================================================
# Boss relics
# =========================================================

def test_philosophers_stone_should_grant_energy_and_enemy_strength_in_boss_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        energy=3,
        relics=[{"name": "Burning Blood"}, {"name": "Philosopher's Stone"}],
        monsters=[make_state.enemy(name="The Guardian", hp=245)],
    )

    state = _refresh_boss_with_relics(backend, state)

    assert state["energy"] == 4
    assert _get_power_amount(state["monsters"][0], "Strength") == 1


def test_slavers_collar_should_grant_energy_in_boss_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        energy=3,
        relics=[{"name": "Burning Blood"}, {"name": "Slaver's Collar"}],
        monsters=[make_state.enemy(name="Hexaghost", hp=260)],
    )

    state = _refresh_boss_with_relics(backend, state)

    assert state["energy"] == 4


def test_slavers_collar_should_also_grant_energy_in_elite_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        energy=3,
        relics=[{"name": "Burning Blood"}, {"name": "Slaver's Collar"}],
        monsters=[make_state.enemy(name="Gremlin Nob", hp=84)],
    )

    state = _refresh_elite_with_relics(backend, state)

    assert state["energy"] == 4


def test_sacred_bark_should_double_block_potion_effect(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Sacred Bark"}],
        potions=[
            {
                "name": "Block Potion",
                "usable": True,
                "empty": False,
                "requires_target": False,
                "rarity": "Common",
            },
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ],
    )
    backend.state = state

    ok = backend._apply_use_potion(backend.state, potion_index=0)

    assert ok is True
    assert backend.state["player"]["block"] == 24


def test_sacred_bark_should_double_strength_potion_effect(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Sacred Bark"}],
        potions=[
            {
                "name": "Strength Potion",
                "usable": True,
                "empty": False,
                "requires_target": False,
                "rarity": "Common",
            },
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ],
    )
    backend.state = state

    ok = backend._apply_use_potion(backend.state, potion_index=0)

    assert ok is True
    assert _get_power_amount(backend.state["player"], "Strength") == 4


def test_velvet_choker_should_block_seventh_card_play_in_same_turn(make_state, backend):
    state = make_state(
        hand=[make_state.card("Anger")],
        relics=[{"name": "Burning Blood"}, {"name": "Velvet Choker"}],
        energy=3,
        monsters=[make_state.enemy(hp=30)],
    )
    state["combat_meta"]["cards_played_this_turn"] = 6
    backend.state = state

    cmd = backend.encode_action_to_command if False else None  # no-op to keep style tools quiet
    out_state, illegal = backend._apply_play_card(
        backend.state,
        type("TmpCmd", (), {"command_type": "play_card", "hand_index": 0, "target_index": 0, "potion_index": None})(),
    )

    assert illegal is True
    assert out_state["combat_meta"]["cards_played_this_turn"] == 6
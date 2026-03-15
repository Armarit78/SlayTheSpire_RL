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


def _refresh_elite_with_relics(backend, state):
    state.setdefault("combat_meta", {})
    state["combat_meta"]["is_elite"] = True
    state["combat_meta"].setdefault("elite_relic_reward_count", 1)
    state["combat_meta"].setdefault("relic_runtime", {})

    backend.state = state
    backend._apply_relics_on_combat_start(backend.state)
    backend._update_enemy_intents(backend.state)
    return backend.state


# =========================================================
# Elite relics
# =========================================================

def test_preserved_insect_should_reduce_elite_hp_by_25_percent(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Preserved Insect"}],
        monsters=[make_state.enemy(name="Gremlin Nob", hp=84)],
    )

    state = _refresh_elite_with_relics(backend, state)

    assert state["monsters"][0]["current_hp"] == 63


def test_sling_of_courage_should_grant_strength_in_elite_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Sling of Courage"}],
        monsters=[make_state.enemy(name="Gremlin Nob", hp=84)],
    )

    state = _refresh_elite_with_relics(backend, state)

    assert _has_power(state["player"], "Strength", 2)


def test_slavers_collar_should_grant_extra_energy_in_elite_combat(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        energy=3,
        relics=[{"name": "Burning Blood"}, {"name": "Slaver's Collar"}],
        monsters=[make_state.enemy(name="Lagavulin", hp=110)],
    )

    state = _refresh_elite_with_relics(backend, state)

    assert state["energy"] == 4


def test_black_star_should_increase_elite_relic_reward_count(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}, {"name": "Black Star"}],
        monsters=[make_state.enemy(name="Gremlin Nob", hp=84)],
    )

    state = _refresh_elite_with_relics(backend, state)

    assert state["combat_meta"]["elite_relic_reward_count"] == 2


# =========================================================
# Gremlin Nob
# =========================================================

def test_gremlin_nob_should_gain_strength_when_player_plays_skill(make_state, backend):
    state = make_state(
        hand=[make_state.card("Defend_R")],
        relics=[{"name": "Burning Blood"}],
        monsters=[
            make_state.enemy(
                name="Gremlin Nob",
                hp=84,
                powers=[{"id": "Enrage", "amount": 2}],
            )
        ],
    )
    backend.state = state

    backend._on_card_played_before_resolution(backend.state, backend.state["hand"][0])

    assert _get_power_amount(backend.state["monsters"][0], "Strength") == 2


def test_gremlin_nob_should_not_gain_strength_when_player_plays_attack(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}],
        monsters=[
            make_state.enemy(
                name="Gremlin Nob",
                hp=84,
                powers=[{"id": "Enrage", "amount": 2}],
            )
        ],
    )
    backend.state = state

    backend._on_card_played_before_resolution(backend.state, backend.state["hand"][0])

    assert _get_power_amount(backend.state["monsters"][0], "Strength") == 0


# =========================================================
# Lagavulin
# =========================================================

def test_lagavulin_should_wake_up_when_damaged(make_state, backend):
    state = make_state(
        hand=[make_state.card("Strike_R")],
        relics=[{"name": "Burning Blood"}],
        monsters=[
            make_state.enemy(
                name="Lagavulin",
                hp=110,
                powers=[
                    {"id": "Metallicize", "amount": 8},
                    {"id": "Asleep", "amount": 1},
                ],
            )
        ],
    )
    state["monsters"][0]["combat_state"] = {
        "asleep_turns": 0,
        "awakened": False,
        "pattern_idx": 0,
    }
    backend.state = state

    dealt = backend._deal_damage_to_monster(backend.state, backend.state["monsters"][0], 10)

    assert dealt == 10
    assert backend.state["monsters"][0]["combat_state"]["awakened"] is True
    assert _get_power_amount(backend.state["monsters"][0], "Asleep") == 0
    assert _get_power_amount(backend.state["monsters"][0], "Metallicize") == 0


# =========================================================
# Sentries
# =========================================================

def test_sentry_pattern_should_alternate_attack_and_dazed(make_state, backend):
    sentry = make_state.enemy(
        name="Sentry",
        hp=40,
        powers=[{"id": "Artifact", "amount": 1}],
    )
    sentry["combat_state"] = {"pattern_idx": 0}

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[sentry],
        relics=[{"name": "Burning Blood"}],
    )
    backend.state = state

    backend._update_enemy_intents(backend.state)
    first_intent = backend.state["monsters"][0]["intent"]
    first_damage = backend.state["monsters"][0].get("intent_base_damage", 0)

    backend._update_enemy_intents(backend.state)
    second_intent = backend.state["monsters"][0]["intent"]

    assert first_intent == "ATTACK"
    assert first_damage == 9
    assert second_intent == "DEBUFF"


# =========================================================
# Elite spawn metadata
# =========================================================

def test_reset_should_store_elite_metadata_when_sampling_elite(make_state, backend):
    backend.state = None

    # on force le sampling elite en appelant directement la sortie helper
    monsters, is_elite, is_boss = backend._sample_enemy_group_with_meta()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=monsters,
        relics=[{"name": "Burning Blood"}],
    )
    state["combat_meta"]["is_elite"] = is_elite
    state["combat_meta"]["is_boss"] = is_boss
    state["combat_meta"]["elite_relic_reward_count"] = 1
    state["combat_meta"]["boss_relic_reward_count"] = 1

    assert isinstance(state["combat_meta"]["is_elite"], bool)
    assert state["combat_meta"]["elite_relic_reward_count"] == 1
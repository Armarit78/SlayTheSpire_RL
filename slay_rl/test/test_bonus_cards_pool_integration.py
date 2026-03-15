from __future__ import annotations

import pytest


CURRENT_BONUS_POOL = [
    # Attacks / basics
    "Pommel Strike", "Anger", "Twin Strike", "Headbutt", "Iron Wave",
    "Cleave", "Clothesline", "Uppercut", "Carnage", "Pummel",
    "Whirlwind", "Sword Boomerang", "Perfected Strike", "Heavy Blade",
    "Wild Strike", "Thunderclap", "Dropkick", "Hemokinesis", "Rampage",

    # Skills / block / setup
    "Shrug It Off", "True Grit", "Warcry", "Armaments", "Battle Trance",
    "Ghostly Armor", "Power Through", "Flame Barrier", "Disarm",
    "Second Wind", "Entrench", "Seeing Red", "Bloodletting", "Burning Pact",

    # Powers / scaling
    "Inflame", "Metallicize", "Combust", "Feel No Pain",
    "Dark Embrace", "Rupture", "Fire Breathing", "Brutality",

    # High impact / already coded
    "Shockwave", "Offering", "Immolate", "Fiend Fire", "Reaper",
]

TARGETED_BONUS_CARDS = {
    "Pommel Strike",
    "Anger",
    "Twin Strike",
    "Headbutt",
    "Iron Wave",
    "Clothesline",
    "Uppercut",
    "Carnage",
    "Pummel",
    "Sword Boomerang",
    "Perfected Strike",
    "Heavy Blade",
    "Wild Strike",
    "Dropkick",
    "Hemokinesis",
    "Rampage",
    "Disarm",
    "Fiend Fire",
}

NEEDS_EXTRA_HAND_CARD = {
    "True Grit",
    "Warcry",
    "Armaments",
    "Second Wind",
    "Burning Pact",
    "Fiend Fire",
}


def _power_amount(entity, power_id: str) -> int:
    for p in entity.get("powers", []):
        if p.get("id") == power_id:
            return int(p.get("amount", 0))
    return 0


def test_sample_bonus_cards_should_only_return_cards_from_current_bonus_pool(backend):
    cards = backend._sample_bonus_cards(k=len(CURRENT_BONUS_POOL))
    ids = [c["id"] for c in cards]

    assert len(ids) == len(set(ids))
    assert set(ids).issubset(set(CURRENT_BONUS_POOL))


@pytest.mark.parametrize("card_id", CURRENT_BONUS_POOL)
def test_bonus_card_should_be_buildable_from_make_card(make_state, card_id):
    card = make_state.card(card_id)

    assert card["id"] == card_id
    assert "cost" in card
    assert "type" in card


@pytest.mark.parametrize("card_id", CURRENT_BONUS_POOL)
def test_bonus_card_should_be_playable_without_crashing(make_state, step_helpers, card_id):
    hand = [make_state.card(card_id)]

    if card_id in NEEDS_EXTRA_HAND_CARD:
        hand.append(make_state.card("Strike_R"))

    draw_pile = [
        make_state.card("Strike_R"),
        make_state.card("Defend_R"),
        make_state.card("Bash"),
        make_state.card("Anger"),
        make_state.card("Shrug It Off"),
    ]

    state = make_state(
        hand=hand,
        draw_pile=draw_pile,
        monsters=[make_state.enemy(hp=80)],
        energy=6,
    )
    step_helpers.set_state(state)

    if card_id in TARGETED_BONUS_CARDS:
        next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)
    else:
        next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False, f"{card_id} was illegal to play"
    assert isinstance(next_state, dict)
    assert "player" in next_state
    assert "monsters" in next_state


def test_true_grit_should_exhaust_another_card_from_hand(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("True Grit"), make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    exhausted_ids = [c["id"] for c in next_state["exhaust_pile"]]
    assert "Strike_R" in exhausted_ids


def test_warcry_should_not_crash_and_should_cost_zero_energy(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Warcry"), make_state.card("Strike_R")],
        draw_pile=[make_state.card("Defend_R")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] == 3


def test_armaments_should_grant_block_when_played(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Armaments"), make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["block"] >= 5


def test_battle_trance_should_draw_cards(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Battle Trance")],
        draw_pile=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
            make_state.card("Anger"),
        ],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert len(next_state["hand"]) >= 3


def test_shockwave_should_apply_weak_and_vulnerable_to_all_enemies(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Shockwave")],
        monsters=[make_state.enemy(hp=40), make_state.enemy(name="Cultist", hp=40)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    for monster in next_state["monsters"]:
        powers = {p["id"]: int(p["amount"]) for p in monster.get("powers", [])}
        assert powers.get("Weak", 0) >= 3
        assert powers.get("Vulnerable", 0) >= 3


def test_pummel_should_exhaust_itself_after_play(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Pummel")],
        monsters=[make_state.enemy(hp=60)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    exhausted_ids = [c["id"] for c in next_state["exhaust_pile"]]
    assert "Pummel" in exhausted_ids


def test_whirlwind_should_spend_available_energy(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Whirlwind")],
        monsters=[make_state.enemy(hp=50), make_state.enemy(name="Cultist", hp=50)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] == 0


def test_power_through_should_add_wounds_to_hand(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Power Through")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    wound_count = sum(1 for c in next_state["hand"] if c["id"] == "Wound")
    assert wound_count >= 2


def test_flame_barrier_should_grant_block_and_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Flame Barrier")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["block"] >= 12
    assert _power_amount(next_state["player"], "Flame Barrier") >= 4


def test_seeing_red_should_increase_energy(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Seeing Red")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["energy"] >= 4


def test_bloodletting_should_lose_hp_and_gain_energy(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Bloodletting")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["current_hp"] == 47
    assert next_state["energy"] >= 5


def test_inflame_should_gain_strength(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Inflame")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Strength") >= 2


def test_metallicize_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Metallicize")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Metallicize") >= 3


def test_combust_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Combust")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Combust") >= 5


def test_feel_no_pain_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Feel No Pain")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Feel No Pain") >= 3


def test_dark_embrace_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Dark Embrace")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Dark Embrace") >= 1


def test_rupture_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Rupture")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Rupture") >= 1


def test_fire_breathing_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Fire Breathing")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Fire Breathing") >= 6


def test_brutality_should_apply_power(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Brutality")],
        monsters=[make_state.enemy()],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert _power_amount(next_state["player"], "Brutality") >= 1


def test_offering_should_lose_hp_gain_energy_and_draw(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Offering")],
        draw_pile=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
            make_state.card("Anger"),
        ],
        monsters=[make_state.enemy()],
        energy=3,
    )
    state["player"]["current_hp"] = 50
    state["player"]["max_hp"] = 80
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["player"]["current_hp"] == 44
    assert next_state["energy"] >= 5
    assert len(next_state["hand"]) >= 3


def test_immolate_should_damage_all_and_add_burn_to_discard(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Immolate")],
        monsters=[make_state.enemy(hp=50), make_state.enemy(name="Cultist", hp=50)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] <= 29
    assert next_state["monsters"][1]["current_hp"] <= 29
    discard_ids = [c["id"] for c in next_state["discard_pile"]]
    assert "Burn" in discard_ids


def test_fiend_fire_should_exhaust_other_cards(make_state, step_helpers):
    state = make_state(
        hand=[
            make_state.card("Fiend Fire"),
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
        ],
        monsters=[make_state.enemy(hp=100)],
        energy=3,
    )
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0, target_index=0)

    assert illegal is False
    exhausted_ids = [c["id"] for c in next_state["exhaust_pile"]]
    assert "Strike_R" in exhausted_ids
    assert "Defend_R" in exhausted_ids


def test_reaper_should_damage_all_and_heal_player(make_state, step_helpers):
    state = make_state(
        hand=[make_state.card("Reaper")],
        monsters=[make_state.enemy(hp=20), make_state.enemy(name="Cultist", hp=20)],
        energy=3,
    )
    state["player"]["current_hp"] = 40
    state["player"]["max_hp"] = 80
    step_helpers.set_state(state)

    next_state, illegal = step_helpers.play_card(hand_index=0)

    assert illegal is False
    assert next_state["monsters"][0]["current_hp"] <= 16
    assert next_state["monsters"][1]["current_hp"] <= 16
    assert next_state["player"]["current_hp"] >= 41
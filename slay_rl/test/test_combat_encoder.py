from __future__ import annotations

import pytest

from slay_rl.config import CARD_TO_IDX, RELIC_TO_IDX, get_default_config
from slay_rl.features.combat_encoder import CombatEncoder, flatten_combat_obs


def _layout():
    cfg = get_default_config()
    max_hand = cfg.combat_obs.max_hand_cards
    max_enemies = cfg.combat_obs.max_enemies
    max_potions = cfg.combat_obs.max_potions

    max_choose_hand = cfg.combat_action.max_choose_hand_actions
    max_choose_option = cfg.combat_action.max_choose_option_actions
    max_choose_discard = cfg.combat_action.max_choose_discard_actions
    max_choose_exhaust = cfg.combat_action.max_choose_exhaust_actions

    targeted_base = max_hand
    targeted_size = max_hand * max_enemies
    end_turn_idx = targeted_base + targeted_size

    potion_base = end_turn_idx + 1
    potion_target_base = potion_base + max_potions
    potion_target_size = max_potions * max_enemies

    choose_hand_base = potion_target_base + potion_target_size
    choose_option_base = choose_hand_base + max_choose_hand
    choose_discard_base = choose_option_base + max_choose_option
    choose_exhaust_base = choose_discard_base + max_choose_discard

    return {
        "cfg": cfg,
        "end_turn_idx": end_turn_idx,
        "targeted_base": targeted_base,
        "max_enemies": max_enemies,
        "choose_hand_base": choose_hand_base,
        "choose_option_base": choose_option_base,
        "choose_discard_base": choose_discard_base,
        "choose_exhaust_base": choose_exhaust_base,
    }


def test_encoder_output_shapes_masks_and_new_blocks(make_state):
    cfg = get_default_config()
    encoder = CombatEncoder(cfg)

    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Bash")],
        draw_pile=[make_state.card("Defend_R")],
        discard_pile=[make_state.card("Burn")],
        exhaust_pile=[make_state.card("Offering")],
        monsters=[
            make_state.enemy(hp=40, intent="ATTACK", intent_base_damage=11),
            make_state.enemy(name="Cultist", hp=35, intent="BUFF", intent_base_damage=0),
        ],
        relics=[{"name": "Burning Blood"}, {"name": "Anchor"}],
        player_powers=[{"id": "Strength", "amount": 2}, {"id": "Artifact", "amount": 1}],
        potions=[
            {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Dexterity Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Uncommon"},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ],
        combat_meta={
            "cards_played_this_turn": 1,
            "attacks_played_this_turn": 1,
            "double_tap_charges": 1,
            "cannot_draw_more_this_turn": False,
            "last_x_energy_spent": 0,
            "attack_counter": 1,
            "next_attack_double": False,
            "first_attack_done": True,
            "is_elite": True,
            "is_boss": False,
        },
        energy=3,
    )

    encoded = encoder.encode(state)

    assert encoded["player_scalars"].shape == (cfg.combat_obs.player_scalar_dim,)
    assert encoded["hand_cards"].shape == (cfg.combat_obs.max_hand_cards, encoder.card_feature_dim)
    assert encoded["hand_mask"].shape == (cfg.combat_obs.max_hand_cards,)
    assert encoded["enemies"].shape == (cfg.combat_obs.max_enemies, encoder.enemy_feature_dim)
    assert encoded["enemy_mask"].shape == (cfg.combat_obs.max_enemies,)
    assert encoded["potions"].shape == (cfg.combat_obs.max_potions, encoder.potion_feature_dim)
    assert encoded["potion_mask"].shape == (cfg.combat_obs.max_potions,)
    assert encoded["combat_context"].shape == (cfg.combat_obs.combat_context_dim,)
    assert encoded["deck_counts"].shape == (cfg.combat_obs.card_vocab_size,)
    assert encoded["discard_counts"].shape == (cfg.combat_obs.card_vocab_size,)
    assert encoded["exhaust_counts"].shape == (cfg.combat_obs.card_vocab_size,)
    assert encoded["relics"].shape == (cfg.combat_obs.relic_vocab_size,)
    assert encoded["valid_action_mask"].shape == (cfg.combat_action.total_actions,)

    assert encoded["hand_mask"].tolist()[:5] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert encoded["enemy_mask"].tolist()[:5] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert encoded["potion_mask"].tolist()[:5] == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_encoder_counts_cards_in_piles_and_relics(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[],
        draw_pile=[make_state.card("Strike_R"), make_state.card("Strike_R"), make_state.card("Defend_R")],
        discard_pile=[make_state.card("Burn")],
        exhaust_pile=[make_state.card("Offering")],
        relics=[{"name": "Burning Blood"}, {"name": "Anchor"}],
        monsters=[make_state.enemy()],
    )

    encoded = encoder.encode(state)

    strike_idx = CARD_TO_IDX["Strike_R"]
    defend_idx = CARD_TO_IDX["Defend_R"]
    burn_idx = CARD_TO_IDX["Burn"]
    offering_idx = CARD_TO_IDX["Offering"]
    burning_blood_idx = RELIC_TO_IDX["Burning Blood"]
    anchor_idx = RELIC_TO_IDX["Anchor"]

    assert encoded["deck_counts"][strike_idx].item() == pytest.approx(2 / 8)
    assert encoded["deck_counts"][defend_idx].item() == pytest.approx(1 / 8)
    assert encoded["discard_counts"][burn_idx].item() == pytest.approx(1 / 8)
    assert encoded["exhaust_counts"][offering_idx].item() == pytest.approx(1 / 8)
    assert encoded["relics"][burning_blood_idx].item() == 1.0
    assert encoded["relics"][anchor_idx].item() == 1.0


def test_encoder_marks_unplayable_and_target_features(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Wound"), make_state.card("Bash")],
        monsters=[make_state.enemy()],
        energy=1,
    )

    encoded = encoder.encode(state)
    hand_cards = encoded["hand_cards"]

    wound_row = hand_cards[0]
    bash_row = hand_cards[1]

    assert wound_row[-2].item() == 0.0
    assert wound_row[-1].item() == 0.0
    assert bash_row[-2].item() == 0.0
    assert bash_row[-1].item() == 1.0


def test_encoder_player_scalars_reflect_key_powers_and_threat(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[
            make_state.enemy(hp=40, intent="ATTACK", intent_base_damage=12),
            make_state.enemy(name="Cultist", hp=35, intent="BUFF", intent_base_damage=0),
        ],
        hp=60,
        max_hp=80,
        block=10,
        energy=2,
        player_powers=[
            {"id": "Strength", "amount": 3},
            {"id": "Dexterity", "amount": 1},
            {"id": "Artifact", "amount": 2},
            {"id": "Rage", "amount": 3},
            {"id": "Barricade", "amount": 1},
            {"id": "Corruption", "amount": 1},
        ],
        combat_meta={"double_tap_charges": 1},
    )

    encoded = encoder.encode(state)
    player = encoded["player_scalars"]

    assert player.shape[0] == encoder.cfg.combat_obs.player_scalar_dim
    assert player[0].item() == pytest.approx(60 / 80)
    assert player[2].item() == pytest.approx(10 / 120)
    assert player[3].item() == pytest.approx(2 / 10)
    assert player[4].item() > 0.0
    assert player[5].item() > 0.0
    assert player[9].item() > 0.0
    assert player[12].item() > 0.0
    assert player[20].item() == 1.0
    assert player[21].item() == 1.0
    assert player[22].item() > 0.0
    assert player[23].item() > 0.0
    assert player[24].item() > 0.0


def test_encoder_enemy_features_include_multi_hit_and_enemy_powers(make_state):
    encoder = CombatEncoder()

    enemy = make_state.enemy(
        name="Jaw Worm",
        hp=30,
        block=6,
        intent="ATTACK",
        intent_base_damage=7,
        powers=[
            {"id": "Strength", "amount": 2},
            {"id": "Artifact", "amount": 1},
            {"id": "Metallicize", "amount": 3},
            {"id": "Ritual", "amount": 4},
        ],
    )
    enemy["intent_hits"] = 2

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[enemy],
    )

    encoded = encoder.encode(state)
    enemy_row = encoded["enemies"][0]

    assert enemy_row.shape[0] == encoder.enemy_feature_dim
    assert enemy_row.sum().item() > 0.0
    assert enemy_row[-7].item() > 0.0
    assert enemy_row[-4].item() > 0.0
    assert enemy_row[-3].item() > 0.0
    assert enemy_row[-1].item() > 0.0


def test_encoder_potions_block_is_present_and_masks_empty_slots(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
        potions=[
            {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Dexterity Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Uncommon"},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ],
    )

    encoded = encoder.encode(state)
    potions = encoded["potions"]
    potion_mask = encoded["potion_mask"]

    assert potions.shape[1] == encoder.potion_feature_dim
    assert potion_mask.tolist()[:5] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert potions[0].sum().item() > 0.0
    assert potions[1].sum().item() > 0.0
    assert potion_mask[2].item() == 0.0
    assert potions[2].sum().item() > 0.0


def test_encoder_combat_context_tracks_turn_meta(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R"), make_state.card("Inflame")],
        monsters=[make_state.enemy(hp=40, intent="ATTACK", intent_base_damage=10)],
        energy=2,
        turn=3,
        combat_meta={
            "cards_played_this_turn": 2,
            "attacks_played_this_turn": 1,
            "attack_counter": 2,
            "double_tap_charges": 1,
            "last_x_energy_spent": 1,
            "cannot_draw_more_this_turn": True,
            "next_attack_double": True,
            "first_attack_done": True,
            "is_elite": True,
            "is_boss": False,
        },
    )

    encoded = encoder.encode(state)
    ctx = encoded["combat_context"]

    assert ctx.shape[0] == encoder.cfg.combat_obs.combat_context_dim
    assert ctx[0].item() == pytest.approx(3 / 20)
    assert ctx[1].item() == pytest.approx(2 / 10)
    assert ctx[2].item() == pytest.approx(1 / 10)
    assert ctx[4].item() > 0.0
    assert ctx[5].item() > 0.0
    assert ctx[6].item() == 1.0
    assert ctx[7].item() == 1.0
    assert ctx[8].item() == 1.0
    assert ctx[9].item() == 1.0
    assert ctx[10].item() == 0.0


def test_flatten_combat_obs_includes_new_blocks(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
        potions=[{"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"}]
        + [{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
        combat_meta={"is_elite": True},
    )

    encoded = encoder.encode(state)
    flat = flatten_combat_obs(encoded)

    expected = (
        encoded["player_scalars"].numel()
        + encoded["hand_cards"].numel()
        + encoded["hand_mask"].numel()
        + encoded["enemies"].numel()
        + encoded["enemy_mask"].numel()
        + encoded["potions"].numel()
        + encoded["potion_mask"].numel()
        + encoded["combat_context"].numel()
        + encoded["deck_counts"].numel()
        + encoded["discard_counts"].numel()
        + encoded["exhaust_counts"].numel()
        + encoded["relics"].numel()
    )

    assert flat.ndim == 1
    assert flat.numel() == expected


def test_encoder_marks_dead_enemy_slots_as_not_targetable(make_state):
    encoder = CombatEncoder()
    L = _layout()

    dead_enemy = make_state.enemy(hp=0, intent="ATTACK", intent_base_damage=8)
    dead_enemy["isDead"] = True

    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[dead_enemy, make_state.enemy(name="Cultist", hp=25, intent="BUFF", intent_base_damage=0)],
        energy=3,
    )

    encoded = encoder.encode(state)

    assert encoded["enemy_mask"].tolist()[:5] == [0.0, 1.0, 0.0, 0.0, 0.0]

    mask = encoded["valid_action_mask"].tolist()
    bash_t0 = L["targeted_base"] + 0 * L["max_enemies"] + 0
    bash_t1 = L["targeted_base"] + 0 * L["max_enemies"] + 1

    assert mask[bash_t0] == 0.0
    assert mask[bash_t1] == 1.0


def test_encoder_player_scalars_capture_zero_incoming_damage_state(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30, intent="BUFF", intent_base_damage=0)],
        hp=50,
        max_hp=80,
        block=0,
        energy=1,
        player_powers=[],
        combat_meta={"double_tap_charges": 0},
    )

    encoded = encoder.encode(state)
    player = encoded["player_scalars"]

    assert player[23].item() == pytest.approx(0.0)


def test_encoder_combat_context_distinguishes_elite_and_boss_flags(make_state):
    encoder = CombatEncoder()

    elite_state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=40)],
        combat_meta={"is_elite": True, "is_boss": False},
    )
    boss_state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=150)],
        combat_meta={"is_elite": False, "is_boss": True},
    )

    elite_ctx = encoder.encode(elite_state)["combat_context"]
    boss_ctx = encoder.encode(boss_state)["combat_context"]

    assert elite_ctx[9].item() == 1.0
    assert elite_ctx[10].item() == 0.0
    assert boss_ctx[9].item() == 0.0
    assert boss_ctx[10].item() == 1.0


def test_encoder_hand_mask_zeroes_unused_slots(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R")],
        monsters=[make_state.enemy()],
    )

    encoded = encoder.encode(state)
    hand_mask = encoded["hand_mask"].tolist()

    assert hand_mask[0] == 1.0
    assert hand_mask[1] == 1.0
    assert hand_mask[2:] == [0.0] * (len(hand_mask) - 2)


def test_encoder_choice_option_mask_disables_normal_combat_actions(make_state):
    encoder = CombatEncoder()
    L = _layout()

    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=40)],
        energy=2,
    )
    state["pending_choice"] = {
        "choice_type": "choose_option",
        "options": ["Anger", "Shrug It Off"],
    }

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[0] == 0.0
    assert mask[L["targeted_base"]] == 0.0
    assert mask[L["end_turn_idx"]] == 0.0
    assert mask[L["choose_option_base"] + 0] == 1.0
    assert mask[L["choose_option_base"] + 1] == 1.0


def test_encoder_choice_hand_defaults_to_existing_hand_when_indices_missing(make_state):
    encoder = CombatEncoder()
    L = _layout()

    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R"), make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_hand_card",
    }

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[L["choose_hand_base"] + 0] == 1.0
    assert mask[L["choose_hand_base"] + 1] == 1.0
    assert mask[L["choose_hand_base"] + 2] == 1.0
    assert mask[L["choose_hand_base"] + 3] == 0.0


def test_encoder_choice_type_aliases_are_normalized(make_state):
    encoder = CombatEncoder()

    assert encoder._normalize_choice_type({"choice_type": "discard"}) == "choose_discard_target"
    assert encoder._normalize_choice_type({"choice_type": "exhaust"}) == "choose_exhaust_target"
    assert encoder._normalize_choice_type({"choice_type": "option"}) == "choose_option"
    assert encoder._normalize_choice_type({"choice_type": "hand_card"}) == "choose_hand_card"
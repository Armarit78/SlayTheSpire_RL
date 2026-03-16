from __future__ import annotations

from slay_rl.features.combat_encoder import CombatEncoder
from slay_rl.agents.combat_agent import encode_command_to_action_index, CombatCommand
from slay_rl.config import get_default_config


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
        "max_hand": max_hand,
        "max_enemies": max_enemies,
        "max_potions": max_potions,
        "max_choose_hand": max_choose_hand,
        "max_choose_option": max_choose_option,
        "max_choose_discard": max_choose_discard,
        "max_choose_exhaust": max_choose_exhaust,
        "targeted_base": targeted_base,
        "end_turn_idx": end_turn_idx,
        "potion_base": potion_base,
        "potion_target_base": potion_target_base,
        "choose_hand_base": choose_hand_base,
        "choose_option_base": choose_option_base,
        "choose_discard_base": choose_discard_base,
        "choose_exhaust_base": choose_exhaust_base,
    }


def test_action_mask_marks_playable_non_target_card_and_end_turn(make_state):
    encoder = CombatEncoder()
    L = _layout()

    state = make_state(
        hand=[make_state.card("Defend_R"), make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40)],
        energy=1,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[0] == 1.0
    assert mask[1] == 0.0
    assert mask[L["end_turn_idx"]] == 1.0


def test_action_mask_marks_targeted_card_only_for_living_targets(make_state):
    encoder = CombatEncoder()
    L = _layout()

    dead_enemy = make_state.enemy(hp=0)
    dead_enemy["isDead"] = True
    state = make_state(
        hand=[make_state.card("Bash")],
        monsters=[make_state.enemy(hp=40), dead_enemy],
        energy=3,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    bash_t0 = L["targeted_base"] + 0 * L["max_enemies"] + 0
    bash_t1 = L["targeted_base"] + 0 * L["max_enemies"] + 1
    assert mask[bash_t0] == 1.0
    assert mask[bash_t1] == 0.0


def test_action_mask_marks_potions_in_correct_slots(make_state):
    encoder = CombatEncoder()
    L = _layout()

    potions = [
        {"name": "Explosive Potion", "usable": True, "empty": False, "requires_target": False},
        {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(3)],
    ]
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=40), make_state.enemy(name="Cultist", hp=35)],
        potions=potions,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[L["potion_base"] + 0] == 1.0
    assert mask[L["potion_base"] + 1] == 0.0
    assert mask[L["potion_target_base"] + 1 * L["max_enemies"] + 0] == 1.0
    assert mask[L["potion_target_base"] + 1 * L["max_enemies"] + 1] == 1.0


def test_encode_command_to_action_index_round_trip_layout():
    L = _layout()

    non_target = encode_command_to_action_index(CombatCommand("play_card", hand_index=2, target_index=None))
    targeted = encode_command_to_action_index(CombatCommand("play_card", hand_index=3, target_index=4))
    end_turn = encode_command_to_action_index(CombatCommand("end_turn"))
    potion = encode_command_to_action_index(CombatCommand("use_potion", potion_index=1, target_index=None))
    targeted_potion = encode_command_to_action_index(CombatCommand("use_potion", potion_index=2, target_index=3))

    assert non_target == 2
    assert targeted == L["targeted_base"] + 3 * L["max_enemies"] + 4
    assert end_turn == L["end_turn_idx"]
    assert potion == L["potion_base"] + 1
    assert targeted_potion == L["potion_target_base"] + 2 * L["max_enemies"] + 3


def test_action_mask_disables_end_turn_only_when_no_general_rule_forbids_it(make_state):
    encoder = CombatEncoder()
    L = _layout()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=30)],
        energy=1,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[L["end_turn_idx"]] == 1.0


def test_action_mask_marks_only_living_targets_for_targeted_potion(make_state):
    encoder = CombatEncoder()
    L = _layout()

    potions = [
        {"name": "Fear Potion", "usable": True, "empty": False, "requires_target": True},
        *[{"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False} for _ in range(4)],
    ]

    dead_enemy = make_state.enemy(hp=0)
    dead_enemy["isDead"] = True

    state = make_state(
        hand=[],
        monsters=[make_state.enemy(hp=40), dead_enemy],
        potions=potions,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    fear_t0 = L["potion_target_base"] + 0 * L["max_enemies"] + 0
    fear_t1 = L["potion_target_base"] + 0 * L["max_enemies"] + 1

    assert mask[L["potion_base"] + 0] == 0.0
    assert mask[fear_t0] == 1.0
    assert mask[fear_t1] == 0.0


def test_action_mask_disables_unplayable_hand_slots_beyond_current_hand(make_state):
    encoder = CombatEncoder()

    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy(hp=40)],
        energy=1,
    )

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[1] == 0.0
    assert mask[2] == 0.0
    assert mask[9] == 0.0


def test_action_mask_choice_option_activates_only_option_slots(make_state):
    encoder = CombatEncoder()
    L = _layout()

    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Defend_R")],
        monsters=[make_state.enemy(hp=30)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_option",
        "options": ["Anger", "Pommel Strike", "Shrug It Off"],
    }

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[L["choose_option_base"] + 0] == 1.0
    assert mask[L["choose_option_base"] + 1] == 1.0
    assert mask[L["choose_option_base"] + 2] == 1.0
    assert mask[L["choose_option_base"] + 3] == 0.0

    assert sum(mask[: L["end_turn_idx"] + 1]) == 0.0
    assert sum(mask[L["choose_hand_base"]:L["choose_option_base"]]) == 0.0
    assert sum(mask[L["choose_discard_base"]:L["choose_exhaust_base"] + L["max_choose_exhaust"]]) == 0.0


def test_action_mask_choice_discard_activates_only_valid_hand_indices(make_state):
    encoder = CombatEncoder()
    L = _layout()

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
        "valid_hand_indices": [0, 2],
    }

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[L["choose_discard_base"] + 0] == 1.0
    assert mask[L["choose_discard_base"] + 1] == 0.0
    assert mask[L["choose_discard_base"] + 2] == 1.0
    assert mask[L["choose_discard_base"] + 3] == 0.0

    assert sum(mask[: L["end_turn_idx"] + 1]) == 0.0


def test_action_mask_choice_exhaust_activates_only_valid_hand_indices(make_state):
    encoder = CombatEncoder()
    L = _layout()

    state = make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Defend_R"),
            make_state.card("Bash"),
            make_state.card("Inflame"),
        ],
        monsters=[make_state.enemy(hp=30)],
    )
    state["pending_choice"] = {
        "choice_type": "choose_exhaust_target",
        "valid_hand_indices": [1, 3],
    }

    encoded = encoder.encode(state)
    mask = encoded["valid_action_mask"].tolist()

    assert mask[L["choose_exhaust_base"] + 0] == 0.0
    assert mask[L["choose_exhaust_base"] + 1] == 1.0
    assert mask[L["choose_exhaust_base"] + 2] == 0.0
    assert mask[L["choose_exhaust_base"] + 3] == 1.0


def test_encode_command_to_action_index_round_trip_layout_for_choose_actions():
    L = _layout()

    choose_hand = encode_command_to_action_index(CombatCommand("choose_hand_card", hand_index=4))
    choose_option = encode_command_to_action_index(CombatCommand("choose_option", target_index=2))
    choose_discard = encode_command_to_action_index(CombatCommand("choose_discard_target", hand_index=7))
    choose_exhaust = encode_command_to_action_index(CombatCommand("choose_exhaust_target", hand_index=1))

    assert choose_hand == L["choose_hand_base"] + 4
    assert choose_option == L["choose_option_base"] + 2
    assert choose_discard == L["choose_discard_base"] + 7
    assert choose_exhaust == L["choose_exhaust_base"] + 1
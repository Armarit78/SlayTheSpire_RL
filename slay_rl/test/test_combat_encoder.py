from __future__ import annotations

import pytest

from slay_rl.config import CARD_TO_IDX, RELIC_TO_IDX
from slay_rl.features.combat_encoder import CombatEncoder, flatten_combat_obs



def test_encoder_output_shapes_and_masks(make_state):
    encoder = CombatEncoder()
    state = make_state(
        hand=[make_state.card("Strike_R"), make_state.card("Bash")],
        draw_pile=[make_state.card("Defend_R")],
        discard_pile=[make_state.card("Burn")],
        exhaust_pile=[make_state.card("Offering")],
        monsters=[make_state.enemy(hp=40), make_state.enemy(name="Cultist", hp=35)],
        relics=[{"name": "Burning Blood"}, {"name": "Anchor"}],
        player_powers=[{"id": "Strength", "amount": 2}],
        energy=3,
    )

    encoded = encoder.encode(state)

    assert encoded["player_scalars"].shape == (8,)
    assert encoded["hand_cards"].shape[0] == 10
    assert encoded["enemies"].shape[0] == 5
    assert encoded["hand_mask"].tolist()[:4] == [1.0, 1.0, 0.0, 0.0]
    assert encoded["enemy_mask"].tolist()[:4] == [1.0, 1.0, 0.0, 0.0]
    assert encoded["valid_action_mask"].shape == (91,)



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

    assert encoded["deck_counts"][strike_idx].item() == pytest.approx(2 / 3)
    assert encoded["deck_counts"][defend_idx].item() == pytest.approx(1 / 3)
    assert encoded["discard_counts"][burn_idx].item() == 1.0
    assert encoded["exhaust_counts"][offering_idx].item() == 1.0
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

    assert wound_row[-2].item() == 0.0  # is_playable
    assert wound_row[-1].item() == 0.0  # has_target
    assert bash_row[-2].item() == 0.0   # not enough energy
    assert bash_row[-1].item() == 1.0   # targeted attack



def test_flatten_combat_obs_returns_single_vector(make_state):
    encoder = CombatEncoder()
    state = make_state(
        hand=[make_state.card("Strike_R")],
        monsters=[make_state.enemy()],
    )

    encoded = encoder.encode(state)
    flat = flatten_combat_obs(encoded)

    expected = (
        encoded["player_scalars"].numel()
        + encoded["hand_cards"].numel()
        + encoded["hand_mask"].numel()
        + encoded["enemies"].numel()
        + encoded["enemy_mask"].numel()
        + encoded["deck_counts"].numel()
        + encoded["discard_counts"].numel()
        + encoded["exhaust_counts"].numel()
        + encoded["relics"].numel()
    )

    assert flat.ndim == 1
    assert flat.numel() == expected
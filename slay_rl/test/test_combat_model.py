from __future__ import annotations

import torch

from slay_rl.models.combat_model import CombatModel, compute_ppo_loss, stack_encoded_obs


def _sample_state(make_state):
    return make_state(
        hand=[
            make_state.card("Strike_R"),
            make_state.card("Bash"),
            make_state.card("Defend_R"),
            make_state.card("Inflame"),
        ],
        monsters=[
            make_state.enemy(hp=35, intent="ATTACK", intent_base_damage=10),
            make_state.enemy(name="Cultist", hp=48, intent="BUFF", intent_base_damage=0),
        ],
        potions=[
            {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Dexterity Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Uncommon"},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ],
        player_powers=[
            {"id": "Strength", "amount": 2},
            {"id": "Rage", "amount": 3},
        ],
        combat_meta={
            "cards_played_this_turn": 1,
            "attacks_played_this_turn": 1,
            "double_tap_charges": 0,
            "cannot_draw_more_this_turn": False,
            "last_x_energy_spent": 0,
            "attack_counter": 1,
            "next_attack_double": False,
            "first_attack_done": True,
            "is_elite": False,
            "is_boss": False,
        },
        energy=3,
    )


def test_combat_model_forward_unbatched(make_state):
    model = CombatModel()
    encoded = model.encode_state(_sample_state(make_state), device="cpu")

    out = model.forward(encoded)

    assert out["logits"].shape == (1, model.total_actions)
    assert out["value"].shape == (1,)
    assert out["probs"].shape == (1, model.total_actions)

    probs_sum = out["probs"].sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_combat_model_forward_batched(make_state):
    model = CombatModel()

    encoded_1 = model.encode_state(_sample_state(make_state), device="cpu")
    encoded_2 = model.encode_state(_sample_state(make_state), device="cpu")
    batch = stack_encoded_obs([encoded_1, encoded_2])

    out = model.forward(batch)

    assert out["logits"].shape == (2, model.total_actions)
    assert out["value"].shape == (2,)
    assert out["probs"].shape == (2, model.total_actions)

    probs_sum = out["probs"].sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_combat_model_masks_invalid_actions(make_state):
    model = CombatModel()

    state = make_state(
        hand=[make_state.card("Bash")],   # ciblée
        monsters=[make_state.enemy(hp=40)],
        energy=2,
    )
    encoded = model.encode_state(state, device="cpu")

    out = model.forward(encoded)
    logits = out["logits"].squeeze(0)
    mask = encoded["valid_action_mask"]

    invalid_indices = (mask <= 0.0).nonzero(as_tuple=False).flatten()
    assert invalid_indices.numel() > 0

    for idx in invalid_indices[:10]:
        assert logits[idx].item() < -1e8


def test_combat_model_fallback_when_all_actions_invalid():
    model = CombatModel()

    total_actions = model.total_actions
    max_hand = model.cfg.combat_obs.max_hand_cards
    max_enemies = model.cfg.combat_obs.max_enemies
    fallback_idx = max_hand + max_hand * max_enemies

    encoded = {
        "player_scalars": torch.zeros(model.player_dim),
        "hand_cards": torch.zeros(model.cfg.combat_obs.max_hand_cards, model.encoder.card_feature_dim),
        "hand_mask": torch.zeros(model.cfg.combat_obs.max_hand_cards),
        "enemies": torch.zeros(model.cfg.combat_obs.max_enemies, model.enemy_dim),
        "enemy_mask": torch.zeros(model.cfg.combat_obs.max_enemies),
        "potions": torch.zeros(model.cfg.combat_obs.max_potions, model.potion_dim),
        "potion_mask": torch.zeros(model.cfg.combat_obs.max_potions),
        "combat_context": torch.zeros(model.context_dim),
        "deck_counts": torch.zeros(model.deck_dim),
        "discard_counts": torch.zeros(model.deck_dim),
        "exhaust_counts": torch.zeros(model.deck_dim),
        "relics": torch.zeros(model.relic_dim),
        "valid_action_mask": torch.zeros(total_actions),
    }

    out = model.forward(encoded)
    logits = out["logits"].squeeze(0)

    assert logits[fallback_idx].item() > -1e8


def test_combat_model_act_returns_valid_action(make_state):
    model = CombatModel()
    encoded = model.encode_state(_sample_state(make_state), device="cpu")

    out = model.act(encoded, deterministic=False)

    action = int(out["action"].item())
    valid_mask = encoded["valid_action_mask"]

    assert 0 <= action < model.total_actions
    assert valid_mask[action].item() == 1.0
    assert out["log_prob"].shape == (1,)
    assert out["value"].shape == (1,)


def test_combat_model_act_deterministic_returns_valid_action(make_state):
    model = CombatModel()
    encoded = model.encode_state(_sample_state(make_state), device="cpu")

    out = model.act(encoded, deterministic=True)

    action = int(out["action"].item())
    valid_mask = encoded["valid_action_mask"]

    assert 0 <= action < model.total_actions
    assert valid_mask[action].item() == 1.0


def test_combat_model_evaluate_actions_shapes(make_state):
    model = CombatModel()

    encoded_1 = model.encode_state(_sample_state(make_state), device="cpu")
    encoded_2 = model.encode_state(_sample_state(make_state), device="cpu")
    batch = stack_encoded_obs([encoded_1, encoded_2])

    act_out = model.act(batch, deterministic=False)
    actions = act_out["action"]

    eval_out = model.evaluate_actions(batch, actions)

    assert eval_out["log_prob"].shape == (2,)
    assert eval_out["entropy"].shape == (2,)
    assert eval_out["value"].shape == (2,)
    assert eval_out["logits"].shape == (2, model.total_actions)


def test_stack_encoded_obs_stacks_everything(make_state):
    model = CombatModel()

    e1 = model.encode_state(_sample_state(make_state), device="cpu")
    e2 = model.encode_state(_sample_state(make_state), device="cpu")
    batch = stack_encoded_obs([e1, e2])

    assert batch["player_scalars"].shape[0] == 2
    assert batch["hand_cards"].shape[0] == 2
    assert batch["enemies"].shape[0] == 2
    assert batch["potions"].shape[0] == 2
    assert batch["combat_context"].shape[0] == 2
    assert batch["valid_action_mask"].shape[0] == 2


def test_compute_ppo_loss_runs_end_to_end(make_state):
    model = CombatModel()

    e1 = model.encode_state(_sample_state(make_state), device="cpu")
    e2 = model.encode_state(_sample_state(make_state), device="cpu")
    batch = stack_encoded_obs([e1, e2])

    act_out = model.act(batch, deterministic=False)
    actions = act_out["action"]
    old_log_probs = act_out["log_prob"].detach()

    returns = torch.tensor([1.0, 0.5], dtype=torch.float32)
    advantages = torch.tensor([0.3, -0.2], dtype=torch.float32)

    loss_out = compute_ppo_loss(
        model=model,
        batch_obs=batch,
        batch_actions=actions,
        batch_old_log_probs=old_log_probs,
        batch_returns=returns,
        batch_advantages=advantages,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    assert "loss" in loss_out
    assert loss_out["loss"].ndim == 0
    assert torch.isfinite(loss_out["loss"])
    assert torch.isfinite(loss_out["policy_loss"])
    assert torch.isfinite(loss_out["value_loss"])
    assert torch.isfinite(loss_out["entropy"])


def test_combat_model_act_from_state_runs(make_state):
    model = CombatModel()

    out = model.act_from_state(_sample_state(make_state), deterministic=False)

    action = int(out["action"].item())
    assert 0 <= action < model.total_actions
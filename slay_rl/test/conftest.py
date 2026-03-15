from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Ajoute la racine du projet au sys.path pour permettre les imports du package slay_rl
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports normaux du projet
from slay_rl import config as config_mod  # noqa: F401
from slay_rl import sts_env as env_mod
from slay_rl.agents import combat_agent as agent_mod
from slay_rl.features import combat_encoder as encoder_mod  # noqa: F401
from slay_rl.models import combat_model as model_mod  # noqa: F401
from slay_rl.rewards import combat_reward as reward_mod  # noqa: F401
from slay_rl.train import train_combat as train_mod  # noqa: F401


@pytest.fixture
def env_api() -> Dict[str, Any]:
    return {
        "MockGameBackend": env_mod.MockGameBackend,
        "STSEnv": env_mod.STSEnv,
        "CombatCommand": agent_mod.CombatCommand,
        "make_card": env_mod.make_card,
        "CARD_DEFS": env_mod.CARD_DEFS,
        "POTION_DEFS": env_mod.POTION_DEFS,
        "config_mod": config_mod,
        "encoder_mod": encoder_mod,
        "model_mod": model_mod,
        "reward_mod": reward_mod,
        "train_mod": train_mod,
    }


@pytest.fixture
def backend(env_api):
    return env_api["MockGameBackend"](seed=123)


@pytest.fixture
def make_state(env_api):
    make_card = env_api["make_card"]

    def _enemy(
        name: str = "Jaw Worm",
        hp: int = 40,
        block: int = 0,
        intent: str = "ATTACK",
        intent_base_damage: int = 0,
        powers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "current_hp": hp,
            "max_hp": hp,
            "block": block,
            "powers": copy.deepcopy(powers or []),
            "isDead": False,
            "move_history": [],
            "last_move": None,
            "combat_state": {},
            "intent": intent,
            "intent_base_damage": intent_base_damage,
        }

    def _state(
        *,
        hand: List[Dict[str, Any]],
        draw_pile: Optional[List[Dict[str, Any]]] = None,
        discard_pile: Optional[List[Dict[str, Any]]] = None,
        exhaust_pile: Optional[List[Dict[str, Any]]] = None,
        monsters: Optional[List[Dict[str, Any]]] = None,
        potions: Optional[List[Dict[str, Any]]] = None,
        energy: int = 3,
        hp: int = 80,
        max_hp: int = 80,
        block: int = 0,
        player_powers: Optional[List[Dict[str, Any]]] = None,
        relics: Optional[List[Dict[str, Any]]] = None,
        combat_meta: Optional[Dict[str, Any]] = None,
        turn: int = 1,
    ) -> Dict[str, Any]:
        default_meta = {
            "hp_loss_count": 0,
            "cards_played_this_turn": 0,
            "attacks_played_this_turn": 0,
            "cannot_draw_more_this_turn": False,
            "double_tap_charges": 0,
            "strength_down_at_end_turn": 0,
            "dexterity_down_at_end_turn": 0,
            "pending_max_hp_gain": 0,
            "last_card_played_name": None,
            "last_damage_dealt": 0,
            "last_unblocked_damage_dealt": 0,
            "last_x_energy_spent": 0,
            "last_potion_used": None,
            "illegal_action_count": 0,
        }
        if combat_meta:
            default_meta.update(copy.deepcopy(combat_meta))

        return {
            "in_combat": True,
            "combat_over": False,
            "game_over": False,
            "victory": False,
            "turn": turn,
            "energy": energy,
            "player": {
                "current_hp": hp,
                "max_hp": max_hp,
                "block": block,
                "powers": copy.deepcopy(player_powers or []),
                "relics": copy.deepcopy(relics or [{"name": "Burning Blood"}]),
                "gold": 99,
            },
            "hand": copy.deepcopy(hand),
            "draw_pile": copy.deepcopy(draw_pile or []),
            "discard_pile": copy.deepcopy(discard_pile or []),
            "exhaust_pile": copy.deepcopy(exhaust_pile or []),
            "potions": copy.deepcopy(
                potions
                or [
                    {
                        "name": "Empty Slot",
                        "usable": False,
                        "empty": True,
                        "requires_target": False,
                    }
                    for _ in range(5)
                ]
            ),
            "combat_meta": default_meta,
            "hp_loss_breakdown": {
                "enemy": 0,
                "burn": 0,
                "pain": 0,
                "decay": 0,
                "other": 0,
            },
            "monsters": copy.deepcopy(monsters or [_enemy()]),
        }

    _state.enemy = _enemy
    _state.card = make_card
    return _state


@pytest.fixture
def step_helpers(env_api, backend):
    CombatCommand = env_api["CombatCommand"]

    class Helpers:
        @staticmethod
        def set_state(state: Dict[str, Any]) -> Dict[str, Any]:
            backend.state = copy.deepcopy(state)
            return backend.state

        @staticmethod
        def get_state() -> Dict[str, Any]:
            return backend.state

        @staticmethod
        def play_card(hand_index: int, target_index: Optional[int] = None):
            return backend.step(
                CombatCommand(
                    "play_card",
                    hand_index=hand_index,
                    target_index=target_index,
                )
            )

        @staticmethod
        def end_turn():
            return backend.step(CombatCommand("end_turn"))

        @staticmethod
        def use_potion(potion_index: int, target_index: Optional[int] = None):
            return backend.step(
                CombatCommand(
                    "use_potion",
                    potion_index=potion_index,
                    target_index=target_index,
                )
            )

        @staticmethod
        def command(
            action_type: str,
            hand_index: Optional[int] = None,
            target_index: Optional[int] = None,
            potion_index: Optional[int] = None,
        ):
            return backend.step(
                CombatCommand(
                    action_type,
                    hand_index=hand_index,
                    target_index=target_index,
                    potion_index=potion_index,
                )
            )

    return Helpers()
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slay_rl.config import get_default_config
from slay_rl.models.combat_model import CombatModel
from slay_rl.agents.combat_agent import CombatAgent, RuleBasedCombatAgent, CombatCommand


# =========================================================
# SETTINGS
# =========================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "slay_rl"
    / "checkpoints"
    / "experiment_2_vec_bigrollout"
    / "combat_model_update_1100.pt"
)

DETERMINISTIC = True
FALLBACK_TO_RULE_AGENT = True

# Hors combat = manuel
COMBAT_ONLY_MODE = True

# Potions réactivées
DISABLE_POTIONS = False

# Si ta version de CommunicationMod attend un slot 1-based au lieu de 0-based,
# change juste cette valeur à 1.
POTION_SLOT_OFFSET = 0

# Temps d'attente hors combat pour ne pas spammer STATE
OUTSIDE_COMBAT_WAIT_FRAMES = 30
NOT_READY_WAIT_FRAMES = 5


# =========================================================
# LOG
# =========================================================

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def _norm_name(x: Any) -> str:
    return str(x or "").strip().lower()


def _normalize_power_list(raw_powers: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_powers, list):
        return []
    out: List[Dict[str, Any]] = []
    for p in raw_powers:
        if not isinstance(p, dict):
            continue
        out.append({
            "id": p.get("id", p.get("name", "UnknownPower")),
            "name": p.get("name", p.get("id", "UnknownPower")),
            "amount": p.get("amount", p.get("stack_amount", p.get("stackAmount", 0))),
        })
    return out


def _normalize_relics(raw_relics: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_relics, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in raw_relics:
        if not isinstance(r, dict):
            continue
        out.append({
            "id": r.get("id", r.get("name", "UnknownRelic")),
            "name": r.get("name", r.get("id", "UnknownRelic")),
            "counter": r.get("counter", -1),
        })
    return out


def _normalize_cards(raw_cards: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_cards, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in raw_cards:
        if not isinstance(c, dict):
            continue
        out.append({
            "id": c.get("id", c.get("card_id", c.get("name", "UnknownCard"))),
            "name": c.get("name", c.get("id", "UnknownCard")),
            "cost": c.get("cost", c.get("costForTurn", c.get("cost_for_turn", 99))),
            "cost_for_turn": c.get("cost_for_turn", c.get("costForTurn", c.get("cost", 99))),
            "type": c.get("type", "UNKNOWN"),
            "upgraded": bool(c.get("upgraded", c.get("upgrades", 0) > 0)),
            "exhaust": bool(c.get("exhaust", c.get("exhausts", False))),
            "ethereal": bool(c.get("ethereal", c.get("isEthereal", False))),
            "is_playable": bool(c.get("is_playable", c.get("isPlayable", c.get("playable", False)))),
            "has_target": bool(c.get("has_target", False)),
            "uuid": c.get("uuid"),
        })
    return out


def _normalize_monsters(raw_monsters: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_monsters, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in raw_monsters:
        if not isinstance(m, dict):
            continue
        out.append({
            "id": m.get("id", m.get("name", "UnknownEnemy")),
            "name": m.get("name", m.get("id", "UnknownEnemy")),
            "current_hp": m.get("current_hp", m.get("currentHealth", m.get("hp", 0))),
            "max_hp": m.get("max_hp", m.get("maxHealth", 1)),
            "block": m.get("block", m.get("currentBlock", 0)),
            "intent": m.get("intent", "UNKNOWN"),
            "intent_base_damage": m.get("intent_base_damage", m.get("move_base_damage", m.get("intentBaseDmg", 0))),
            "powers": _normalize_power_list(m.get("powers", [])),
            "isDead": bool(m.get("isDead", m.get("dead", False))),
            "escaped": bool(m.get("escaped", False)),
            "is_gone": bool(m.get("is_gone", False)),
            "half_dead": bool(m.get("half_dead", False)),
        })
    return out


def _normalize_potions(raw_potions: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_potions, list):
        return []

    out: List[Dict[str, Any]] = []
    for p in raw_potions:
        if not isinstance(p, dict):
            continue

        usable = bool(p.get("can_use", p.get("usable", False)))
        requires_target = bool(p.get("requires_target", False))
        empty = _norm_name(p.get("id", "")) == "potion slot" or _norm_name(p.get("name", "")) == "potion slot"

        if DISABLE_POTIONS:
            usable = False

        out.append({
            "id": p.get("id", p.get("name", "Potion")),
            "name": p.get("name", p.get("id", "Potion")),
            "usable": usable,
            "requires_target": requires_target,
            "empty": empty,
        })
    return out


def communicationmod_to_internal_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    game_state = payload.get("game_state", {}) or {}
    combat_state = game_state.get("combat_state", {}) or {}

    player_raw = combat_state.get("player", {}) or {}
    player = {
        "current_hp": player_raw.get("current_hp", game_state.get("current_hp", 0)),
        "max_hp": player_raw.get("max_hp", game_state.get("max_hp", 1)),
        "block": player_raw.get("block", 0),
        "energy": player_raw.get("energy", 0),
        "powers": _normalize_power_list(player_raw.get("powers", [])),
        "relics": _normalize_relics(game_state.get("relics", [])),
    }

    return {
        "turn": combat_state.get("turn", 1),
        "energy": player.get("energy", 0),
        "player": player,
        "hand": _normalize_cards(combat_state.get("hand", [])),
        "draw_pile": _normalize_cards(combat_state.get("draw_pile", [])),
        "discard_pile": _normalize_cards(combat_state.get("discard_pile", [])),
        "exhaust_pile": _normalize_cards(combat_state.get("exhaust_pile", [])),
        "monsters": _normalize_monsters(combat_state.get("monsters", [])),
        "potions": _normalize_potions(game_state.get("potions", [])),
        "combat_over": str(game_state.get("room_phase", "")).upper() != "COMBAT",
        "screen_type": game_state.get("screen_type", "UNKNOWN"),
        "room_phase": game_state.get("room_phase", "UNKNOWN"),
        "floor": game_state.get("floor"),
        "act": game_state.get("act"),
        "gold": game_state.get("gold"),
    }


# =========================================================
# GAME STATE HELPERS
# =========================================================

def get_available(payload: Dict[str, Any]) -> List[str]:
    return [str(x).lower() for x in payload.get("available_commands", [])]


def is_in_combat(payload: Dict[str, Any]) -> bool:
    game_state = payload.get("game_state", {}) or {}
    room_phase = str(game_state.get("room_phase", "")).upper()
    return room_phase == "COMBAT"


def is_real_combat_turn(payload: Dict[str, Any]) -> bool:
    game_state = payload.get("game_state", {}) or {}
    available = get_available(payload)
    has_combat_state = isinstance(game_state.get("combat_state", None), dict)
    return is_in_combat(payload) and has_combat_state and ("play" in available or "end" in available or "potion" in available)


def is_combat_choice_screen(payload: Dict[str, Any]) -> bool:
    available = get_available(payload)
    return is_in_combat(payload) and "choose" in available


# =========================================================
# TACTICAL PLANNER
# =========================================================

def living_enemy_indices(state: Dict[str, Any]) -> List[int]:
    out = []
    for i, m in enumerate(state.get("monsters", [])):
        dead = bool(m.get("isDead", False))
        escaped = bool(m.get("escaped", False))
        gone = bool(m.get("is_gone", False))
        half_dead = bool(m.get("half_dead", False))
        hp = int(m.get("current_hp", 0) or 0)
        if not dead and not escaped and not gone and not half_dead and hp > 0:
            out.append(i)
    return out


def choose_lowest_hp_target(state: Dict[str, Any], only_attacking: bool = False) -> Optional[int]:
    best_idx = None
    best_hp = None
    for i in living_enemy_indices(state):
        m = state["monsters"][i]
        intent = str(m.get("intent", "UNKNOWN")).upper()
        if only_attacking and "ATTACK" not in intent:
            continue
        effective_hp = int(m.get("current_hp", 0) or 0) + int(m.get("block", 0) or 0)
        if best_hp is None or effective_hp < best_hp:
            best_hp = effective_hp
            best_idx = i
    return best_idx


def card_cost(card: Dict[str, Any]) -> int:
    c = card.get("cost_for_turn", card.get("cost", 99))
    try:
        c = int(c)
    except Exception:
        c = 99
    if c < 0:
        return 99
    return c


def is_attack(card: Dict[str, Any]) -> bool:
    return _norm_name(card.get("type")) == "attack"


def is_playable_card(card: Dict[str, Any]) -> bool:
    return bool(card.get("is_playable", False))


def card_key(card: Dict[str, Any]) -> str:
    return _norm_name(card.get("id") or card.get("name"))


def has_affordable_followup_attack(state: Dict[str, Any], remaining_energy: int, exclude_idx: Optional[int] = None) -> bool:
    for i, card in enumerate(state.get("hand", [])):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if not is_playable_card(card):
            continue
        if not is_attack(card):
            continue
        if card_cost(card) <= remaining_energy:
            return True
    return False


def score_attack_card(card: Dict[str, Any]) -> float:
    name = card_key(card)

    base_scores = {
        "bludgeon": 20.0,
        "immolate": 18.0,
        "feed": 17.0,
        "uppercut": 14.0,
        "bash": 13.0,
        "heavy blade": 13.0,
        "heavy_blade": 13.0,
        "carnage": 13.0,
        "hemokinesis": 12.0,
        "clothesline": 11.0,
        "pommel strike": 10.0,
        "pommel_strike": 10.0,
        "twin strike": 10.0,
        "twin_strike": 10.0,
        "sword boomerang": 10.0,
        "sword_boomerang": 10.0,
        "pummel": 10.0,
        "headbutt": 9.0,
        "iron wave": 8.5,
        "iron_wave": 8.5,
        "strike_r": 6.0,
        "anger": 7.0,
        "thunderclap": 8.0,
    }

    s = base_scores.get(name, 5.0)
    s += 0.35 * max(0, 3 - card_cost(card))
    if bool(card.get("upgraded", False)):
        s += 0.5
    return s


def choose_best_affordable_attack(state: Dict[str, Any], energy_budget: int, exclude_idx: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    best_idx = None
    best_target = None
    best_score = None

    for i, card in enumerate(state.get("hand", [])):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if not is_playable_card(card):
            continue
        if not is_attack(card):
            continue
        if card_cost(card) > energy_budget:
            continue

        score = score_attack_card(card)
        if bool(card.get("has_target", False)):
            target = choose_lowest_hp_target(state)
            if target is None:
                continue
        else:
            target = None

        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
            best_target = target

    return best_idx, best_target


def planner_choose_command(state: Dict[str, Any]) -> Optional[CombatCommand]:
    """
    Petite couche tactique au-dessus du modèle:
    - Rage avant attaque
    - Inflame / Flex avant attaque
    - Spot Weakness avant attaque si un ennemi attaque
    - Double Tap avant une grosse attaque
    """
    hand = state.get("hand", [])
    energy = int(state.get("energy", 0) or 0)

    # index des cartes clés
    rage_idx = None
    inflame_idx = None
    flex_idx = None
    double_tap_idx = None
    spot_weakness_idx = None

    for i, card in enumerate(hand):
        if not is_playable_card(card):
            continue
        key = card_key(card)
        if key == "rage":
            rage_idx = i
        elif key == "inflame":
            inflame_idx = i
        elif key == "flex":
            flex_idx = i
        elif key in {"double tap", "double_tap"}:
            double_tap_idx = i
        elif key in {"spot weakness", "spot_weakness"}:
            spot_weakness_idx = i

    # 1) Double Tap avant la meilleure grosse attaque
    if double_tap_idx is not None:
        dt_cost = card_cost(hand[double_tap_idx])
        if dt_cost <= energy:
            atk_idx, _ = choose_best_affordable_attack(state, energy_budget=max(0, energy - dt_cost), exclude_idx=double_tap_idx)
            if atk_idx is not None and card_cost(hand[atk_idx]) >= 1:
                return CombatCommand(command_type="play_card", hand_index=double_tap_idx, target_index=None)

    # 2) Rage avant une attaque rentable
    if rage_idx is not None:
        c = card_cost(hand[rage_idx])
        if c <= energy and has_affordable_followup_attack(state, remaining_energy=energy - c, exclude_idx=rage_idx):
            return CombatCommand(command_type="play_card", hand_index=rage_idx, target_index=None)

    # 3) Spot Weakness avant attaque, si un ennemi attaque
    if spot_weakness_idx is not None:
        c = card_cost(hand[spot_weakness_idx])
        target = choose_lowest_hp_target(state, only_attacking=True)
        if c <= energy and target is not None and has_affordable_followup_attack(state, remaining_energy=energy - c, exclude_idx=spot_weakness_idx):
            return CombatCommand(command_type="play_card", hand_index=spot_weakness_idx, target_index=target)

    # 4) Inflame avant attaque
    if inflame_idx is not None:
        c = card_cost(hand[inflame_idx])
        if c <= energy and has_affordable_followup_attack(state, remaining_energy=energy - c, exclude_idx=inflame_idx):
            return CombatCommand(command_type="play_card", hand_index=inflame_idx, target_index=None)

    # 5) Flex avant attaque
    if flex_idx is not None:
        c = card_cost(hand[flex_idx])
        if c <= energy and has_affordable_followup_attack(state, remaining_energy=energy - c, exclude_idx=flex_idx):
            return CombatCommand(command_type="play_card", hand_index=flex_idx, target_index=None)

    return None


# =========================================================
# COMMAND CONVERSION
# =========================================================

def internal_command_to_text(command: Any) -> str:
    if command is None:
        return "END"

    command_type = getattr(command, "command_type", None)

    if command_type == "play_card":
        hand_index = getattr(command, "hand_index", None)
        target_index = getattr(command, "target_index", None)

        if hand_index is None:
            return "END"

        card_index_1_based = int(hand_index) + 1

        if target_index is None:
            return f"PLAY {card_index_1_based}"

        return f"PLAY {card_index_1_based} {int(target_index)}"

    if command_type == "end_turn":
        return "END"

    if command_type == "use_potion":
        potion_index = getattr(command, "potion_index", None)
        target_index = getattr(command, "target_index", None)

        if potion_index is None:
            return "END"

        slot = int(potion_index) + POTION_SLOT_OFFSET

        if target_index is None:
            return f"POTION Use {slot}"

        return f"POTION Use {slot} {int(target_index)}"

    return "END"


# =========================================================
# MODEL BRIDGE
# =========================================================

class LiveModelBridge:
    def __init__(self) -> None:
        self.cfg = get_default_config()
        self.device = "cuda" if torch.cuda.is_available() and self.cfg.train.device == "cuda" else "cpu"

        self.model = CombatModel(cfg=self.cfg).to(self.device)
        self.agent = CombatAgent(self.model, cfg=self.cfg, device=self.device)
        self.rule_agent = RuleBasedCombatAgent(cfg=self.cfg)

        self.loaded_model = False
        try:
            payload = torch.load(CHECKPOINT_PATH, map_location=self.device)
            state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.loaded_model = True
            log(f"[bridge] loaded checkpoint: {CHECKPOINT_PATH}")
        except Exception as exc:
            log(f"[bridge] failed to load checkpoint: {CHECKPOINT_PATH}")
            log(f"[bridge] load error: {exc}")
            if not FALLBACK_TO_RULE_AGENT:
                raise

    def choose_combat_text_command(self, payload: Dict[str, Any]) -> str:
        available = get_available(payload)

        # Si on ne peut plus rien jouer mais qu'on peut finir, on finit.
        if "end" in available and "play" not in available and "potion" not in available:
            return "END"

        state = communicationmod_to_internal_state(payload)

        # 1) planner tactique
        planned = planner_choose_command(state)
        if planned is not None:
            text_cmd = internal_command_to_text(planned)
            log(f"[planner] {planned.to_dict()} -> {text_cmd}")
            return text_cmd

        # 2) modèle
        try:
            if self.loaded_model:
                command = self.agent.choose_command(state, deterministic=DETERMINISTIC)
                text_cmd = internal_command_to_text(command)
                log(f"[model] {command.to_dict()} -> {text_cmd}")
                return text_cmd

            command = self.rule_agent.choose_command(state)
            text_cmd = internal_command_to_text(command)
            log(f"[rule] {command.to_dict()} -> {text_cmd}")
            return text_cmd

        except Exception as exc:
            log(f"[bridge] combat decision error: {exc}")
            return "END"


# =========================================================
# MAIN LOOP
# =========================================================

def main() -> None:
    bridge = LiveModelBridge()

    print("Ready")
    sys.stdout.flush()
    log("[bridge] Ready sent")

    while True:
        line = sys.stdin.readline()
        if not line:
            log("[bridge] stdin closed, exiting")
            break

        raw = line.strip()
        if not raw:
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print("STATE")
            sys.stdout.flush()
            continue

        if "error" in payload:
            log(f"[bridge] CommunicationMod error: {payload.get('error')}")
            print(f"WAIT {NOT_READY_WAIT_FRAMES}")
            sys.stdout.flush()
            continue

        in_game = bool(payload.get("in_game", False))
        ready_for_command = bool(payload.get("ready_for_command", False))

        if not in_game:
            print("STATE")
            sys.stdout.flush()
            continue

        if not ready_for_command:
            print(f"WAIT {NOT_READY_WAIT_FRAMES}")
            sys.stdout.flush()
            continue

        # Combat normal = IA
        if is_real_combat_turn(payload):
            cmd = bridge.choose_combat_text_command(payload)
            print(cmd)
            sys.stdout.flush()
            continue

        # Ecran spécial pendant le combat = éviter le blocage
        if is_combat_choice_screen(payload):
            print("CHOOSE 0")
            sys.stdout.flush()
            continue

        # Hors combat = tu joues à la main
        if COMBAT_ONLY_MODE:
            print(f"WAIT {OUTSIDE_COMBAT_WAIT_FRAMES}")
            sys.stdout.flush()
            continue

        print("STATE")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
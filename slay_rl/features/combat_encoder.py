from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from slay_rl.config import (
    CARD_TO_IDX,
    ENEMY_TO_IDX,
    RELIC_TO_IDX,
    Config,
    get_default_config,
)


# ============================================================
# Small helpers
# ============================================================

INTENT_TO_IDX = {
    "ATTACK": 0,
    "ATTACK_BUFF": 1,
    "ATTACK_DEBUFF": 2,
    "ATTACK_DEFEND": 3,
    "BUFF": 4,
    "DEBUFF": 5,
    "STRONG_DEBUFF": 6,
    "DEFEND": 7,
    "DEFEND_DEBUFF": 8,
    "DEFEND_BUFF": 9,
    "DEFEND_ALLY": 10,
    "ESCAPE": 11,
    "MAGIC": 12,
    "CHARGE": 13,
    "NONE": 14,
    "SLEEP": 15,
    "STUN": 16,
    "UNKNOWN": 17,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp01_from_max(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    x = value / max_value
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def one_hot(index: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


# ============================================================
# Parsed structures
# ============================================================

@dataclass
class ParsedCard:
    card_id: str
    cost: int
    upgraded: int
    exhausts: int
    ethereal: int
    is_playable: int
    has_target: int
    raw: Dict[str, Any]


@dataclass
class ParsedEnemy:
    name: str
    hp: float
    max_hp: float
    block: float
    intent: str
    intent_base_damage: float
    strength: float
    weak: float
    vulnerable: float
    alive: int
    raw: Dict[str, Any]


@dataclass
class ParsedPlayer:
    hp: float
    max_hp: float
    block: float
    energy: float
    strength: float
    dexterity: float
    weak: float
    vulnerable: float
    frail: float
    raw: Dict[str, Any]


# ============================================================
# Main encoder
# ============================================================

class CombatEncoder:
    """
    Encode a combat state into tensors usable by the combat model.

    Output dictionary:
    - player_scalars:        [player_scalar_dim]
    - hand_cards:            [max_hand_cards, card_feature_dim]
    - hand_mask:             [max_hand_cards]
    - enemies:               [max_enemies, enemy_feature_dim]
    - enemy_mask:            [max_enemies]
    - deck_counts:           [card_vocab_size]
    - discard_counts:        [card_vocab_size]
    - exhaust_counts:        [card_vocab_size]
    - relics:                [relic_vocab_size]
    - valid_action_mask:     [combat_action.total_actions]
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_default_config()

        self.max_hand_cards = self.cfg.combat_obs.max_hand_cards
        self.max_enemies = self.cfg.combat_obs.max_enemies

        self.card_vocab_size = self.cfg.combat_obs.card_vocab_size
        self.relic_vocab_size = self.cfg.combat_obs.relic_vocab_size
        self.enemy_vocab_size = self.cfg.combat_obs.enemy_vocab_size

        self.total_actions = self.cfg.combat_action.total_actions

        # Features per card in hand:
        # [card_one_hot..., cost_norm, upgraded, exhausts, ethereal, is_playable, has_target]
        self.card_feature_dim = self.card_vocab_size + 6

        # Features per enemy:
        # [enemy_one_hot..., hp_ratio, hp_abs_norm, block_norm, intent_one_hot..., intent_dmg_norm,
        #  strength_norm, weak_norm, vulnerable_norm, alive]
        self.enemy_feature_dim = (
            self.enemy_vocab_size
            + 4
            + len(INTENT_TO_IDX)
            + 4
        )

    # ========================================================
    # Public API
    # ========================================================

    def encode(self, state: Dict[str, Any], device: str = "cpu") -> Dict[str, torch.Tensor]:
        player = self._parse_player(state)
        hand = self._parse_hand(state)
        enemies = self._parse_enemies(state)
        draw_pile = self._extract_card_list(state, ["draw_pile", "drawPile"])
        discard_pile = self._extract_card_list(state, ["discard_pile", "discardPile"])
        exhaust_pile = self._extract_card_list(state, ["exhaust_pile", "exhaustPile"])
        relics = self._extract_relic_list(state)

        player_scalars = self._encode_player(player)
        hand_cards, hand_mask = self._encode_hand(hand)
        enemy_tensor, enemy_mask = self._encode_enemies(enemies)

        deck_counts = self._count_cards(draw_pile)
        discard_counts = self._count_cards(discard_pile)
        exhaust_counts = self._count_cards(exhaust_pile)
        relic_vec = self._encode_relics(relics)

        valid_action_mask = self.build_valid_action_mask(state, hand, enemies)

        return {
            "player_scalars": torch.tensor(player_scalars, dtype=torch.float32, device=device),
            "hand_cards": torch.tensor(hand_cards, dtype=torch.float32, device=device),
            "hand_mask": torch.tensor(hand_mask, dtype=torch.float32, device=device),
            "enemies": torch.tensor(enemy_tensor, dtype=torch.float32, device=device),
            "enemy_mask": torch.tensor(enemy_mask, dtype=torch.float32, device=device),
            "deck_counts": torch.tensor(deck_counts, dtype=torch.float32, device=device),
            "discard_counts": torch.tensor(discard_counts, dtype=torch.float32, device=device),
            "exhaust_counts": torch.tensor(exhaust_counts, dtype=torch.float32, device=device),
            "relics": torch.tensor(relic_vec, dtype=torch.float32, device=device),
            "valid_action_mask": torch.tensor(valid_action_mask, dtype=torch.float32, device=device),
        }

    def build_valid_action_mask(
            self,
            state: Dict[str, Any],
            hand: Optional[List[ParsedCard]] = None,
            enemies: Optional[List[ParsedEnemy]] = None,
    ) -> List[float]:
        """
        Safe action mask:
        - playable non-target cards
        - playable targeted cards with valid living targets
        - usable non-empty non-target potions
        - targeted potions only when a valid living target exists
        - end turn remains available during combat
        """
        hand = hand if hand is not None else self._parse_hand(state)
        enemies = enemies if enemies is not None else self._parse_enemies(state)

        mask = [0.0] * self.total_actions

        targeted_base = self.max_hand_cards
        targeted_size = self.max_hand_cards * self.max_enemies
        end_turn_idx = targeted_base + targeted_size
        potion_base = end_turn_idx + 1
        potion_slots = 5
        potion_target_base = potion_base + potion_slots

        for i in range(self.max_hand_cards):
            if i < len(hand):
                card = hand[i]
                if card.is_playable and card.has_target == 0:
                    mask[i] = 1.0

        for i in range(self.max_hand_cards):
            if i >= len(hand):
                continue
            card = hand[i]
            if not card.is_playable or card.has_target != 1:
                continue

            for target_idx in range(self.max_enemies):
                flat_idx = targeted_base + (i * self.max_enemies) + target_idx
                if target_idx < len(enemies) and enemies[target_idx].alive == 1:
                    mask[flat_idx] = 1.0

        if not self._is_combat_finished(state):
            mask[end_turn_idx] = 1.0

        potions = self._extract_potions(state)
        for slot_idx in range(potion_slots):
            if slot_idx >= len(potions):
                continue

            potion = potions[slot_idx]
            if potion.get("empty", False) or not potion.get("usable", True):
                continue

            requires_target = bool(potion.get("requires_target", False))

            if not requires_target:
                flat_idx = potion_base + slot_idx
                mask[flat_idx] = 1.0
                continue

            for target_idx in range(self.max_enemies):
                flat_idx = potion_target_base + slot_idx * self.max_enemies + target_idx
                if target_idx < len(enemies) and enemies[target_idx].alive == 1:
                    mask[flat_idx] = 1.0

        if sum(mask) == 0:
            mask[end_turn_idx] = 1.0

        return mask

    # ========================================================
    # Parsing
    # ========================================================

    def _parse_player(self, state: Dict[str, Any]) -> ParsedPlayer:
        player = state.get("player", state)

        powers = self._extract_powers(player)

        hp = safe_float(player.get("current_hp", player.get("currentHealth", player.get("hp", 0))))
        max_hp = safe_float(player.get("max_hp", player.get("maxHealth", 1)))
        block = safe_float(player.get("block", player.get("currentBlock", 0)))
        energy = safe_float(
            player.get("energy", state.get("energy", state.get("player_energy", 0)))
        )

        return ParsedPlayer(
            hp=hp,
            max_hp=max_hp,
            block=block,
            energy=energy,
            strength=self._power_amount(powers, ["Strength"]),
            dexterity=self._power_amount(powers, ["Dexterity"]),
            weak=self._power_amount(powers, ["Weak"]),
            vulnerable=self._power_amount(powers, ["Vulnerable"]),
            frail=self._power_amount(powers, ["Frail"]),
            raw=player,
        )

    def _parse_hand(self, state: Dict[str, Any]) -> List[ParsedCard]:
        hand_raw = self._extract_card_list(state, ["hand", "hand_cards", "handCards"])
        hand: List[ParsedCard] = []

        available_energy = safe_int(
            state.get("energy", state.get("player_energy", state.get("player", {}).get("energy", 0)))
        )

        for c in hand_raw[: self.max_hand_cards]:
            card_id = self._normalize_card_id(c.get("id", c.get("card_id", c.get("name", "UnknownCard"))))
            cost = safe_int(c.get("cost_for_turn", c.get("costForTurn", c.get("cost", 99))), default=99)

            # Sometimes status/curses can have weird cost fields
            if cost < -1:
                cost = -1

            upgraded = 1 if c.get("upgraded", False) else 0
            exhausts = 1 if c.get("exhaust", c.get("exhausts", False)) else 0
            ethereal = 1 if c.get("isEthereal", c.get("ethereal", False)) else 0

            # Best effort for "playable"
            explicitly_playable = c.get("is_playable", c.get("playable", None))
            if explicitly_playable is None:
                is_playable = int(cost == -1 or cost <= available_energy)
            else:
                is_playable = 1 if explicitly_playable else 0

            has_target = self._card_has_target(c)

            hand.append(
                ParsedCard(
                    card_id=card_id,
                    cost=cost,
                    upgraded=upgraded,
                    exhausts=exhausts,
                    ethereal=ethereal,
                    is_playable=is_playable,
                    has_target=has_target,
                    raw=c,
                )
            )

        return hand

    def _parse_enemies(self, state: Dict[str, Any]) -> List[ParsedEnemy]:
        monsters = state.get("monsters", state.get("enemies", []))
        parsed: List[ParsedEnemy] = []

        for m in monsters[: self.max_enemies]:
            powers = self._extract_powers(m)

            hp = safe_float(m.get("current_hp", m.get("currentHealth", m.get("hp", 0))))
            max_hp = safe_float(m.get("max_hp", m.get("maxHealth", 1)))
            block = safe_float(m.get("block", m.get("currentBlock", 0)))

            intent = str(m.get("intent", "UNKNOWN")).upper()
            if intent not in INTENT_TO_IDX:
                intent = "UNKNOWN"

            intent_base_damage = safe_float(
                m.get("intent_base_damage", m.get("intentBaseDmg", m.get("move_base_damage", 0)))
            )

            is_dead = bool(m.get("isDead", False)) or bool(m.get("dead", False))
            escaped = bool(m.get("escaped", False))
            alive = 0 if is_dead or escaped or hp <= 0 else 1

            name = str(m.get("name", "UnknownEnemy"))

            parsed.append(
                ParsedEnemy(
                    name=name,
                    hp=hp,
                    max_hp=max_hp,
                    block=block,
                    intent=intent,
                    intent_base_damage=intent_base_damage,
                    strength=self._power_amount(powers, ["Strength"]),
                    weak=self._power_amount(powers, ["Weak"]),
                    vulnerable=self._power_amount(powers, ["Vulnerable"]),
                    alive=alive,
                    raw=m,
                )
            )

        return parsed

    # ========================================================
    # Encoding
    # ========================================================

    def _encode_player(self, player: ParsedPlayer) -> List[float]:
        max_hp = max(player.max_hp, 1.0)

        return [
            clamp01_from_max(player.hp, max_hp),          # hp ratio
            min(player.hp / 200.0, 1.0),                 # hp absolute normalized
            min(player.block / 100.0, 1.0),              # block
            min(player.energy / 10.0, 1.0),              # energy
            min(max(player.strength, 0.0) / 10.0, 1.0),  # strength
            min(max(player.dexterity, 0.0) / 10.0, 1.0), # dexterity
            min(max(player.vulnerable, 0.0) / 5.0, 1.0), # vulnerable
            min(max(player.frail, 0.0) / 5.0, 1.0),      # frail
        ]

    def _encode_hand(self, hand: List[ParsedCard]) -> Tuple[List[List[float]], List[float]]:
        rows: List[List[float]] = []
        mask: List[float] = []

        for card in hand[: self.max_hand_cards]:
            card_idx = CARD_TO_IDX.get(card.card_id, -1)
            card_one_hot = one_hot(card_idx, self.card_vocab_size)

            if card.cost == -1:
                cost_norm = 0.0
            elif card.cost >= 99:
                cost_norm = 1.0
            else:
                cost_norm = min(card.cost / 5.0, 1.0)

            row = card_one_hot + [
                cost_norm,
                float(card.upgraded),
                float(card.exhausts),
                float(card.ethereal),
                float(card.is_playable),
                float(card.has_target),
            ]
            rows.append(row)
            mask.append(1.0)

        while len(rows) < self.max_hand_cards:
            rows.append([0.0] * self.card_feature_dim)
            mask.append(0.0)

        return rows, mask

    def _encode_enemies(self, enemies: List[ParsedEnemy]) -> Tuple[List[List[float]], List[float]]:
        rows: List[List[float]] = []
        mask: List[float] = []

        for enemy in enemies[: self.max_enemies]:
            enemy_idx = ENEMY_TO_IDX.get(enemy.name, -1)
            enemy_one_hot = one_hot(enemy_idx, self.enemy_vocab_size)

            intent_one_hot = one_hot(INTENT_TO_IDX.get(enemy.intent, INTENT_TO_IDX["UNKNOWN"]), len(INTENT_TO_IDX))

            max_hp = max(enemy.max_hp, 1.0)
            row = enemy_one_hot + [
                clamp01_from_max(enemy.hp, max_hp),
                min(enemy.hp / 250.0, 1.0),
                min(enemy.block / 100.0, 1.0),
                1.0 if enemy.alive else 0.0,
            ] + intent_one_hot + [
                min(enemy.intent_base_damage / 50.0, 1.0),
                min(max(enemy.strength, 0.0) / 10.0, 1.0),
                min(max(enemy.weak, 0.0) / 5.0, 1.0),
                min(max(enemy.vulnerable, 0.0) / 5.0, 1.0),
            ]

            rows.append(row)
            mask.append(1.0)

        while len(rows) < self.max_enemies:
            rows.append([0.0] * self.enemy_feature_dim)
            mask.append(0.0)

        return rows, mask

    def _count_cards(self, cards: List[Dict[str, Any]]) -> List[float]:
        counts = [0.0] * self.card_vocab_size
        for c in cards:
            card_id = self._normalize_card_id(c.get("id", c.get("card_id", c.get("name", "UnknownCard"))))
            idx = CARD_TO_IDX.get(card_id)
            if idx is not None:
                counts[idx] += 1.0

        # Soft normalize
        total = sum(counts)
        if total > 0:
            counts = [min(x / total, 1.0) for x in counts]

        return counts

    def _encode_relics(self, relics: List[Dict[str, Any]]) -> List[float]:
        vec = [0.0] * self.relic_vocab_size

        for relic in relics:
            relic_name = self._extract_relic_name(relic)
            idx = RELIC_TO_IDX.get(relic_name)
            if idx is not None:
                vec[idx] = 1.0

        return vec

    def _normalize_relic_name_text(self, name: str) -> str:
        return str(name or "").strip()

    def _canonicalize_relic_name(self, name: str) -> str:
        raw = self._normalize_relic_name_text(name)

        alias_map = {
            "Paper Frog": "Paper Phrog",
            "Paper Phrog": "Paper Phrog",

            "Cultist Mask": "Cultist Headpiece",
            "Cultist Headpiece": "Cultist Headpiece",

            "Gremlin Mask": "Gremlin Visage",
            "Gremlin Visage": "Gremlin Visage",

            "Captain wheel": "Captain's Wheel",
            "Captain's Wheel": "Captain's Wheel",

            "Wing Boots": "Winged Greaves",
            "Winged Greaves": "Winged Greaves",

            "Sling": "Sling of Courage",
            "Sling of Courage": "Sling of Courage",

            "Nloth's Gift": "N'loth's Gift",
            "N'loth's Gift": "N'loth's Gift",

            "Boot": "The Boot",
            "The Boot": "The Boot",
        }

        return alias_map.get(raw, raw)

    def _extract_relic_name(self, relic: Dict[str, Any]) -> str:
        raw_name = relic.get("name", relic.get("id", ""))
        return self._canonicalize_relic_name(str(raw_name))

    def _has_relic_name(self, relics: List[Dict[str, Any]], relic_name: str) -> bool:
        target = self._canonicalize_relic_name(relic_name)
        for relic in relics:
            if self._extract_relic_name(relic) == target:
                return True
        return False

    # ========================================================
    # Extraction helpers
    # ========================================================

    def _extract_card_list(self, state: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
        for key in keys:
            if key in state and isinstance(state[key], list):
                return state[key]

        # Sometimes nested under player
        player = state.get("player", {})
        for key in keys:
            if key in player and isinstance(player[key], list):
                return player[key]

        return []

    def _extract_relic_list(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ["relics", "player_relics"]:
            if key in state and isinstance(state[key], list):
                return state[key]

        player = state.get("player", {})
        for key in ["relics", "player_relics"]:
            if key in player and isinstance(player[key], list):
                return player[key]

        return []

    def _extract_potions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ["potions", "player_potions"]:
            if key in state and isinstance(state[key], list):
                return state[key]

        player = state.get("player", {})
        for key in ["potions", "player_potions"]:
            if key in player and isinstance(player[key], list):
                return player[key]

        return []

    def _extract_powers(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ["powers", "buffs", "debuffs"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return []

    def _power_amount(self, powers: List[Dict[str, Any]], names: List[str]) -> float:
        target_names = set(n.lower() for n in names)

        total = 0.0
        for p in powers:
            power_name = str(p.get("id", p.get("name", ""))).lower()
            if power_name in {x.lower() for x in names}:
                total += safe_float(p.get("amount", p.get("stack", 0)))
            else:
                # More permissive substring match
                for n in target_names:
                    if n in power_name:
                        total += safe_float(p.get("amount", p.get("stack", 0)))
                        break
        return total

    def _normalize_card_id(self, raw_id: str) -> str:
        card_id = str(raw_id).strip()

        # Some connectors might use spaces or lowercase names
        if card_id in CARD_TO_IDX:
            return card_id

        # Best effort normalization
        candidates = [
            card_id.replace(" ", "_"),
            card_id.replace("_", " "),
            card_id.title().replace(" ", "_"),
        ]
        for c in candidates:
            if c in CARD_TO_IDX:
                return c

        return card_id

    def _card_has_target(self, card: Dict[str, Any]) -> int:
        """
        Best effort target inference.
        Many attack cards target one enemy, but some attacks hit all enemies.
        Some skill/power cards do not target.
        """
        # Explicit metadata first
        if "has_target" in card:
            return 1 if card["has_target"] else 0
        if "target" in card:
            target = str(card["target"]).upper()
            if target in {"ENEMY", "SELF_AND_ENEMY"}:
                return 1
            if target in {"ALL_ENEMY", "SELF", "NONE", "ALL"}:
                return 0

        card_id = self._normalize_card_id(card.get("id", card.get("name", "")))

        known_target_cards = {
            "Bash",
            "Body Slam",
            "Clash",
            "Heavy Blade",
            "Pommel Strike",
            "Sword Boomerang",
            "Thunderclap",
            "Carnage",
            "Disarm",
            "Dropkick",
            "Hemokinesis",
            "Pummel",
            "Reckless Charge",
            "Uppercut",
            "Bludgeon",
            "Feed",
        }

        known_no_target_cards = {
            "Defend_R",
            "Armaments",
            "Cleave",
            "Flex",
            "Ghostly Armor",
            "Inflame",
            "Shrug It Off",
            "Warcry",
            "Whirlwind",
            "Battle Trance",
            "Bloodletting",
            "Combust",
            "Entrench",
            "Evolve",
            "Feel No Pain",
            "Fire Breathing",
            "Flame Barrier",
            "Infernal Blade",
            "Metallicize",
            "Power Through",
            "Rage",
            "Second Wind",
            "Seeing Red",
            "Shockwave",
            "Barricade",
            "Berserk",
            "Brutality",
            "Corruption",
            "Demon Form",
            "Double Tap",
            "Exhume",
            "Fiend Fire",
            "Immolate",
            "Impervious",
            "Juggernaut",
            "Limit Break",
            "Offering",
            "Reaper",
        }

        if card_id in known_target_cards:
            return 1
        if card_id in known_no_target_cards:
            return 0

        # Fallback heuristic from type/name
        card_type = str(card.get("type", "")).upper()
        if card_type == "ATTACK":
            return 1

        return 0

    def _is_combat_finished(self, state: Dict[str, Any]) -> bool:
        if state.get("combat_over", False):
            return True
        if state.get("is_terminal", False):
            return True

        enemies = self._parse_enemies(state)
        if len(enemies) > 0 and all(e.alive == 0 for e in enemies):
            return True

        player = self._parse_player(state)
        if player.hp <= 0:
            return True

        return False


# ============================================================
# Flatten helper for the model
# ============================================================

def flatten_combat_obs(encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Turn the structured encoded dict into a single 1D tensor.

    This is useful for a first MLP-based combat model.
    """
    parts = [
        encoded["player_scalars"].flatten(),
        encoded["hand_cards"].flatten(),
        encoded["hand_mask"].flatten(),
        encoded["enemies"].flatten(),
        encoded["enemy_mask"].flatten(),
        encoded["deck_counts"].flatten(),
        encoded["discard_counts"].flatten(),
        encoded["exhaust_counts"].flatten(),
        encoded["relics"].flatten(),
    ]
    return torch.cat(parts, dim=0)


# ============================================================
# Debug example
# ============================================================

if __name__ == "__main__":
    cfg = get_default_config()
    encoder = CombatEncoder(cfg)

    sample_state = {
        "energy": 3,
        "player": {
            "current_hp": 72,
            "max_hp": 80,
            "block": 5,
            "powers": [
                {"id": "Strength", "amount": 2},
                {"id": "Dexterity", "amount": 1},
            ],
            "relics": [
                {"name": "Burning Blood"},
                {"name": "Anchor"},
            ],
        },
        "hand": [
            {"id": "Strike_R", "cost": 1, "type": "ATTACK", "upgraded": False},
            {"id": "Bash", "cost": 2, "type": "ATTACK", "upgraded": False},
            {"id": "Defend_R", "cost": 1, "type": "SKILL", "upgraded": False},
            {"id": "Inflame", "cost": 1, "type": "POWER", "upgraded": False},
        ],
        "draw_pile": [
            {"id": "Strike_R"},
            {"id": "Defend_R"},
            {"id": "Bash"},
        ],
        "discard_pile": [],
        "exhaust_pile": [],
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 38,
                "max_hp": 42,
                "block": 0,
                "intent": "ATTACK",
                "intent_base_damage": 11,
                "powers": [{"id": "Strength", "amount": 0}],
            },
            {
                "name": "Cultist",
                "current_hp": 48,
                "max_hp": 48,
                "block": 0,
                "intent": "BUFF",
                "intent_base_damage": 0,
                "powers": [],
            },
        ],
        "potions": [
            {"name": "Fire Potion", "usable": True, "empty": False},
            {"name": "Empty Slot", "usable": False, "empty": True},
        ],
    }

    encoded = encoder.encode(sample_state)

    print("=== Encoded keys ===")
    for k, v in encoded.items():
        print(k, tuple(v.shape))

    flat = flatten_combat_obs(encoded)
    print("Flat obs shape:", tuple(flat.shape))
    print("Valid action mask:", encoded["valid_action_mask"])
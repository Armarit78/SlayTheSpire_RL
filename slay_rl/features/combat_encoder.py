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

# Ce vocab potion n'a pas besoin d'être exhaustif au caractère près.
# Le but ici est d'exposer une représentation stable et riche au modèle.
POTION_CLASS_TO_IDX = {
    "empty": 0,
    "attack_target": 1,
    "attack_aoe": 2,
    "block": 3,
    "strength": 4,
    "dexterity": 5,
    "draw": 6,
    "energy": 7,
    "artifact": 8,
    "intangible": 9,
    "utility": 10,
    "unknown": 11,
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


def clamp_signed(value: float, abs_max: float) -> float:
    if abs_max <= 0:
        return 0.0
    x = value / abs_max
    if x < -1.0:
        return -1.0
    if x > 1.0:
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
    - potions:               [max_potions, potion_feature_dim]
    - potion_mask:           [max_potions]
    - combat_context:        [combat_context_dim]
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
        self.max_potions = self.cfg.combat_obs.max_potions

        self.card_vocab_size = self.cfg.combat_obs.card_vocab_size
        self.relic_vocab_size = self.cfg.combat_obs.relic_vocab_size
        self.enemy_vocab_size = self.cfg.combat_obs.enemy_vocab_size

        self.total_actions = self.cfg.combat_action.total_actions

        # [card_one_hot..., cost_norm, upgraded, exhausts, ethereal, is_playable, has_target]
        self.card_feature_dim = self.card_vocab_size + 6

        # enemy_one_hot
        # + hp_ratio, hp_abs_norm, block_norm, alive
        # + intent_one_hot
        # + intent_hit_norm, intent_total_norm, multi_count_norm
        # + strength, weak, vulnerable, artifact, metallicize, regen, ritual
        self.enemy_feature_dim = (
            self.enemy_vocab_size
            + 4
            + len(INTENT_TO_IDX)
            + 3
            + 7
        )

        # potion_class_one_hot
        # + usable, requires_target, empty, rarity_common, rarity_uncommon, rarity_rare
        # + is_attack_like, is_defense_like, is_buff_like
        self.potion_feature_dim = len(POTION_CLASS_TO_IDX) + 9

        # riche mais compact
        self.combat_context_dim = self.cfg.combat_obs.combat_context_dim

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
        potions = self._extract_potions(state)

        player_scalars = self._encode_player(player, state, enemies, draw_pile, discard_pile, exhaust_pile, potions)
        hand_cards, hand_mask = self._encode_hand(hand)
        enemy_tensor, enemy_mask = self._encode_enemies(enemies)
        potion_tensor, potion_mask = self._encode_potions(potions)
        combat_context = self._encode_combat_context(state, player, enemies, hand, potions)

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
            "potions": torch.tensor(potion_tensor, dtype=torch.float32, device=device),
            "potion_mask": torch.tensor(potion_mask, dtype=torch.float32, device=device),
            "combat_context": torch.tensor(combat_context, dtype=torch.float32, device=device),
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
        potion_target_base = potion_base + self.max_potions

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
        for slot_idx in range(self.max_potions):
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
        energy = safe_float(player.get("energy", state.get("energy", state.get("player_energy", 0))))

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

            if cost < -1:
                cost = -1

            upgraded = 1 if c.get("upgraded", False) else 0
            exhausts = 1 if c.get("exhaust", c.get("exhausts", False)) else 0
            ethereal = 1 if c.get("isEthereal", c.get("ethereal", False)) else 0

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
            is_gone = bool(m.get("is_gone", False))
            half_dead = bool(m.get("half_dead", False))

            alive = 0 if is_dead or escaped or is_gone or half_dead or hp <= 0 else 1

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

    def _encode_player(
        self,
        player: ParsedPlayer,
        state: Dict[str, Any],
        enemies: List[ParsedEnemy],
        draw_pile: List[Dict[str, Any]],
        discard_pile: List[Dict[str, Any]],
        exhaust_pile: List[Dict[str, Any]],
        potions: List[Dict[str, Any]],
    ) -> List[float]:
        powers = self._extract_powers(player.raw)
        combat_meta = state.get("combat_meta", {}) or {}

        incoming_damage = self._estimate_incoming_damage_from_parsed_enemies(enemies)
        alive_enemies = sum(e.alive for e in enemies)
        usable_potions = sum(
            1 for p in potions
            if not p.get("empty", False) and p.get("usable", True)
        )

        out = [
            clamp01_from_max(player.hp, max(player.max_hp, 1.0)),             # 0
            min(player.hp / 200.0, 1.0),                                      # 1
            min(player.block / 120.0, 1.0),                                   # 2
            min(player.energy / 10.0, 1.0),                                   # 3
            clamp_signed(player.strength, 20.0),                              # 4
            clamp_signed(player.dexterity, 20.0),                             # 5
            min(max(player.weak, 0.0) / 6.0, 1.0),                            # 6
            min(max(player.vulnerable, 0.0) / 6.0, 1.0),                      # 7
            min(max(player.frail, 0.0) / 6.0, 1.0),                           # 8
            min(max(self._power_amount(powers, ["Artifact"]), 0.0) / 5.0, 1.0),      # 9
            min(max(self._power_amount(powers, ["Metallicize"]), 0.0) / 10.0, 1.0),   # 10
            min(max(self._power_amount(powers, ["Plated Armor"]), 0.0) / 10.0, 1.0),  # 11
            min(max(self._power_amount(powers, ["Rage"]), 0.0) / 10.0, 1.0),          # 12
            min(max(self._power_amount(powers, ["Combust"]), 0.0) / 12.0, 1.0),       # 13
            min(max(self._power_amount(powers, ["Dark Embrace"]), 0.0) / 3.0, 1.0),   # 14
            min(max(self._power_amount(powers, ["Evolve"]), 0.0) / 3.0, 1.0),          # 15
            min(max(self._power_amount(powers, ["Feel No Pain"]), 0.0) / 8.0, 1.0),    # 16
            min(max(self._power_amount(powers, ["Fire Breathing"]), 0.0) / 12.0, 1.0), # 17
            min(max(self._power_amount(powers, ["Rupture"]), 0.0) / 5.0, 1.0),         # 18
            min(max(self._power_amount(powers, ["Juggernaut"]), 0.0) / 12.0, 1.0),     # 19
            1.0 if self._has_power(powers, "Barricade") else 0.0,                       # 20
            1.0 if self._has_power(powers, "Corruption") else 0.0,                      # 21
            min(max(combat_meta.get("double_tap_charges", 0), 0) / 3.0, 1.0),          # 22
            min(incoming_damage / 60.0, 1.0),                                            # 23
            min(alive_enemies / max(self.max_enemies, 1), 1.0),                          # 24
        ]

        # garde la taille sous contrôle si config diverge
        if len(out) != self.cfg.combat_obs.player_scalar_dim:
            raise ValueError(
                f"player_scalars dim mismatch: got {len(out)}, "
                f"expected {self.cfg.combat_obs.player_scalar_dim}"
            )

        return out

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

            intent_one_hot = one_hot(
                INTENT_TO_IDX.get(enemy.intent, INTENT_TO_IDX["UNKNOWN"]),
                len(INTENT_TO_IDX),
            )

            powers = self._extract_powers(enemy.raw)

            intent_hits = self._extract_enemy_intent_hits(enemy.raw)
            intent_total_damage = max(0.0, enemy.intent_base_damage * intent_hits)

            max_hp = max(enemy.max_hp, 1.0)
            row = enemy_one_hot + [
                clamp01_from_max(enemy.hp, max_hp),
                min(enemy.hp / 250.0, 1.0),
                min(enemy.block / 120.0, 1.0),
                1.0 if enemy.alive else 0.0,
            ] + intent_one_hot + [
                min(enemy.intent_base_damage / 40.0, 1.0),
                min(intent_total_damage / 80.0, 1.0),
                min(intent_hits / 6.0, 1.0),
                clamp_signed(enemy.strength, 20.0),
                min(max(enemy.weak, 0.0) / 6.0, 1.0),
                min(max(enemy.vulnerable, 0.0) / 6.0, 1.0),
                min(max(self._power_amount(powers, ["Artifact"]), 0.0) / 5.0, 1.0),
                min(max(self._power_amount(powers, ["Metallicize"]), 0.0) / 10.0, 1.0),
                min(max(self._power_amount(powers, ["Regenerate", "Regen"]), 0.0) / 15.0, 1.0),
                min(max(self._power_amount(powers, ["Ritual"]), 0.0) / 10.0, 1.0),
            ]

            rows.append(row)
            mask.append(1.0 if enemy.alive else 0.0)

        while len(rows) < self.max_enemies:
            rows.append([0.0] * self.enemy_feature_dim)
            mask.append(0.0)

        if len(rows[0]) != self.cfg.combat_obs.enemy_scalar_dim:
            raise ValueError(
                f"enemy feature dim mismatch: got {len(rows[0])}, "
                f"expected {self.cfg.combat_obs.enemy_scalar_dim}"
            )

        return rows, mask

    def _encode_potions(self, potions: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[float]]:
        rows: List[List[float]] = []
        mask: List[float] = []

        for potion in potions[: self.max_potions]:
            potion_class = self._classify_potion(potion)
            potion_idx = POTION_CLASS_TO_IDX.get(potion_class, POTION_CLASS_TO_IDX["unknown"])
            one_hot_class = one_hot(potion_idx, len(POTION_CLASS_TO_IDX))

            rarity = str(potion.get("rarity", "Common")).lower()
            usable = 1.0 if potion.get("usable", True) and not potion.get("empty", False) else 0.0
            requires_target = 1.0 if potion.get("requires_target", False) else 0.0
            empty = 1.0 if potion.get("empty", False) else 0.0

            is_attack_like = 1.0 if potion_class in {"attack_target", "attack_aoe"} else 0.0
            is_defense_like = 1.0 if potion_class == "block" else 0.0
            is_buff_like = 1.0 if potion_class in {
                "strength", "dexterity", "artifact", "intangible", "energy", "draw", "utility"
            } else 0.0

            row = one_hot_class + [
                usable,
                requires_target,
                empty,
                1.0 if rarity == "common" else 0.0,
                1.0 if rarity == "uncommon" else 0.0,
                1.0 if rarity == "rare" else 0.0,
                is_attack_like,
                is_defense_like,
                is_buff_like,
            ]

            rows.append(row)
            mask.append(0.0 if empty else 1.0)

        while len(rows) < self.max_potions:
            rows.append([0.0] * self.potion_feature_dim)
            mask.append(0.0)

        if len(rows[0]) != self.cfg.combat_obs.potion_scalar_dim:
            raise ValueError(
                f"potion feature dim mismatch: got {len(rows[0])}, "
                f"expected {self.cfg.combat_obs.potion_scalar_dim}"
            )

        return rows, mask

    def _encode_combat_context(
        self,
        state: Dict[str, Any],
        player: ParsedPlayer,
        enemies: List[ParsedEnemy],
        hand: List[ParsedCard],
        potions: List[Dict[str, Any]],
    ) -> List[float]:
        combat_meta = state.get("combat_meta", {}) or {}

        alive_enemies = [e for e in enemies if e.alive]
        incoming_damage = self._estimate_incoming_damage_from_parsed_enemies(enemies)
        attacks_in_hand = sum(1 for c in hand if self._card_type(c.raw) == "ATTACK")
        skills_in_hand = sum(1 for c in hand if self._card_type(c.raw) == "SKILL")
        powers_in_hand = sum(1 for c in hand if self._card_type(c.raw) == "POWER")
        playable_cards = sum(1 for c in hand if c.is_playable)
        targeted_cards = sum(1 for c in hand if c.has_target)
        usable_potions = sum(
            1 for p in potions
            if not p.get("empty", False) and p.get("usable", True)
        )

        lethalable_enemies = 0
        estimated_best_single_hit = self._estimate_best_single_target_damage(hand, player)
        for enemy in alive_enemies:
            effective_hp = enemy.hp + enemy.block
            if estimated_best_single_hit >= effective_hp:
                lethalable_enemies += 1

        context = [
            min(safe_float(state.get("turn", 1)) / 20.0, 1.0),                                  # 0
            min(max(combat_meta.get("cards_played_this_turn", 0), 0) / 10.0, 1.0),            # 1
            min(max(combat_meta.get("attacks_played_this_turn", 0), 0) / 10.0, 1.0),          # 2
            min(max(combat_meta.get("attack_counter", 0), 0) / 20.0, 1.0),                    # 3
            min(max(combat_meta.get("double_tap_charges", 0), 0) / 3.0, 1.0),                 # 4
            min(max(combat_meta.get("last_x_energy_spent", 0), 0) / 5.0, 1.0),                # 5
            1.0 if combat_meta.get("cannot_draw_more_this_turn", False) else 0.0,             # 6
            1.0 if combat_meta.get("next_attack_double", False) else 0.0,                      # 7
            1.0 if combat_meta.get("first_attack_done", False) else 0.0,                       # 8
            1.0 if combat_meta.get("is_elite", False) else 0.0,                                # 9
            1.0 if combat_meta.get("is_boss", False) else 0.0,                                 # 10
            min(incoming_damage / 80.0, 1.0),                                                  # 11
            min(len(alive_enemies) / max(self.max_enemies, 1), 1.0),                           # 12
            min(attacks_in_hand / max(self.max_hand_cards, 1), 1.0),                           # 13
            min(skills_in_hand / max(self.max_hand_cards, 1), 1.0),                            # 14
            min(powers_in_hand / max(self.max_hand_cards, 1), 1.0),                            # 15
            min(playable_cards / max(self.max_hand_cards, 1), 1.0),                            # 16
            min(usable_potions / max(self.max_potions, 1), 1.0),                               # 17
        ]

        if len(context) != self.combat_context_dim:
            raise ValueError(
                f"combat_context dim mismatch: got {len(context)}, expected {self.combat_context_dim}"
            )

        return context

    # ========================================================
    # Counts / bag-of-cards
    # ========================================================

    def _count_cards(self, cards: List[Dict[str, Any]]) -> List[float]:
        counts = [0.0] * self.card_vocab_size
        for c in cards:
            card_id = self._normalize_card_id(c.get("id", c.get("card_id", c.get("name", ""))))
            idx = CARD_TO_IDX.get(card_id, -1)
            if 0 <= idx < self.card_vocab_size:
                counts[idx] += 1.0

        # soft cap to keep the scale bounded
        return [min(x / 8.0, 1.0) for x in counts]

    def _encode_relics(self, relics: List[Dict[str, Any]]) -> List[float]:
        vec = [0.0] * self.relic_vocab_size
        for relic in relics:
            name = self._extract_relic_name(relic)
            idx = RELIC_TO_IDX.get(name, -1)
            if 0 <= idx < self.relic_vocab_size:
                vec[idx] = 1.0
        return vec

    # ========================================================
    # Tactical estimates
    # ========================================================

    def _estimate_incoming_damage_from_parsed_enemies(self, enemies: List[ParsedEnemy]) -> float:
        total = 0.0
        for enemy in enemies:
            if not enemy.alive:
                continue
            hits = self._extract_enemy_intent_hits(enemy.raw)
            total += max(0.0, enemy.intent_base_damage) * hits
        return total

    def _estimate_best_single_target_damage(self, hand: List[ParsedCard], player: ParsedPlayer) -> float:
        best = 0.0
        strength = player.strength

        for card in hand:
            if not card.is_playable:
                continue

            raw = card.raw
            card_type = self._card_type(raw)
            if card_type != "ATTACK":
                continue

            base = safe_float(raw.get("damage", 0))
            hits = safe_float(raw.get("hits", 1))

            # très simple mais utile
            if raw.get("damage_from_block", False):
                est = player.block
            elif raw.get("strength_mult") is not None:
                est = base + strength * safe_float(raw.get("strength_mult", 1))
            elif raw.get("damage_per_exhausted_card") is not None:
                # estimation prudente
                est = safe_float(raw.get("damage_per_exhausted_card", 0)) * 3.0
            else:
                est = (base + strength) * max(hits, 1.0)

            best = max(best, est)

        return best

    def _extract_enemy_intent_hits(self, enemy_raw: Dict[str, Any]) -> float:
        for key in ["intent_hits", "intentHits", "move_hits", "hits", "multi"]:
            if key in enemy_raw:
                hits = safe_float(enemy_raw.get(key, 1.0), 1.0)
                return max(1.0, hits)
        return 1.0

    # ========================================================
    # Extraction helpers
    # ========================================================

    def _extract_card_list(self, state: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
        for key in keys:
            if key in state and isinstance(state[key], list):
                return state[key]

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
                for n in target_names:
                    if n in power_name:
                        total += safe_float(p.get("amount", p.get("stack", 0)))
                        break
        return total

    def _has_power(self, powers: List[Dict[str, Any]], name: str) -> bool:
        target = name.lower()
        for p in powers:
            power_name = str(p.get("id", p.get("name", ""))).lower()
            if power_name == target or target in power_name:
                return True
        return False

    def _extract_relic_name(self, relic: Dict[str, Any]) -> str:
        return str(relic.get("name", relic.get("id", "UnknownRelic")))

    def _classify_potion(self, potion: Dict[str, Any]) -> str:
        if potion.get("empty", False):
            return "empty"

        name = str(potion.get("name", potion.get("id", "Potion"))).lower()
        requires_target = bool(potion.get("requires_target", False))

        if "fire potion" in name or "poison potion" in name or "fear potion" in name or "explosive" in name:
            return "attack_target" if requires_target else "attack_aoe"
        if "block" in name or "dexterity" in name or "essence of steel" in name:
            return "block"
        if "strength" in name or "cultist potion" in name:
            return "strength"
        if "swift potion" in name or "gamblers brew" in name:
            return "draw"
        if "energy" in name:
            return "energy"
        if "artifact" in name:
            return "artifact"
        if "ghost" in name:
            return "intangible"
        if "dexterity" in name:
            return "dexterity"
        if "power potion" in name or "skill potion" in name or "attack potion" in name or "potion" in name:
            return "utility"

        if requires_target:
            return "attack_target"

        return "unknown"

    def _normalize_card_id(self, raw_id: str) -> str:
        card_id = str(raw_id).strip()

        if card_id in CARD_TO_IDX:
            return card_id

        candidates = [
            card_id.replace(" ", "_"),
            card_id.replace("_", " "),
            card_id.title().replace(" ", "_"),
        ]
        for c in candidates:
            if c in CARD_TO_IDX:
                return c

        return card_id

    def _card_type(self, card: Dict[str, Any]) -> str:
        return str(card.get("type", "")).upper()

    def _card_has_target(self, card: Dict[str, Any]) -> int:
        """
        Best effort target inference.
        """
        if "has_target" in card:
            return 1 if card["has_target"] else 0

        if "target" in card:
            target = str(card["target"]).upper()
            if target in {"ENEMY", "SINGLE", "MONSTER"}:
                return 1
            if target in {"ALL", "SELF", "NONE"}:
                return 0

        if card.get("targeted", False):
            return 1
        if card.get("aoe_damage", None) is not None:
            return 0
        if card.get("apply_weak_all", None) is not None:
            return 0
        if card.get("apply_vulnerable_all", None) is not None:
            return 0
        if card.get("x_aoe_damage", None) is not None:
            return 0

        card_id = self._normalize_card_id(card.get("id", card.get("card_id", card.get("name", ""))))

        known_target_cards = {
            "Strike_R",
            "Bash",
            "Body Slam",
            "Clash",
            "Clothesline",
            "Headbutt",
            "Heavy Blade",
            "Iron Wave",
            "Perfected Strike",
            "Pommel Strike",
            "Twin Strike",
            "Wild Strike",
            "Blood for Blood",
            "Carnage",
            "Disarm",
            "Dropkick",
            "Hemokinesis",
            "Pummel",
            "Rampage",
            "Reckless Charge",
            "Searing Blow",
            "Sever Soul",
            "Spot Weakness",
            "Uppercut",
            "Bludgeon",
            "Feed",
            "Fiend Fire",
        }

        known_no_target_cards = {
            "Defend_R",
            "Anger",
            "Armaments",
            "Cleave",
            "Flex",
            "Havoc",
            "Shrug It Off",
            "Sword Boomerang",
            "Thunderclap",
            "True Grit",
            "Warcry",
            "Battle Trance",
            "Bloodletting",
            "Burning Pact",
            "Combust",
            "Dark Embrace",
            "Dual Wield",
            "Entrench",
            "Evolve",
            "Feel No Pain",
            "Fire Breathing",
            "Flame Barrier",
            "Ghostly Armor",
            "Infernal Blade",
            "Inflame",
            "Intimidate",
            "Metallicize",
            "Power Through",
            "Rage",
            "Rupture",
            "Second Wind",
            "Seeing Red",
            "Shockwave",
            "Whirlwind",
            "Barricade",
            "Berserk",
            "Brutality",
            "Corruption",
            "Demon Form",
            "Double Tap",
            "Exhume",
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

        card_type = self._card_type(card)
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
    """
    parts = [
        encoded["player_scalars"].flatten(),
        encoded["hand_cards"].flatten(),
        encoded["hand_mask"].flatten(),
        encoded["enemies"].flatten(),
        encoded["enemy_mask"].flatten(),
        encoded["potions"].flatten(),
        encoded["potion_mask"].flatten(),
        encoded["combat_context"].flatten(),
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
        "turn": 2,
        "energy": 3,
        "player": {
            "current_hp": 72,
            "max_hp": 80,
            "block": 5,
            "powers": [
                {"id": "Strength", "amount": 2},
                {"id": "Dexterity", "amount": 1},
                {"id": "Artifact", "amount": 1},
                {"id": "Rage", "amount": 3},
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
        "combat_meta": {
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
                "powers": [{"id": "Ritual", "amount": 3}],
            },
        ],
        "potions": [
            {"name": "Fire Potion", "usable": True, "empty": False, "requires_target": True, "rarity": "Common"},
            {"name": "Dexterity Potion", "usable": True, "empty": False, "requires_target": False, "rarity": "Uncommon"},
            {"name": "Empty Slot", "usable": False, "empty": True, "requires_target": False},
        ],
    }

    encoded = encoder.encode(sample_state)

    print("=== Encoded keys ===")
    for k, v in encoded.items():
        print(k, tuple(v.shape))

    flat = flatten_combat_obs(encoded)
    print("Flat obs shape:", tuple(flat.shape))
    print("Valid action mask:", encoded["valid_action_mask"])
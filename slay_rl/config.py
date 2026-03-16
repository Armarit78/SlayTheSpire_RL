from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

SUPPORTED_BUILD_TARGETS = [
    "ironclad_strength",
    "ironclad_block",
    "ironclad_exhaust",
]

# Full Ironclad combat vocab + helper status cards used by Ironclad.
IRONCLAD_CARD_VOCAB = [
    "Strike_R",
    "Defend_R",
    "Bash",
    "Anger",
    "Armaments",
    "Body Slam",
    "Clash",
    "Cleave",
    "Clothesline",
    "Flex",
    "Havoc",
    "Headbutt",
    "Heavy Blade",
    "Iron Wave",
    "Perfected Strike",
    "Pommel Strike",
    "Shrug It Off",
    "Sword Boomerang",
    "Thunderclap",
    "True Grit",
    "Twin Strike",
    "Warcry",
    "Wild Strike",
    "Battle Trance",
    "Blood for Blood",
    "Bloodletting",
    "Burning Pact",
    "Carnage",
    "Combust",
    "Dark Embrace",
    "Disarm",
    "Dropkick",
    "Dual Wield",
    "Entrench",
    "Evolve",
    "Feel No Pain",
    "Fire Breathing",
    "Flame Barrier",
    "Ghostly Armor",
    "Hemokinesis",
    "Infernal Blade",
    "Inflame",
    "Intimidate",
    "Metallicize",
    "Power Through",
    "Pummel",
    "Rage",
    "Rampage",
    "Reckless Charge",
    "Rupture",
    "Searing Blow",
    "Second Wind",
    "Seeing Red",
    "Sentinel",
    "Sever Soul",
    "Shockwave",
    "Spot Weakness",
    "Uppercut",
    "Whirlwind",
    "Barricade",
    "Berserk",
    "Bludgeon",
    "Brutality",
    "Corruption",
    "Demon Form",
    "Double Tap",
    "Exhume",
    "Feed",
    "Fiend Fire",
    "Immolate",
    "Impervious",
    "Juggernaut",
    "Limit Break",
    "Offering",
    "Reaper",

    # status cards
    "Wound",
    "Dazed",
    "Burn",
    "Slimed",
    "Void",

    # curse cards
    "AscendersBane",
    "Clumsy",
    "Curse of the Bell",
    "Decay",
    "Doubt",
    "Injury",
    "Normality",
    "Pain",
    "Parasite",
    "Pride",
    "Regret",
    "Shame",
    "Writhe",
]

IRONCLAD_RELIC_VOCAB = [
    # Ironclad-specific
    "Burning Blood",
    "Red Skull",
    "Paper Phrog",
    "Self-Forming Clay",
    "Champion Belt",
    "Charon's Ashes",
    "Magic Flower",
    "Black Blood",
    "Mark of Pain",
    "Runic Cube",
    "Brimstone",

    # Shared / all characters
    "Circlet",
    "Neow's Blessing",

    # Common
    "Akabeko",
    "Anchor",
    "Ancient Tea Set",
    "Art of War",
    "Bag of Marbles",
    "Bag of Preparation",
    "Blood Vial",
    "Bronze Scales",
    "Centennial Puzzle",
    "Ceramic Fish",
    "Dream Catcher",
    "Happy Flower",
    "Juzu Bracelet",
    "Lantern",
    "Maw Bank",
    "Meal Ticket",
    "Nunchaku",
    "Oddly Smooth Stone",
    "Omamori",
    "Orichalcum",
    "Pen Nib",
    "Potion Belt",
    "Preserved Insect",
    "Regal Pillow",
    "Smiling Mask",
    "Strawberry",
    "The Boot",
    "Tiny Chest",
    "Toy Ornithopter",
    "Vajra",
    "War Paint",
    "Whetstone",

    # Uncommon
    "Blue Candle",
    "Bottled Flame",
    "Bottled Lightning",
    "Bottled Tornado",
    "Darkstone Periapt",
    "Eternal Feather",
    "Frozen Egg",
    "Gremlin Horn",
    "Horn Cleat",
    "Ink Bottle",
    "Kunai",
    "Letter Opener",
    "Matryoshka",
    "Meat on the Bone",
    "Mercury Hourglass",
    "Molten Egg",
    "Mummified Hand",
    "Ornamental Fan",
    "Pantograph",
    "Pear",
    "Question Card",
    "Shuriken",
    "Singing Bowl",
    "Strike Dummy",
    "Sundial",
    "The Courier",
    "Toxic Egg",
    "White Beast Statue",

    # Rare
    "Bird-Faced Urn",
    "Calipers",
    "Captain's Wheel",
    "Dead Branch",
    "Du-Vu Doll",
    "Fossilized Helix",
    "Gambling Chip",
    "Ginger",
    "Girya",
    "Ice Cream",
    "Incense Burner",
    "Lizard Tail",
    "Mango",
    "Old Coin",
    "Peace Pipe",
    "Pocketwatch",
    "Prayer Wheel",
    "Shovel",
    "Stone Calendar",
    "Thread and Needle",
    "Torii",
    "Tungsten Rod",
    "Turnip",
    "Unceasing Top",
    "Winged Greaves",

    # Shop
    "Cauldron",
    "Chemical X",
    "Clockwork Souvenir",
    "Dolly's Mirror",
    "Frozen Eye",
    "Hand Drill",
    "Lee's Waffle",
    "Medical Kit",
    "Membership Card",
    "Orange Pellets",
    "Orrery",
    "Prismatic Shard",
    "Sling of Courage",
    "Strange Spoon",
    "The Abacus",
    "Toolbox",

    # Boss
    "Astrolabe",
    "Black Star",
    "Busted Crown",
    "Calling Bell",
    "Coffee Dripper",
    "Cursed Key",
    "Ectoplasm",
    "Empty Cage",
    "Fusion Hammer",
    "Pandora's Box",
    "Philosopher's Stone",
    "Runic Dome",
    "Runic Pyramid",
    "Sacred Bark",
    "Slaver's Collar",
    "Snecko Eye",
    "Sozu",
    "Tiny House",
    "Velvet Choker",

    # Event
    "Bloody Idol",
    "Cultist Headpiece",
    "Enchiridion",
    "Face of Cleric",
    "Golden Idol",
    "Gremlin Visage",
    "Mark of the Bloom",
    "Mutagenic Strength",
    "N'loth's Gift",
    "Nloth's Hungry Face",
    "Necronomicon",
    "Nilry's Codex",
    "Odd Mushroom",
    "Red Mask",
    "Spirit Poop",
    "Ssserpent Head",
    "Warped Tongs",
]

ENEMY_VOCAB = [
    # Act 1 normal monsters
    "Jaw Worm",
    "Cultist",
    "Red Louse",
    "Green Louse",
    "Fungi Beast",
    "Acid Slime (S)",
    "Acid Slime (M)",
    "Acid Slime (L)",
    "Spike Slime (S)",
    "Spike Slime (M)",
    "Spike Slime (L)",
    "Blue Slaver",
    "Red Slaver",
    "Looter",
    "Mad Gremlin",
    "Sneaky Gremlin",
    "Fat Gremlin",
    "Shield Gremlin",
    "Gremlin Wizard",

    # existing elites / bosses
    "Gremlin Nob",
    "Lagavulin",
    "Sentry",
    "The Guardian",
    "Slime Boss",
    "Hexaghost",
]

CARD_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(IRONCLAD_CARD_VOCAB)}
RELIC_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(IRONCLAD_RELIC_VOCAB)}
ENEMY_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ENEMY_VOCAB)}


@dataclass
class CombatObsConfig:
    max_hand_cards: int = 10
    max_draw_pile_cards: int = 40
    max_discard_pile_cards: int = 40
    max_exhaust_pile_cards: int = 40
    max_enemies: int = 5
    max_potions: int = 5

    # Dims alignées avec slay_rl/features/combat_encoder.py
    # enemy features = enemy_vocab_size + 4 + len(INTENT_TO_IDX) + 3 + 7 (avec le vocab actuel => 57)
    # potion features = len(POTION_CLASS_TO_IDX) + 9 => 21
    player_scalar_dim: int = 25
    enemy_scalar_dim: int = 57
    potion_scalar_dim: int = 21
    combat_context_dim: int = 18

    @property
    def card_vocab_size(self) -> int:
        return len(IRONCLAD_CARD_VOCAB)

    @property
    def relic_vocab_size(self) -> int:
        return len(IRONCLAD_RELIC_VOCAB)

    @property
    def enemy_vocab_size(self) -> int:
        return len(ENEMY_VOCAB)


@dataclass
class MacroObsConfig:
    max_deck_cards: int = 60
    max_map_nodes_visible: int = 20
    max_shop_items: int = 10
    max_reward_cards: int = 3
    player_scalar_dim: int = 6
    build_target_dim: int = len(SUPPORTED_BUILD_TARGETS)

    @property
    def card_vocab_size(self) -> int:
        return len(IRONCLAD_CARD_VOCAB)

    @property
    def relic_vocab_size(self) -> int:
        return len(IRONCLAD_RELIC_VOCAB)


@dataclass
class CombatActionConfig:
    max_play_actions: int = 10
    max_target_actions: int = 5
    allow_end_turn: bool = True
    allow_potions: bool = True
    total_actions: int = 91


@dataclass
class MacroActionConfig:
    max_path_choices: int = 6
    max_reward_card_choices: int = 4
    max_shop_choices: int = 12
    max_rest_choices: int = 4
    total_actions: int = 32


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_eps: float = 0.2
    value_coef: float = 0.25
    entropy_coef: float = 0.012
    lr: float = 2e-4
    batch_size: int = 1024
    rollout_steps: int = 8192
    epochs: int = 4
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 32
    device: str = "cuda"
    run_name: str = "experiment_4"

    total_updates: int = 1200
    save_every: int = 50
    log_every: int = 10
    eval_every: int = 20

    eval_num_episodes: int = 50
    eval_seed_start: int = 100000

    robust_eval_every: int = 50
    robust_eval_num_episodes: int = 100
    robust_eval_seed_start: int = 200000

    num_envs: int = 8


@dataclass
class CurriculumPhaseConfig:
    until_update: int
    elite_chance: float
    boss_chance: float


@dataclass
class CombatCurriculumConfig:
    enabled: bool = True
    phases: list[CurriculumPhaseConfig] = field(default_factory=lambda: [
        CurriculumPhaseConfig(until_update=300, elite_chance=0.10, boss_chance=0.00),
        CurriculumPhaseConfig(until_update=700, elite_chance=0.13, boss_chance=0.02),
        CurriculumPhaseConfig(until_update=10_000_000, elite_chance=0.17, boss_chance=0.06),
    ])


@dataclass
class CombatRewardConfig:
    # Core outcome
    damage_dealt_scale: float = 0.070
    damage_taken_scale: float = -0.120
    kill_enemy_bonus: float = 1.25
    win_combat_bonus: float = 6.0
    lose_combat_penalty: float = -6.0
    illegal_action_penalty: float = -0.50

    # Tempo / stalling
    end_turn_small_penalty: float = -0.06
    wasted_energy_penalty_scale: float = -0.035
    good_energy_use_reward_scale: float = 0.006

    # Defensive quality
    useful_block_scale: float = 0.012
    overblock_penalty_scale: float = -0.004
    survival_hp_ratio_scale: float = 0.050

    # Enemy pressure / control
    threat_reduction_scale: float = 0.030
    enemy_vulnerable_scale: float = 0.040
    enemy_weak_scale: float = 0.030

    # Player buffs
    strength_gain_scale: float = 0.015
    dex_gain_scale: float = 0.010
    metallicize_gain_scale: float = 0.008
    plated_gain_scale: float = 0.008
    artifact_gain_scale: float = 0.010
    intangible_gain_scale: float = 0.040

    # Setup / sequencing
    rage_setup_scale: float = 0.015
    double_tap_setup_scale: float = 0.035
    inflame_setup_scale: float = 0.020
    spot_weakness_setup_scale: float = 0.020
    corruption_setup_scale: float = 0.025
    barricade_setup_scale: float = 0.020
    feel_no_pain_setup_scale: float = 0.020
    dark_embrace_setup_scale: float = 0.015
    demon_form_setup_scale: float = 0.030
    evolve_setup_scale: float = 0.015
    combust_setup_scale: float = 0.010
    juggernaut_setup_scale: float = 0.015
    rupture_setup_scale: float = 0.010

    # Lethal / focus
    lethal_reward_scale: float = 0.080
    near_lethal_reward_scale: float = 0.030

    # Hand pollution / self-damage
    status_curse_hand_reduce_scale: float = 0.050
    self_bad_hp_loss_scale: float = -0.030

    # Potions
    potion_low_threat_penalty: float = -0.040
    potion_medium_threat_penalty: float = -0.015
    potion_high_threat_penalty: float = -0.003
    potion_emergency_bonus: float = 0.015
    potion_lethal_bonus: float = 0.020


@dataclass
class MacroRewardConfig:
    floor_progress_bonus: float = 0.25
    elite_bonus: float = 1.0
    boss_bonus: float = 3.0
    death_penalty: float = -4.0
    deck_coherence_scale: float = 0.15
    skip_good_bonus: float = 0.05
    bad_pick_penalty: float = -0.05
    path_to_elite_bonus: float = 0.10


@dataclass
class GameConfig:
    character: str = "IRONCLAD"
    ascension: int = 0
    build_target: str = "ironclad_strength"
    verbose: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    max_act: int = 1
    max_floor: int = 17


@dataclass
class Config:
    game: GameConfig = field(default_factory=GameConfig)
    combat_obs: CombatObsConfig = field(default_factory=CombatObsConfig)
    macro_obs: MacroObsConfig = field(default_factory=MacroObsConfig)
    combat_action: CombatActionConfig = field(default_factory=CombatActionConfig)
    macro_action: MacroActionConfig = field(default_factory=MacroActionConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    combat_curriculum: CombatCurriculumConfig = field(default_factory=CombatCurriculumConfig)
    combat_reward: CombatRewardConfig = field(default_factory=CombatRewardConfig)
    macro_reward: MacroRewardConfig = field(default_factory=MacroRewardConfig)


def get_default_config() -> Config:
    cfg = Config()
    if cfg.game.build_target not in SUPPORTED_BUILD_TARGETS:
        raise ValueError(f"Unknown build target: {cfg.game.build_target}. Supported: {SUPPORTED_BUILD_TARGETS}")
    return cfg

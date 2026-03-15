from __future__ import annotations


def test_reset_should_use_4_bonus_cards_and_1_extra_relic_for_normal_fight(backend, monkeypatch):
    calls = {}

    def fake_sample_enemy_group_with_meta():
        return ([backend._make_enemy("Jaw Worm")], False, False)

    def fake_build_random_player_deck(bonus_cards: int = 4):
        calls["bonus_cards"] = bonus_cards
        return [backend._starter_deck_pool()[0] for _ in range(12)]

    def fake_build_starting_relics(extra_relics: int = 1):
        calls["extra_relics"] = extra_relics
        relics = [{"name": "Burning Blood"}]
        for i in range(extra_relics):
            relics.append({"name": f"Test Relic {i}"})
        return relics

    monkeypatch.setattr(backend, "_sample_enemy_group_with_meta", fake_sample_enemy_group_with_meta)
    monkeypatch.setattr(backend, "_build_random_player_deck", fake_build_random_player_deck)
    monkeypatch.setattr(backend, "_build_starting_relics", fake_build_starting_relics)

    state = backend.reset()

    assert calls["bonus_cards"] == 4
    assert calls["extra_relics"] == 1
    assert state["combat_meta"]["is_elite"] is False
    assert state["combat_meta"]["is_boss"] is False
    assert len(state["player"]["relics"]) == 2  # Burning Blood + 1


def test_reset_should_use_5_bonus_cards_and_2_extra_relics_for_elite_fight(backend, monkeypatch):
    calls = {}

    def fake_sample_enemy_group_with_meta():
        return ([backend._make_enemy("Gremlin Nob")], True, False)

    def fake_build_random_player_deck(bonus_cards: int = 4):
        calls["bonus_cards"] = bonus_cards
        return [backend._starter_deck_pool()[0] for _ in range(14)]

    def fake_build_starting_relics(extra_relics: int = 1):
        calls["extra_relics"] = extra_relics
        relics = [{"name": "Burning Blood"}]
        for i in range(extra_relics):
            relics.append({"name": f"Elite Relic {i}"})
        return relics

    monkeypatch.setattr(backend, "_sample_enemy_group_with_meta", fake_sample_enemy_group_with_meta)
    monkeypatch.setattr(backend, "_build_random_player_deck", fake_build_random_player_deck)
    monkeypatch.setattr(backend, "_build_starting_relics", fake_build_starting_relics)

    state = backend.reset()

    assert calls["bonus_cards"] == 5
    assert calls["extra_relics"] == 2
    assert state["combat_meta"]["is_elite"] is True
    assert state["combat_meta"]["is_boss"] is False
    assert len(state["player"]["relics"]) == 3  # Burning Blood + 2


def test_reset_should_use_6_bonus_cards_and_3_extra_relics_for_boss_fight(backend, monkeypatch):
    calls = {}

    def fake_sample_enemy_group_with_meta():
        return ([backend._make_enemy("Hexaghost")], False, True)

    def fake_build_random_player_deck(bonus_cards: int = 4):
        calls["bonus_cards"] = bonus_cards
        return [backend._starter_deck_pool()[0] for _ in range(15)]

    def fake_build_starting_relics(extra_relics: int = 1):
        calls["extra_relics"] = extra_relics
        relics = [{"name": "Burning Blood"}]
        for i in range(extra_relics):
            relics.append({"name": f"Boss Relic {i}"})
        return relics

    monkeypatch.setattr(backend, "_sample_enemy_group_with_meta", fake_sample_enemy_group_with_meta)
    monkeypatch.setattr(backend, "_build_random_player_deck", fake_build_random_player_deck)
    monkeypatch.setattr(backend, "_build_starting_relics", fake_build_starting_relics)

    state = backend.reset()

    assert calls["bonus_cards"] == 6
    assert calls["extra_relics"] == 3
    assert state["combat_meta"]["is_elite"] is False
    assert state["combat_meta"]["is_boss"] is True
    assert len(state["player"]["relics"]) == 4  # Burning Blood + 3
from slay_rl.run_controller import run_game, RunController, save_episode
from slay_rl.train.train_combat import train_combat
from slay_rl.config import get_default_config


def main():
    # =====================================================
    # CONFIG UTILISATEUR
    # =====================================================

    mode = "play_model"
    # modes possibles:
    # "play_rule"
    # "play_model"
    # "play_random"
    # "train"

    render = False
    episodes = 1000

    # chemin d'un modèle entraîné
    checkpoint_path = "checkpoints/experiment_3/combat_model_best_robust.pt"

    # pour play_model : test stable
    deterministic_model = True

    # =====================================================
    # TRAIN
    # =====================================================

    if mode == "train":
        print("\n=== TRAINING START ===")
        train_combat()
        return

    # =====================================================
    # PLAY RULE / RANDOM
    # =====================================================

    if mode == "play_rule":
        print(f"\n=== RUN MODE: {mode} ===")
        for ep in range(episodes):
            print(f"\n----- Episode {ep + 1} -----")
            result = run_game(
                agent_type="rule",
                render=render,
                seed=ep,
            )
            print("\nEpisode resultv2.txt:")
            print(result.to_dict())
            save_episode(result, ep + 1, mode)
        return

    if mode == "play_random":
        print(f"\n=== RUN MODE: {mode} ===")

        results = []

        for ep in range(episodes):
            print(f"\n----- Episode {ep + 1} -----")
            result = run_game(
                agent_type="random",
                render=render,
                seed=ep,
            )

            results.append(result)

            print("\nEpisode resultv2.txt:")
            print(result.to_dict())
            save_episode(result, ep + 1, mode)

        # ==============================
        # GLOBAL STATS
        # ==============================
        total_rewards = [r.total_reward for r in results]
        total_steps = [r.steps for r in results]
        wins = [1 if r.won else 0 for r in results]
        losses = [1 if r.lost else 0 for r in results]
        final_hps = [r.final_player_hp for r in results]

        avg_reward = sum(total_rewards) / max(len(total_rewards), 1)
        avg_steps = sum(total_steps) / max(len(total_steps), 1)
        win_rate = sum(wins) / max(len(wins), 1)
        loss_rate = sum(losses) / max(len(losses), 1)
        avg_final_hp = sum(final_hps) / max(len(final_hps), 1)

        best_reward = max(total_rewards) if total_rewards else 0.0
        worst_reward = min(total_rewards) if total_rewards else 0.0
        max_steps_reached = sum(1 for r in results if r.steps >= 500)

        print("\n==============================")
        print("GLOBAL RANDOM STATS")
        print("==============================")
        print(f"Episodes played      : {len(results)}")
        print(f"Wins                 : {sum(wins)}")
        print(f"Losses               : {sum(losses)}")
        print(f"Win rate             : {win_rate:.2%}")
        print(f"Loss rate            : {loss_rate:.2%}")
        print(f"Average reward       : {avg_reward:.3f}")
        print(f"Average steps        : {avg_steps:.2f}")
        print(f"Average final HP     : {avg_final_hp:.2f}")
        print(f"Best reward          : {best_reward:.3f}")
        print(f"Worst reward         : {worst_reward:.3f}")
        print(f"Max-step truncations : {max_steps_reached}")

        return

    # =====================================================
    # PLAY MODEL
    # =====================================================

    if mode == "play_model":
        print(f"\n=== RUN MODE: {mode} ===")

        cfg = get_default_config()
        results = []

        for ep in range(episodes):
            print(f"\n----- Episode {ep + 1} -----")

            controller = RunController(cfg=cfg, seed=ep)

            if checkpoint_path is not None:
                controller.load_combat_model(checkpoint_path)
                controller.set_eval_mode()

            result = controller.run_episode_with_model(
                deterministic=deterministic_model,
                max_steps=500,
                render=render,
            )

            results.append(result)

            print("\nEpisode resultv2.txt:")
            print(result.to_dict())
            save_episode(result, ep + 1, mode)

        # ==============================
        # GLOBAL STATS
        # ==============================
        total_rewards = [r.total_reward for r in results]
        total_steps = [r.steps for r in results]
        wins = [1 if r.won else 0 for r in results]
        losses = [1 if r.lost else 0 for r in results]
        final_hps = [r.final_player_hp for r in results]

        avg_reward = sum(total_rewards) / max(len(total_rewards), 1)
        avg_steps = sum(total_steps) / max(len(total_steps), 1)
        win_rate = sum(wins) / max(len(wins), 1)
        loss_rate = sum(losses) / max(len(losses), 1)
        avg_final_hp = sum(final_hps) / max(len(final_hps), 1)

        best_reward = max(total_rewards) if total_rewards else 0.0
        worst_reward = min(total_rewards) if total_rewards else 0.0
        max_steps_reached = sum(1 for r in results if r.steps >= 500)

        print("\n==============================")
        print("GLOBAL MODEL STATS")
        print("==============================")
        print(f"Episodes played      : {len(results)}")
        print(f"Wins                 : {sum(wins)}")
        print(f"Losses               : {sum(losses)}")
        print(f"Win rate             : {win_rate:.2%}")
        print(f"Loss rate            : {loss_rate:.2%}")
        print(f"Average reward       : {avg_reward:.3f}")
        print(f"Average steps        : {avg_steps:.2f}")
        print(f"Average final HP     : {avg_final_hp:.2f}")
        print(f"Best reward          : {best_reward:.3f}")
        print(f"Worst reward         : {worst_reward:.3f}")
        print(f"Max-step truncations : {max_steps_reached}")

        return

if __name__ == "__main__":
    main()
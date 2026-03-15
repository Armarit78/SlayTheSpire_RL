# SlayTheSpire_RL

Reinforcement Learning agent for **Slay the Spire** focused on learning
combat decision making using PPO.

This repository contains a **v1 research prototype** that trains a
combat agent capable of playing fights and completing **Act 1** runs
with increasing reliability. The project is structured to support future
expansion into **full run planning (macro decisions)** such as pathing,
card rewards, shops, and campfires.

------------------------------------------------------------------------

# Project Goals

Main objective:

Train an RL agent that can learn how to play **Slay the Spire Ironclad**
without hard‑coded strategies.

Current scope (v1):

-   Combat training with PPO
-   Structured combat state encoder
-   Reward shaping for tactical play
-   Action masking for legal game actions
-   Mock combat environment
-   Live gameplay adapter for testing

Future scope:

-   Inter‑combat decision agent (map, events, shops)
-   Improved combat representation
-   More accurate simulation environment
-   Better reward shaping and planning behavior

------------------------------------------------------------------------

# Current Capabilities (v1)

The agent can:

-   Play **combat automatically**
-   Use **valid action masking**
-   Learn card ordering strategies
-   Learn basic tactical play
-   Finish **Act 1 runs under time constraints**

The system supports:

-   PPO training
-   Curriculum training for enemies
-   Structured observation encoding
-   Combat reward shaping
-   Combat inference during live gameplay

------------------------------------------------------------------------

# Current Limitations

v1 limitations:

-   Combat model uses a **flat MLP architecture**
-   Environment is a **simplified simulation**
-   No long‑term planning
-   No map / event decision agent
-   Potion logic is limited
-   Deck building decisions are not learned

These limitations will be addressed in future versions.
---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Threes project — context and starting point for Codex
---

# Threes project — context and starting point for Codex

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

This is the original project proposal, preserved for provenance. See [early milestones](EARLY_WORK_LOG.md) for history and [current status](../CURRENT_STATUS.md) for accepted research decisions.

You are my development and experimentation partner for this project. Read this document as working context and an evolving roadmap. **This document does not authorize you to implement the entire roadmap.**

## 1. How we work together

We will move in small increments, examine what works, discuss it, and then decide what comes next. The plan can change based on our observations. I want the freedom to understand choices, try my ideas, and change direction.

Collaboration rules:

- Work only on the agreed milestone. Put ideas for later in notes, not automatically into code.
- Before the first development step, propose a tiny scope and wait until we agree on it.
- Once a milestone is authorized, complete it through a verified demonstration without asking for approval on each file or routine detail.
- At the end of a milestone, briefly explain what works, how to run it, what was verified, and what remains open. Suggest the next small step and let me respond.
- If I say “continue,” use the next increment we discussed; do not interpret that as permission to build the entire project.
- Favor readable code, limited changes, and observable results. Avoid homegrown frameworks, premature abstractions, and refactors without a concrete problem.
- Ask questions only when an answer would materially change a decision. Make and state reasonable assumptions for reversible details.
- Communicate in French, concisely and technically. I am an applied mathematician and I program; explain at that level without assuming I know every reinforcement learning practice.

## 2. Project idea

Build a small equivalent of **Threes!**, then train agents to play it on my personal computer.

Eventually the project should include:

1. A faithful, reproducible game engine that runs without a display.
2. A basic graphical interface for playing and watching games.
3. A reinforcement learning loop with a small neural network.
4. Experiments to compare methods and improve playing strength.
5. Visualizations of training, games, and selected internal network calculations.

The eventual ambition is a very strong agent, ideally better than available references under a comparable protocol. **That is a research ambition, not a promise or the success criterion for the first prototype.**

My personal best score is about 204,000, with a 3,072 tile. Distinguish a game's score from the value of its largest tile. The reference tile under discussion is **6,144**, and possibly later **12,288**.

I want to develop our own understanding and explore variants. We can use existing libraries and research, but this project is not simply about downloading a trained agent and running it.

## 3. Hardware and environment

I work on Windows. I estimate I have a good graphics card with about 11 GB of VRAM and 32 or 64 GB of RAM, but these details remain to be checked. Do not assume the GPU vendor or CUDA compatibility.

At the start:

- Read repository instructions, including `AGENTS.md` if it exists, and inspect files before changing anything.
- Check where you actually work: native Windows, WSL2, or a remote environment. Do not present the hardware of your execution environment as my PC if they are separate machines.
- Find Python and the existing environment. Use what is already suitable; do not reinstall everything by default.
- Detailed GPU setup can wait for the first milestone that needs it. Hello World and the engine do not depend on it.
- Do not start long training or hyperparameter searches until we set their scope and compute budget.

Training should support interruption and resumption when we reach longer experiments. Do not give a precise convergence time before measuring throughput and observing learning.

## 4. Possible technical choices, to confirm when relevant

| Need | Tentative initial choice | When to introduce it |
| --- | --- | --- |
| Main program | Python, with a clear entry point | From Hello World |
| Configuration | Readable JSON | Very small at first, extended as needed |
| Numerical engine | Python; NumPy if useful | During engine development |
| Simple display | Pygame | Once the minimal engine works |
| RL environment interface | Gymnasium adapter | When connecting the first learned agent |
| Networks and optimization | PyTorch | For the first neural prototype |
| Initial learning | PPO, possibly Stable-Baselines3 and `sb3-contrib` for invalid-move masking | Decide during the RL prototype |
| Training tracking | CSV and simple charts, then TensorBoard | As experiments become useful to compare |
| Analysis interface | Streamlit + Plotly | After the first usable training loop |
| Interpretability | PyTorch measurements and possibly Captum | After obtaining a first interesting agent |
| Engine acceleration | Numba, then another solution only if justified | After profiling a real bottleneck |

These are preferences, not requirements. DQN, afterstate value learning, a recurrent network, or another approach can be proposed with a concrete justification. Avoid multiplying algorithms in the first version.

Do not install all dependencies in this table now. Add only those needed for the current milestone. Check compatible versions when actually using them.

Organization principles:

- The engine owns the rules and state; it depends on neither the graphical interface nor PyTorch.
- The graphical interface renders the engine and sends it actions. It does not keep its own copy of the rules.
- The same engine serves human games, simulations, and evaluations.
- Experiments and results remain separate from application code. Later, each experiment will have its configuration, metrics, and saved files in a dated folder under `results/`.
- Use explicit names. Classes can help when they own coherent state, but Hello World needs no class hierarchy.
- Do not create abstract interfaces, model registries, plugins, or a distributed architecture yet.

## 5. Engine fidelity: points to address explicitly

Threes is not simply 2048 with different numbers. During the engine milestone, verify rules against reliable sources and define the reproduced variant clearly.

Cover these points progressively:

- A 4 × 4 board with empty cells.
- Tiles moving at most one cell per action, according to the actual game mechanics.
- A `1` merging with a `2` to make `3`; from `3` upward, two equal values merging.
- Cell processing order, blocked tiles, and merges during one move.
- An impossible move does not advance the game or draw another tile.
- A tile spawning after a valid move in positions allowed by the movement.
- Ordinary tile distribution, the deck or bag, and bonus generation.
- Information actually shown in the next-tile preview for the chosen version.
- Board initialization, scoring, and game termination, including behavior at the final tile.

For ordinary tiles valued `3 × 2^k`, their score is `3^(k+1)`; tiles `1` and `2` score zero. Do not confuse this score with the sum of tile values on the board. Verify its application in the chosen variant too.

It is acceptable to start with a fixed board and deterministic movement, then add random draws. Document any simplification as temporary. Do not describe a simplified variant as faithful, or its results as directly comparable with Threes! results.

Provide a random seed and reproducible games when random draws are introduced. Tests should target real risks: tricky moves and merges, insertion, invalid moves, scoring, and termination. Do not build a test suite merely to check that Hello World prints its title.

## 6. Indicative roadmap

The order below gives us a direction. We may split, merge, postpone, or replace milestones after discussion. **Listing a milestone here does not authorize its immediate implementation.**

### Milestone 0 — A genuinely tiny Hello World

Goal: verify that we can work and run a program in the right environment.

Proposed scope:

- A `main.py` that starts, prints the project name, and shows a fixed 4 × 4 grid of numbers in the terminal.
- A very short JSON configuration if it already helps; no catalog of future options.
- A short `README.md` with a run command suited to my environment.
- The standard library should suffice.

**Done when:** the command works and we can see the result. No complete engine, GUI, network, training, or dashboard is requested here.

### Milestone 1 — The engine, in small slices

Goal: produce correct transitions on easy-to-inspect boards.

Start with an explicit state, legal moves, and movement/merges for selected cases. Then add tile spawning, initialization, scoring, and termination until a complete game can be played in the terminal.

**Done when:** important cases are checked, a game is reproducible, and the engine runs without a GUI. If this exceeds a small increment, split the milestone before implementing it.

### Milestone 2 — See and control the game

Goal: visually check that the engine does what we expect.

Add a simple window: grid, readable numbers, basic colors, arrow-key controls, new game, score, and next-tile preview. We should be able to play and observe problems.

**Done when:** a game is playable with the same engine. No artwork, elaborate animations, complex menu, or online features.

### Milestone 3 — Run games without a display

Goal: prepare a simple experimental environment and measure throughput.

Add a player that chooses among legal moves and run a small configurable number of headless games. Record at least final score, largest tile, move count, and seed. A simple heuristic can then provide a second reference.

**Done when:** we have a reproducible result and speed measurement. Headless mode must reuse the existing engine; there is no second implementation to maintain.

### Milestone 4 — Connect the first network

Goal: verify inputs, outputs, and integration before seeking strong play.

Add a structured observation, a very small PyTorch network, the outputs required by the chosen algorithm, and handling of invalid moves. Check saving and reloading weights. The first trial can use the CPU.

**Done when:** the network controls a game, dimensions are correct, and reloading preserves deterministic behavior. A random network playing poorly is normal at this stage.

### Milestone 5 — First learning proof of concept

Goal: connect experience collection, reward, and weight updates in a short, observable loop.

Choose an initial algorithm and reward together, aligned with the objective. Start with a very short integration check, then a bounded experiment. Display essential metrics and retain the configuration.

**Done when:** the loop runs, numerical values stay healthy, weights actually update, and we have an initial evaluation summary. Distinguish working integration from learning that actually improves play. If performance stalls, seek the cause before immediately enlarging the model.

### Milestone 6 — Reliable experiments and automatic tracking

Goal: compare changes without being misled by chance or a spectacular individual game.

Gradually introduce separate evaluations, resumable saves, multiple training seeds, result folders, and automatic charts. Increase the number of evaluation games when differences become small enough to require it.

**Done when:** two configurations can be compared under an explicit protocol with traceable results.

### Milestone 7 — More ambitious architectures and strategies

Goal: improve play from a reference we understand.

Possible directions, one at a time:

- A more suitable standard network, small CNN, or Transformer.
- History memory, such as a recurrent network or sequence of turns.
- Board symmetries with corresponding action transformations.
- Training with more exposure to endgames and large tiles.
- Learning the value of the position after movement, before the random event.
- Move search combined with a learned value.
- Faster simulation if measurements justify it.

**Done for each experiment when:** we know what changed, what it cost, and whether the gain is confirmed or still uncertain.

### Milestone 8 — Model inspection and analysis interface

Goal: explore games, experiments, and selected internal calculations.

Start with whichever feature is most useful at that point, for example replaying a game with action probabilities. Add other views gradually. Basic tracking need not wait for this milestone.

## 7. Connecting engine and learning

The engine and training can run in one program using function calls and in-memory arrays.

The loop repeats: observe a board, choose an action, apply the move, collect a reward and the next observation, then update the network according to the chosen algorithm.

Later, one network can control several independent games and process their observations in batches. For example, 128 environments for 64 turns yield 8,192 transitions. **These numbers illustrate the principle; they are not required parameters.** Ending a game resets it and must be handled correctly in learning targets.

Collection and learning can alternate. A distributed architecture or asynchronous processes are unnecessary at first. Measure the benefit of a GPU and the number of environments for small networks.

Clearly separate the simulator's internal state from the observation given to the agent. The agent must not accidentally receive the future draw, random generator state, or hidden bag information. Explicitly label any experiment that deliberately gives the agent privileged information.

A Gymnasium-style interface can expose `reset()` and `step(action)`. When introduced, respect the distinction between a real game ending and a time-limit cutoff. Do not let debugging information leak into network inputs.

## 8. Transformers and visualizations: later intentions

I do want to explore Transformers, without assuming they will win before comparison.

One reasonable proposal is one token per cell, row/column positional encoding, the next-tile preview, and policy and value outputs. Two to four layers of width 64 or 128 give an initial scale to discuss. A Transformer receiving only the current board does not automatically remember earlier turns.

Architectures should evolve through small localized changes, without building a generic framework for every imaginable architecture now.

Eventually desired visualizations:

- Performance curves and experiment comparisons.
- Score distributions and large-tile reach rates.
- Game replay with action probabilities and estimated position value.
- Several saved models compared on the same board.
- Selected layer activations and weight or gradient distributions.
- Attention maps by layer and head for a Transformer.
- Estimated cell influence using attribution methods.
- Effects of controlled perturbations or disabling components.
- Possibly a 2D projection of internal representations of selected positions.

An attention map or attribution is not, by itself, causal proof of a model's reasoning. Present results as measurements with limitations.

Capture targeted observations when useful. Do not log every activation of every move across billions of transitions. The dashboard should read logs and saved models; costly analyses can run separately from training.

## 9. Measuring progress and using references

For serious experiments, distinguish:

- Training reward from official evaluation score.
- Mean score, median score, distribution, and individual record.
- Probability of reaching at least a 3,072, 6,144, or 12,288 tile, with the exact convention stated.
- A network alone from a network accompanied by move search.
- Training budget, simulator speed, and decision time during play.
- Variation among games and among independent training runs.

Use reserved evaluation seeds and keep a distinct final evaluation if we tune many models on the same test set. The same random-seed protocol does not guarantee that two policies encounter the same boards after their decisions; avoid overinterpreting a few seeds.

Artificial states or endgame starts may be useful for training, but final evaluation must also cover complete games begun under the chosen rules.

References identified during initial discussion:

| Reference | Reported result | Caution |
| --- | --- | --- |
| Yeh and collaborators, *Multi-Stage Temporal Difference Learning for 2048-like Games*, 2016 preprint, 2017 publication | Across 10,000 games: 6,144-tile rate of 0.45% for TD with depth-3 expectimax, and 7.83% for the three-stage variant at the same depth | `n-tuple` pattern networks, not conventional deep networks; verify study rules and variants |
| Josiah Kiok, *Beating Threes! with Reinforcement Learning*, March 2026, and associated repository | The author later reports 21.5% of games reaching 6,144 across 10,000 trials, without search during play | Self-reported result not reproduced by us; recurrent PPO/PufferLib network, C engine, substantial training; check especially what information the agent receives |

These figures are comparison points, not a guaranteed exhaustive state of the art. Our hardware, budget, and engine are not automatically comparable. Verify sources before claiming we exceed another method.

Starting sources, to consult when they answer a current milestone question:

- [Yeh and collaborators' paper](https://arxiv.org/abs/1606.07374) and [PDF](https://arxiv.org/pdf/1606.07374).
- [Josiah Kiok's article](https://medium.com/@josiah-kiok/beating-threes-with-reinforcement-learning-ae074dd28a68).
- [Kiok project repository and demo](https://github.com/pseudonam-gc/threes-web).
- [nneonneo Threes AI implementation](https://github.com/nneonneo/threes-ai).
- [Gymnasium interface](https://gymnasium.farama.org/api/env/).
- [Maskable PPO in SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
- [Captum](https://captum.ai/docs/introduction).

Do not turn the first milestone into a literature review. Reuse useful knowledge, verify doubtful points when necessary, and respect licenses if you reuse code.

## 10. What I expect from your first response

**Do not start by coding the engine, installing PyTorch, or creating the entire module skeleton.**

1. Read the existing folder and its instructions without making changes.
2. Summarize in a few sentences your understanding of the goal and collaboration style.
3. Propose a concrete version of **milestone 0 only**: the few relevant files, visible result, and intended run command.
4. Mention only uncertainties that genuinely block this tiny start. Ask at most two or three useful questions if necessary.
5. Wait for my reply before implementing this first increment.

After our agreement, deliver that small, complete, verified result. We will decide what follows together.

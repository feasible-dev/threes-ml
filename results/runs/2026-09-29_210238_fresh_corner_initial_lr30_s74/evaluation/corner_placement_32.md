# Corner placement on fixed development games

> Recorded: 2026-09-30T06:56:40-04:00. Training steps refer to the checkpoint evaluated, not the record time.

Each complete checkpoint was evaluated on 32 natural-start games, seeds 880000 through 880031, with the native reproduction simulator and legal greedy actions. The recurrent state was reset per game. At each active decision state, the largest tile counted as in a corner if any tied largest tile occupied a corner. The denominators below count eligible decision states, which are correlated within games.

| Checkpoint steps | Mean score | Largest tile >=12 in corner | Largest tile >=192 in corner |
| ---: | ---: | ---: | ---: |
| 12,066,816 | 1,348 | 513/2,891 (17.7%) | 1/263 (0.4%) |
| 20,094,976 | 1,751.9 | 565/3,522 (16.0%) | 0/459 (0.0%) |
| 98,369,536 | 4,180.6 | 1,067/5,632 (18.9%) | 410/2,206 (18.6%) |
| 100,007,936 | 4,256.9 | 936/5,523 (16.9%) | 210/2,291 (9.2%) |

At the final checkpoint, 24 of 32 games reached tile 192. The corner rate for >=192 increased from the early checkpoints but varied substantially between the last two checkpoints. This small, reused development sample is descriptive; it does not establish that the corner reward caused the change. The matched short pilot is recorded separately in the research status.

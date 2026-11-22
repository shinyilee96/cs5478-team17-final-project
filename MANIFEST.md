# What reproduces what

Every number, table and figure in the body of the report (Sections 1–7), the command that reproduces it, what that
command reads, and where it writes. Three levels:

- `python reproduce.py tables` and `python reproduce.py figures` recompute every body table, number and figure from the
  per-trial files in `results/` (seconds, no simulation).
- `python reproduce.py PART --quick` does the same for one part, then reruns one recorded trial in the simulator and
  compares it with its recorded row (minutes).
- `python reproduce.py PART --full` reruns the whole experiment in the simulator into `runs/PART/`, then recomputes the
  part's numbers from the new runs (hours; see README for runtimes).

Expected values are the report's. Intervals are 95%, from a cluster bootstrap with 10,000 resamples (seed 0).

## Install check

`python reproduce.py check`: package versions, the prepared objects and robot assets against their checksums
(`setup/prepared_data.sha256`, `setup/robot_assets.sha256`), the plan-choice numbers from `results/`, and two robot pickups
against their recorded rows.

## Plan choice (Table 2, rows 1–3; Section 5.3 text)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Replay ranking R − random plan | +14.5 pts (+9.4 to +19.8) | `reproduce.py plan-choice` | `results/plan_choice_pickups.csv`, `results/plan_choice_rankings.csv` | `results/summary/plan_choice.txt` |
| Learned predictor P − random plan | +19.4 pts (+14.0 to +24.7) | same | same | same |
| Learned predictor E − random plan | +17.6 pts (+13.3 to +21.9) | same | same | same |
| P − R, E − R | +4.9 (−0.8 to +10.3), +3.0 (−2.8 to +8.9) | same | same | same |
| Rerun (full): all 48 ranked pairs (47 counted; one has a single candidate plan), every candidate plan, 20 robot pickups each (seeds 9100–9119) | as above | `reproduce.py plan-choice --full` | `inputs/plan_choice_library/`, `data/` | `runs/plan-choice/` |

## Perception (Table 2, rows 4–7)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Known objects: estimated − exact poses | +1.7 pts (−4.0 to +7.9) | `reproduce.py perception` | `results/perception_pickups.csv` | `results/summary/perception.txt` |
| Unseen objects: primitive-shape plan library − random primitive plan | +12.9 pts (+6.9 to +19.0) | same | `results/unseen_pickups.csv`, `inputs/unseen_objects/` | same |
| Unseen objects: library, both objects held | 42.0% (26.8 to 58.2) | same | same | same |
| Unseen objects: true shape (privileged) − library | +18.2 pts (+0.4 to +35.9) | same | same | same |
| Rerun (full): 21 plans × 20 pickups × 2 pose sources; 16 unseen pairs, 335 plans × 20 pickups | as above | `reproduce.py perception --full` | `inputs/plan_library/`, `inputs/primitive_shape_library/`, `inputs/unseen_true_shape_library/` | `runs/perception/` |

## Recovery (Table 2, rows 8–9; Section 5.3 text)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Round table: objects saved per failed two-object trip | 0.25 (0.06 to 0.50) | `reproduce.py recovery` | `results/trips.csv` (column `saved_by_recovery`), `results/errands.csv` | `results/summary/recovery.txt` |
| Long table: objects saved per failed two-object trip | 0.70 (0.33 to 1.10) | same | same | same |
| Rerun (full): DexCarry's 144 errands at 4 m (shared with `errands --full`) | as above | `reproduce.py recovery --full` | as errands | `runs/errands/` |

## Errands: DexCarry against the single-object baseline (Table 1, Figure 2a, Section 5.2 text)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Table 1: delivery-rate change and delivery-share difference, both tables, 1–16 m and 4 m pooled | as in Table 1 (e.g. 4 m pooled: round +25.4% (+12.7 to +37.8), −3.9 pts; long +16.2% (+5.2 to +26.9), −8.7 pts) | `reproduce.py errands` | `results/errands.csv` | `results/summary/table1.txt`, `table1.csv` |
| DexCarry delivers about 84% on both tables; baseline 93% (long) vs 88% (round) at 4 m | 84%, 84%; 93%, 88% | same | same | same |
| Trip statistics at 4 m (pooled): median two-object trip 75 s vs single-object 56–58 s; failure code in 32% vs 12% (round), 46% vs 7% (long) | as stated | same | `results/trips.csv` | `results/summary/trips.txt` |
| Long-table break-even | 0.66 m (−0.49 to 2.73), reported as "around 1 m" | same | `results/trips.csv` (standard setting, 1–16 m) | `results/summary/trips.txt` |
| Figure 2a | the six standard-setting points per table | `reproduce.py figures` | `results/errands.csv` | `results/figures/fig_distance.pdf` |
| Rerun (full): 384 round-table + 384 long-table errands (2 methods; 4 m in three settings, 1, 2, 8, 12, 16 m standard) | as above | `reproduce.py errands --full` | `inputs/`, `weights/success_estimator/`, `data/` | `runs/errands/` (per-trial files in `runs/errands/results/`) |

## Failure-aware against greedy planning (Figure 2b, Section 5.4 text)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Random layouts, round table, 4 m, 72 errands each: failure-aware − greedy | +2.0% (−4.0 to +8.7) | `reproduce.py planners` | `results/errands.csv` | `results/summary/planners.txt` |
| Partner scarcity, 72 errands each: failure-aware − greedy | +6.2% (−1.6 to +14.1) | same | same | same |
| Partner scarcity: greedy's first trip uses the tempting pair in 72 of 72; box delivered 57 of 72 (greedy) vs 72 of 72 (failure-aware) | 72; 57 vs 72 | same | `results/errands.csv` (columns `first_trip_tempting_pair`, `scarce_object_delivered`) | same |
| Figure 2b | the two bars above | `reproduce.py figures` | `results/errands.csv` | `results/figures/fig_lookahead.pdf` |
| Rerun (full): 72 greedy errands (round table, 4 m) + 144 partner-scarcity errands, with DexCarry's 72 round-table errands at 4 m | as above | `reproduce.py planners --full` | as errands | `runs/errands/` |

## Extension: RL grip (Table 3)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Stationary, 1.5 N·m: n 539; RL 63.1, reflex 53.2, fixed 32.8; RL − reflex +8.9 (+2.7 to +15.5); RL − fixed +30.2 (+18.2 to +42.0) | as stated | `reproduce.py ext-grip` | `results/grip_stationary.csv` | `results/summary/table3.txt` |
| Stationary, 1.0 N·m: n 469; 64.2, 61.0, 37.7; +1.9 (−4.6 to +8.4); +24.6 (+10.5 to +38.1) | as stated | same | same | same |
| Carry, 1.5 N·m: n 180; 54.8, 40.8, 37.7; +14.0 (+4.5 to +24.4); +17.0 (−1.0 to +34.1) | as stated | same | `results/grip_carry.csv` | same |
| Carry, 1.0 N·m: n 157; 55.0, 44.3, 43.3; +10.8 (+3.2 to +18.7); +11.7 (−9.5 to +30.7) | as stated | same | same | same |
| Rerun (full): 660 stationary phases (seeds 32600+, three settings) + 220 carry phases (seeds 32700+) | as above | `reproduce.py ext-grip --full` | `inputs/plan_library/`, `weights/grip_policy/` | `runs/ext-grip/` |
| Training (optional): PPO, 6M steps | a new policy (not the frozen one) | `reproduce.py ext-grip --train` | `inputs/grip_training/` | `runs/ext-grip/training/` |

## Extension: LLM requests compiled into the planner (Section 6.2 results paragraph)

| Number | Expected | Command | Reads | Writes |
|---|---|---|---|---|
| Explicit requests compiled exactly | 40 of 40 (Wilson 91 to 100%) | `reproduce.py ext-llm` | `results/llm_requests.csv` | `results/summary/llm.txt` |
| Instruction violations: planner vs direct LLM planning | 0% vs 6.8% | same | same | same |
| Weighted delivery rate in the planner's model: planner − direct LLM | +6.1% (+1.2 to +11.1) | same | same | same |
| Priority object lost: planner vs direct LLM | 14.9% vs 5.8% | same | same | same |
| Simulated errands, 24 per method: weighted delivery rate, planner − direct LLM | +5.0% (−11.1 to +19.2) | same | `results/llm_errands.csv` | same |
| Rerun (full): the 80 request–scene units scored from cached LLM replies (no key, no network), then 48 simulated errands | as above | `reproduce.py ext-llm --full` | `inputs/language/` | `runs/ext-llm/` |

## Not reproduced (illustrations, not results)

The teaser (Figure 1) and the LLM demo panel (Figure 3) are renders. The videos in `videos/` were rendered in Blender
from the simulator's pose logs; the render pipeline is not part of this release.

Two statements in the report body have no command here: the "roughly 25 s of extra handling" of a two-object trip
(Section 7; the trip statistics above give a 17 to 19 s longer median trip at 4 m) and the "nine plans ... from 5 to 20
of 20" of the first robot trials (Section 4.1), which came from early trials not included in this release.

## Columns of the per-trial files

| File | Columns that need explaining |
|---|---|
| `errands.csv` | `table` (round or long), `layouts` (`random` or `partner_scarcity`), `method` (`dexcarry` is the failure-aware planner, `single_object` the baseline, `greedy` the greedy planner); `objects`, `delivered` and `upright` count objects; `seconds` and `metres` are the errand's time and drive; `delivered_objects` lists the indices of the delivered objects; `first_trip_tempting_pair` and `scarce_object_delivered` are filled for partner-scarcity errands only |
| `trips.csv` | One row per trip. `kind` is `pair` or `single`; `first` and `second` are object indices (the first is wrapped against the palm); `began` and `ended` are errand times in seconds; `failure` is the failure code (empty when the trip succeeded); `refused` gives the reason when a trip was refused before it started; `recovery_steps` lists the recovery actions; `saved_by_recovery` counts objects recovery delivered |
| `plan_choice_pickups.csv`, `perception_pickups.csv`, `unseen_pickups.csv` | One row per robot pickup: `both_held` is 1 when both objects were held, `first_failure` the first failure code; `poses` is `exact` or `estimated`; `plans` is `primitive_library` or `true_shape` |
| `plan_choice_rankings.csv` | Each pair's candidate plans and the top three of each rule (R, P, E) |
| `grip_stationary.csv`, `grip_carry.csv` | One row per phase, method and motor cap (`cap_Nm`). `outcome` is `held`, `dropped` or `crushed`; `peak_N` is the peak grip force; `limit_N` the crush limit; `firm_hold_N` the peak force of the gentlest fixed grip that holds; `margin` is `limit_N / firm_hold_N`; `mass_kg` is the hidden mass |
| `llm_requests.csv`, `llm_errands.csv` | One row per request, layout, seed and method (`planner` or `direct_llm`). `rate` is the weighted delivery rate, `share` the share of requested objects delivered, `violations` the instruction violations |

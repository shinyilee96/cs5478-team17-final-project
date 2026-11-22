# DexCarry: Fetching Two Objects at a Time with One Dexterous Hand on a Vision-Guided Mobile Robot

Done By: Team 17
- Lee Shin Yi (A0167415A) 
- Evan Ang Jia-Jun (A0200591R)
- Lee Gui Yong (A0167229X)  

This repository reproduces the experiments and figures in the body of the report, and the 2 extensions.
`MANIFEST.md` lists each number with the command that reproduces it.


**Demo Videos shown in final presentation**

The videos can be found in `videos\`.

| File | What it shows |
|---|---|
| `close_up_grasp.mp4` | Close up view of the side grasp and the tripod grasp |
| `plan_choice_demo.mp4` | Two plans for the same pair to highlight why ranking plans matters |
| `replay.mp4` | Replays of plans in the floating-hand simulator |
| `unseen.mp4` | Unseen objects: depth image, fitted primitive shapes and the pickup |
| `recovery.mp4` | Recovery after failed two-object trips |
| `race.mp4` | DexCarry against the single-object baseline on the same errand |
| `trap.mp4` | Partner scarcity built to trap Greedy: the failure-aware planner against the greedy planner |
| `ext_rl_grip.mp4` | Extension: the RL grip against a fixed grip on a fragile object |
| `ext_llm_demo.mp4` | Extension: a spoken request compiled by an LLM and fetched |

## Requirements

- conda. Tested on Windows 11 only (Python 3.10, 16 CPU threads). CPU only: no GPU is needed, training included.
- Everything the code needs is in this repository (robot model, prepared objects, frozen models, per-trial results);
  the only downloads are the Python packages.
- Simulator commands use at most 14 processes (`--jobs N` for fewer).

## Setup

The third line replaces conda's NumPy by NumPy's own build of the same version, which the recorded runs used.

```
conda env create -f environment.yml
conda activate team17-dexcarry-release
pip install --force-reinstall --no-deps numpy==1.26.4
python reproduce.py check
```

`check` prints the package versions, checks the robot model and prepared objects against their checksums, recomputes
the plan-choice numbers and reruns two recorded robot pickups.

## Reproducing the report

`python reproduce.py tables` recomputes every body table and number from the per-trial files in `results/` and
prints each beside the report's value. `python reproduce.py figures` draws Figure 2 into `results/figures/`.

Each part also runs in the simulator as `python reproduce.py PART [--quick | --full]`. `--quick` (the default) recomputes the
part's numbers and reruns one recorded trial against its recorded row. `--full` reruns the whole experiment into `runs/PART/`
(`errands`, `recovery` and `planners` share `runs/errands/`) and resumes if interrupted.

| Report item | PART | Expected | Quick | Full |
|---|---|---|---|---|
| Table 1, Figure 2a, trip statistics | `errands` | 4 m pooled: round table +25.4% (+12.7 to +37.8), long table +16.2% (+5.2 to +26.9); long-table break-even 0.66 m | 4 min | 2.5–3 h |
| Table 2, plan choice | `plan-choice` | R +14.5, P +19.4, E +17.6 points over a random plan | 3 min | 3.5 h* |
| Table 2, perception | `perception` | known +1.7; unseen +12.9, 42.0%, +18.2 | 5 min | 1.5 h* |
| Table 2, recovery | `recovery` | 0.25 (round), 0.70 (long) objects saved per failed trip | 2 min | 30 min |
| Figure 2b, failure-aware vs greedy | `planners` | +2.0% (random layouts), +6.2% (partner scarcity) | 2.5 min | 1 h |
| Table 3, RL grip | `ext-grip` | RL 63.1 / 64.2% stationary, 54.8 / 55.0% carry (1.5 / 1.0 N·m) | 3.5 min | 5.5 h |
| Section 6.2, LLM requests | `ext-llm` | 40 of 40 compiled; violations 0 vs 6.8%; +6.1%; +5.0% simulated | 2.5 min | 10 min |

Full runtimes are with 14 processes on the machine above (* estimated from the quick rerun's pace).
`python reproduce.py ext-grip --train` retrains the PPO grip (6M steps, about 2.5 h on CPU); the results use the frozen policy
in `weights/`. LLM replies come from the cache in `inputs/language/cache/`: no API key or network is needed.

As in the report, object identity and the collision poses of other objects come from the simulator, and known objects are
planned on their true geometry; unseen objects use primitive shapes fitted to the depth image.

## Exactness

For the machine in Requirements (set up as described) every quick rerun matches its recorded row exactly. Another machine, OS or BLAS build (conda-forge's NumPy included) may produce slghtly different reruns. Full reruns should be able to be reconciled with the
report within its 95% intervals. Do not set `OMP_NUM_THREADS` or other BLAS thread counts: they change the errand outcomes.

## Running the simulation

`run_simulation.py` replays one recorded trial of each part in PyBullet's viewer and prints its outcome beside the recorded one.

*Note that this will differ from the recorded demo videos in `videos/` as it was rendered with blender with added scenary for visual purposes only*

| Command | What it shows |
|---|---|
| `python run_simulation.py grasp` | One two-object pickup: the side wrap, then the tripod grasp |
| `python run_simulation.py ranking` | The replay ranking's first and last choice of plan for one pair |
| `python run_simulation.py perception` | One plan on exact poses, then on poses estimated from the depth image |
| `python run_simulation.py unseen` | An unseen pair: the primitive-shape plan library's first choice |
| `python run_simulation.py recovery` | An errand where an object slips out on the way and recovery takes over |
| `python run_simulation.py errand --planner failure-aware --table round --layout 0 --seed 26300` | One errand |
| `python run_simulation.py rl-grip` | The RL grip, the slip reflex and the fixed grip on one object with a crush limit |
| `python run_simulation.py llm` | A typed request, compiled from the cached LLM reply, then fetched |

`errand` takes `--planner single`, `greedy` or `failure-aware`, `--table round`, `long` or `scarcity` (no greedy on the long
table), layouts 0–11 and seeds 26300–26301 (parnter scarcity: 26500–26505). `--speed 4` plays four times faster, `--headless` skips the viewer, and the viewer waits for Enter at the end of each scene.
`--distance` sets the drive (1, 2, 4, 8, 12 or 16 m; default 4). Distances other than 4 m exist for the single and failure-aware
planners on the round and long tables; beyond 4 m the camera follows the robot. On the round and long tables, layouts 0–3 have 4 objects, 4–7 have 5 and 8–11 have 6.

## File Paths

| Path | What it holds |
|---|---|
| `reproduce.py`, `run_simulation.py` | Entry points (above) |
| `dexcarry/` | The package (methods below; each experiment's driver is in `dexcarry/experiments/`) |
| `assets/` | Robot description files, unchanged from their sources (see Licences) |
| `data/` | Prepared objects (see the last section); the merged robot model is built into `data/urdf/` on first use |
| `configs/` | Robot, hand, simulator and layout settings |
| `inputs/` | Frozen inputs: plan libraries, layouts, trip times, unseen-object plan choices, LLM scenes and cached replies |
| `weights/` | Plan predictors P and E, the RL grip policy |
| `results/` | Per-trial results; `summary/` and `figures/` are recomputed from them |
| `setup/` | Data preparation and checksums |
| `videos/` | Demonstration videos (rendered in Blender; the render pipeline is not included) |

| Method | Code |
|---|---|
| Two-object pickup and trips | `dexcarry/pipeline/two_object.py`, `dexcarry/pipeline/trip_stages.py` |
| Replay ranking R; learned predictors P and E | `dexcarry/models/plan_predictor.py`, `dexcarry/experiments/predictors.py` |
| Perception (depth image, pose estimates, primitive fits) | `dexcarry/perception/` |
| Recovery | `recover_errand` in `dexcarry/pipeline/errand.py` |
| Failure-aware and greedy planners | `dexcarry/plan/errand_planner.py` |
| RL grip | `dexcarry/rl/grip.py`, `dexcarry/experiments/rl_grip.py` |
| LLM requests | `dexcarry/plan/llm_intent.py`, `dexcarry/experiments/language.py` |
| Statistics (cluster bootstrap) | `dexcarry/eval/stats.py` |


## Licences

Third-party files keep their own licences:
- `assets/pick_nav_reach/`: the Fetch robot and the YCB sugar box from the course repository
  (`https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project`).
- `assets/mujoco_menagerie/leap_hand/`: the LEAP hand from MuJoCo Menagerie (`https://github.com/google-deepmind/mujoco_menagerie`,
  MIT licence, see its `LICENSE`).
- `data/`: objects prepared from DexGraspNet as processed by BODex (CC BY-NC 4.0), Google Scanned Objects (CC BY 4.0) and the
  YCB sugar box, plus boxes and cylinders generated here.

## [Additional] Rebuilding `data/` from the original datasets from scratch

*Note: This is not required to run our repo, as we committed the processed datasets in `data/`. This is placed here in case if there is a need to rebuild the data from scratch*

`setup/prepare_data.py` builds `data/` from 12 DexGraspNet objects as processed by BODex (Hugging Face dataset
`JiayiChenPKU/BODex`), 8 Google Scanned Objects
(`kevinzakka/mujoco_scanned_objects`) and the YCB sugar box in `assets/`. It needs git and curl, overwrites
`data/` and checks every file it writes against `setup/prepared_data.sha256`:

```
curl -L --create-dirs -o datasets/DGN_2k_processed.tar.gz https://huggingface.co/datasets/JiayiChenPKU/BODex/resolve/7f25e59b111ef3405aea242199ec3e07f2965c13/object_assets/DGN_2k_processed.tar.gz
tar -xzf datasets/DGN_2k_processed.tar.gz -C datasets DGN_2k/processed_data/core_bottle_70e58e9f89fd9802a9891f6668163611 DGN_2k/processed_data/core_bottle_f49d6c4b75f695c44d34bdc365023cf4 DGN_2k/processed_data/core_can_56dfb6a30f498643bbf0c65ae96423ae DGN_2k/processed_data/ddg_gd_box_poisson_008 DGN_2k/processed_data/ddg_gd_tetra_pak_poisson_003 DGN_2k/processed_data/ddg_gd_toaster_poisson_006 DGN_2k/processed_data/ddg_kit_InstantTomatoSoup DGN_2k/processed_data/mujoco_Epson_LabelWorks_LC5WBN9_Tape_reel_labels_071_x_295_Roll_Black_on_White DGN_2k/processed_data/mujoco_Sonicare_2_Series_Toothbrush_Plaque_Control DGN_2k/processed_data/sem_TissueBox_ddf889fc432a0f869bc18b58d4f2bc4 DGN_2k/processed_data/sem_Vase_32dc55c3e945384dbc5e533ab711fd24 DGN_2k/processed_data/sem_Vase_3a275e00d69810c62600e861c93ad9cc
git clone --filter=blob:none --sparse https://github.com/kevinzakka/mujoco_scanned_objects datasets/mujoco_scanned_objects
git -C datasets/mujoco_scanned_objects sparse-checkout set models/3D_Dollhouse_Lamp models/Perricoen_MD_No_Concealer_Concealer models/Office_Depot_Canon_CLI_221BK_Ink_Cartridge_Black_2946B001 models/Tune_Belt_Sport_Armband_For_Samsung_Galaxy_S3 models/YumYum_D3_Liquid models/BUILD_A_ZOO models/MK7 models/Germanium_GE132
python -I setup/prepare_data.py --dexgraspnet datasets/DGN_2k/processed_data --scanned-objects datasets/mujoco_scanned_objects/models
```

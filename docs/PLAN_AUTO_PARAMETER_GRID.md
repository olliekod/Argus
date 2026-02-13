# Plan: Automatic Parameter Grid Generation for the Research Loop

**Audience:** This document is self-contained and is intended to be sent to an AI (e.g. ChatGPT) that does **not** have access to the Argus codebase. It provides enough context and explicit details to implement “automatic” parameter grids (e.g. generate a list of values from a range and step instead of requiring the user to manually list every value).

---

## 1. What Argus Is and What the Research Loop Does

- **Argus** is a trading/research system. It has a **Strategy Research Loop** that:
  1. Builds **replay packs** (bars, outcomes, regimes, option snapshots) from a database.
  2. Runs **experiments**: each configured strategy is backtested on those packs (replay).
  3. **Evaluates** results (rankings, kill rules, DSR, etc.) and optionally runs **allocation**.

- The loop is driven by a **YAML config file**, e.g. `config/research_loop.yaml`. The user runs:
  - `python scripts/strategy_research_loop.py --config config/research_loop.yaml --once`
  (or without `--once` for a daemon that repeats the cycle.)

- **Strategies** are listed in the config under a top-level key `strategies`. Each entry has:
  - `strategy_class`: string, e.g. `"VRPCreditSpreadStrategy"`.
  - `params`: a YAML object (dict) of **base** parameters passed to the strategy constructor.
  - `sweep`: either `null` (single run with `params` only) or a **path to a YAML file** (e.g. `"config/vrp_sweep.yaml"`) that defines a **parameter grid**. When `sweep` is set, the loop runs **every combination** of the grid and merges each combination with `params`.

- So today, “test all variations” = the user creates a **sweep YAML file** where they **manually list every value** they want to try per parameter. The goal of this plan is to make that **automatic** by allowing the user to specify **ranges and steps** (e.g. “min_vrp from 0.01 to 0.10 step 0.01”) and have the system **expand** that into the full list of values before running the grid.

---

## 2. Current Sweep Format and Where It Is Used

- **Current sweep file format:** A single YAML file (e.g. `config/vrp_sweep.yaml`) whose content is a **single dict** where:
  - **Keys** = parameter names (must match the strategy’s constructor argument dict, e.g. for VRP the “thresholds” dict).
  - **Values** = **lists** of values to try. Every combination is run (Cartesian product).

- **Example of current format (explicit list):**
  ```yaml
  min_vrp: [0.02, 0.05, 0.08]
  max_vol_regime: ["VOL_LOW", "VOL_NORMAL"]
  ```
  This runs 3 × 2 = 6 experiments: each of the three `min_vrp` values is paired with each of the two `max_vol_regime` values.

- **Where this is loaded and used:**
  - **Config loading:** `src/analysis/research_loop_config.py` defines a dataclass `StrategySpec` with fields: `strategy_class: str`, `params: Dict[str, Any]`, `sweep: Optional[str]` (the path to the sweep file). The YAML config is parsed and each strategy entry becomes a `StrategySpec`. The `sweep` field is just a string path (relative to project root or absolute).
  - **Running the sweep:** In `scripts/strategy_research_loop.py`, for each strategy spec, if `spec.sweep` is set and the file exists, the script opens that file and does `grid = yaml.safe_load(f)`. Then it calls `runner.run_parameter_grid(strat_cls, base_config, grid)`.
  - **Grid execution:** In `src/analysis/experiment_runner.py`, the method `run_parameter_grid(self, strategy_cls, base_config, param_grid)` expects `param_grid` to be a `Dict[str, List[Any]]`. It does:
    - `combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]`
    - For each combination, it merges with `base_config.strategy_params` and runs one experiment (replay). So **the contract is: param_grid must be a dict mapping parameter names to lists of values.**

- **Important:** The rest of the pipeline (evaluation, ranking, kill rules, allocation) already consumes the **output** of these experiments. No change is needed there. The only change is **how the grid dict (param_grid) is produced**: today it is exactly the YAML content; the goal is to allow the YAML to describe **ranges/steps** and have a small layer **expand** that into the same `Dict[str, List[Any]]` before calling `run_parameter_grid`.

---

## 3. Strategy Parameters (VRP Example)

- The primary strategy in use is **VRPCreditSpreadStrategy**. It is instantiated with a single argument: a **dict** called “thresholds” in the code. In the config, the user specifies this dict as `params` (and sweep rows are merged into it).
- **Relevant parameter names and types** (for VRP):
  - `min_vrp`: float. Typical range e.g. 0.01–0.15, step 0.01 or 0.02. Meaning: minimum IV − RV to trigger a signal.
  - `max_vol_regime`: string. Categorical, e.g. `"VOL_LOW"`, `"VOL_NORMAL"` (no numeric range).
  - `avoid_trend`: string. Categorical, e.g. `"TREND_DOWN"`.
  - `allow_alpaca_iv`: bool. Usually just `true` or `false` (two values).

- So “automatic” is most useful for **numeric** parameters like `min_vrp` (range + step). Categorical and boolean parameters can stay as explicit lists in YAML.

---

## 4. Goal: What “Automatic” Means Here

- **User-facing goal:** The user can write in the sweep YAML something like:
  - “min_vrp from 0.01 to 0.10 step 0.01” (or equivalent YAML structure),
  and the system **expands** that to the list `[0.01, 0.02, 0.03, ..., 0.10]` and then runs the grid as today.
- **Explicit list format must keep working:** Any parameter that is already a list (e.g. `min_vrp: [0.02, 0.05, 0.08]`) should continue to be used as-is. So the sweep file can **mix**:
  - **Explicit lists** (unchanged behavior),
  - **New “range” spec** (min/max/step or min/max/num_steps) that gets expanded to a list before `run_parameter_grid` is called.

- **No change to:** `StrategySpec`, the research loop config schema (still `sweep: path` to a file), or the contract of `run_parameter_grid(param_grid: Dict[str, List[Any]])`. The only addition is a **preprocessing step**: when loading the sweep file, if a value is a dict describing a range (e.g. `{min: 0.01, max: 0.10, step: 0.01}`), expand it to a list; otherwise if it’s already a list, use it as-is. The result is still a `Dict[str, List[Any]]`.

---

## 5. Concrete File Paths and Entry Points

- **Config file (user edits):** `config/research_loop.yaml`. Contains `strategies:` list; each item has `strategy_class`, `params`, `sweep` (path or null).
- **Config parsing:** `src/analysis/research_loop_config.py` — `load_research_loop_config()`, `StrategySpec` dataclass. The `sweep` field is a string path; no change needed here.
- **Where sweep file is read:** `scripts/strategy_research_loop.py`. In the loop over `config.strategies`, when `spec.sweep` is set and the file exists, it does:
  - `with open(sweep_path) as f: grid = yaml.safe_load(f)`
  - `results = runner.run_parameter_grid(strat_cls, base_config, grid)`
  So **the place to add expansion logic** is either:
  - **Option A:** In `strategy_research_loop.py`, after `grid = yaml.safe_load(f)`, call a small helper that converts any range specs in `grid` into lists, then pass the result to `run_parameter_grid`.
  - **Option B:** In `experiment_runner.py`, at the start of `run_parameter_grid`, accept that `param_grid` values can be either a list or a “range” dict, and normalize to list there.
  Option A keeps the contract of `run_parameter_grid` strictly `Dict[str, List[Any]]` and localizes “sweep file format” to the script that reads the file. Option B keeps all grid semantics inside the runner. Either is fine; the plan should pick one and state it.

---

## 6. Proposed YAML Shape for a “Range” (for the human/AI to agree on)

- So that the sweep file stays one YAML dict, a **value** can be either:
  - A **list** (current behavior): use as-is.
  - A **dict** with keys that indicate a numeric range, for example:
    - `min`, `max`, `step` — expand to `[min, min+step, min+2*step, ...]` up to and including the last value ≤ max (or use a small tolerance to include max).
    - Or `min`, `max`, `num_steps` — expand to `num_steps+1` evenly spaced values from min to max inclusive.
- **Example sweep file with mix of explicit list and range:**
  ```yaml
  # min_vrp: automatic range 0.01 to 0.10 step 0.01
  min_vrp:
    min: 0.01
    max: 0.10
    step: 0.01
  # Categorical: keep explicit list
  max_vol_regime: ["VOL_LOW", "VOL_NORMAL"]
  ```
  After expansion, `min_vrp` becomes `[0.01, 0.02, ..., 0.10]` (10 values), and the grid is 10 × 2 = 20 combinations.

- **Edge cases to handle:** step direction (positive step only is enough for now); ensure the last value does not exceed `max` (or is exactly `max`); if `num_steps` is used, avoid division-by-zero when `num_steps` is 0.

---

## 7. Implementation Plan (Short Checklist)

1. **Define the range spec:** Document (in code or in this doc) the exact keys: e.g. `min`, `max`, `step` OR `min`, `max`, `num_steps`. Decide whether to support both (e.g. if `step` is present use step; else if `num_steps` is present use that).
2. **Implement a small helper** `expand_sweep_grid(raw: Dict[str, Any]) -> Dict[str, List[Any]]`:
   - For each key in `raw`, look at the value:
     - If it’s a list → use as-is (or copy).
     - If it’s a dict with range keys → generate list of numbers (floats or ints as appropriate), then assign that list.
     - Otherwise (e.g. single scalar) → optionally treat as single-value list `[value]` for consistency, or reject.
   - Return a dict that satisfies `run_parameter_grid`: every value is a list.
3. **Integrate:** In `scripts/strategy_research_loop.py`, after `grid = yaml.safe_load(f)`, call `grid = expand_sweep_grid(grid)` (or equivalent), then pass `grid` to `runner.run_parameter_grid(...)`.
4. **Tests:** Add a small unit test: given a dict with one range spec and one explicit list, assert the expanded result has the expected lists and that `itertools.product` over them has the expected number of combinations.
5. **Docs:** Update `docs/strategy_research_loop.md` (or the research engine doc) to describe the sweep file format: explicit list unchanged; new option “range” with `min`/`max`/`step` (and optionally `num_steps`), with one example.

---

## 8. What Not to Do (Scope)

- Do **not** change the research loop config schema for strategies (no new required keys). `sweep` remains a path to a YAML file.
- Do **not** change the signature or contract of `run_parameter_grid` in `experiment_runner.py` if Option A is used (only the *content* of the dict passed to it is expanded before the call). If Option B is used, the contract is “param_grid values can be list or range dict” and normalization happens inside the runner.
- Do **not** add GPU or Heston logic; this is only about how the **parameter grid** for the research loop is built. GPU Heston is for options pricing/PoP, not for strategy parameter search.
- Do **not** implement automatic *selection* of which parameters to sweep or what ranges to use (e.g. no ML or Bayesian optimization in this plan). The user still chooses parameters and ranges in the sweep YAML; the only automation is **expanding a range+step into a list**.

---

## 9. Summary for the AI

- **Codebase:** Argus; research loop script `scripts/strategy_research_loop.py`; config `config/research_loop.yaml`; strategy specs in `src/analysis/research_loop_config.py` (`StrategySpec`); experiment runner `src/analysis/experiment_runner.py` (`run_parameter_grid`).
- **Current behavior:** Sweep file = YAML dict of param name → **list of values**. All combinations are run. Grid is passed to `run_parameter_grid(strat_cls, base_config, grid)`.
- **Desired behavior:** Sweep file may also contain param name → **range spec** (e.g. `min`/`max`/`step`). Before calling `run_parameter_grid`, expand every range spec to a list of values; pass the same `Dict[str, List[Any]]` as today.
- **Deliverable:** A small, explicit plan and (if implementing) one helper to expand the grid, integration in the script that loads the sweep file, a unit test, and a short doc update. No change to strategy code or evaluation pipeline.

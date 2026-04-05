# Hardmax Structural Controller Lane

## Current status

The first hardmax branch split into two separable components:

- structural conditioning path: real
- confidence-gated residual router: currently wrong

This is now supported by both the training result and the controller-native diagnostics.

## Results

### AB2: wallclock-capped 1h

- control: `1.72285249` BPB
- full hardmax route: `1.98164836` BPB

Interpretation:

- the controller learned quickly
- but the implementation was too slow and the routed branch lost badly under the 1h wallclock budget

### AB3: step-matched 800-step follow-up

- control: `1.88248754` BPB
- `structonly`: `1.86855732` BPB
- `fullroute`: `1.89072854` BPB

Interpretation:

- the structural path itself is positive
- the current routing rule is the bad part

### AB4: collapse probe

- control: `1.88495975` BPB
- `structonly`, `1` state: `1.88201609`
- `structonly`, `8` states: `1.87697373`
- `structonly`, `8` states, stronger anti-collapse: `1.86285218`

Interpretation:

- even the `1`-state branch slightly beats control
- more than one state helps, because `8` states beat `1`
- stronger anti-collapse pressure produces the best hardmax branch so far
- but this still does **not** mean the controller is healthy

## Diagnostic follow-up

Residual smokes were run on two validation batches and copied into:

- [20260403_hardmax_structural_diagnostics/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_structural_diagnostics/README.md)

The key findings are:

- `structonly` improves mean NLL on the smoke slice
- generic NLL ACF is not better
- argmax residual ACF is better
- boundary gains are mixed rather than strongly concentrated at boundaries

The controller-native tool [analyze_hardmax_structural_controller.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_hardmax_structural_controller.py) then revealed the real failure mode:

- both `structonly` and `fullroute` have fully collapsed to one state
- confidence is constant `1.0`
- budget is constant `0.4`
- the controller is not dynamic

So the branch currently wins only as a static adapter.

The AB4 anti-collapse branch partially improves this:

- confidence and budget are no longer constant
- confidence anticorrelates with NLL and with boundary strength in the right direction
- but hard argmax state usage is still almost fully collapsed to one state

So the revived signal currently lives more in the soft state distribution than in the hard state path.

## Promotion decision

Promote:

- `structonly` with stronger anti-collapse as the best current hardmax experimental branch

Do not promote:

- `fullroute`
- any claims about dynamic compute allocation
- any claims about successful structural state tracking

## Next steps

1. Run an explicit static-adapter baseline against `structonly`.
2. Run controller-native diagnostics on the `1`-state and `8`-state branches to cleanly decompose adapter gain vs state gain.
3. Increase anti-collapse pressure and re-test state usage before touching routing again.
4. Only revisit routing once:
   - more than one state is used materially
   - confidence varies across positions
   - budget varies across positions
5. If collapse remains persistent, reinterpret the lane as “structural adapter” rather than “hardmax controller”.

## 2026-04-03 engineering continuation

The next pass implemented two concrete changes in the trainer:

1. A true static structural adapter baseline in [logic_register_mlx.py](/home/zaytor/transformer_research/parameter-golf/logic_register_mlx.py) and [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py), enabled with `HARDMAX_STRUCT_STATIC_ADAPTER=1`.
2. Temperature annealing for the discrete controller via:
   - `HARDMAX_STRUCT_TEMPERATURE_START`
   - `HARDMAX_STRUCT_TEMPERATURE_END`
   - `HARDMAX_STRUCT_TEMPERATURE_ANNEAL_FRAC`

The key intent is to answer two questions cleanly:

- how much of the hardmax gain is just “extra structural adapter capacity”?
- does delaying hard commitment help the discrete state path stay alive?

## AB5 launched

Bundle:

- [20260403_185901_mlx-hardmax-structural-ab5-adapter-anneal/manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_185901_mlx-hardmax-structural-ab5-adapter-anneal/manifest.json)

Launch session:

- unified exec session `46368`

Runs:

1. control
2. static adapter
3. `1`-state controller
4. `8`-state controller
5. `8`-state anti-collapse
6. `8`-state anti-collapse + annealing

Routing remains off for the whole sweep. This round is only about:

- adapter gain vs controller gain
- collapse under straight-through hard assignment
- whether annealing helps the controller stay non-degenerate

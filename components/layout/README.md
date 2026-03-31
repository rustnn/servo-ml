# Layout Component Guidance

This component owns Servo's layout integration, including CSS Grid through the
workspace-patched Taffy crate in `scratchpad/taffy`.

## CSS Grid Work

- Treat the CSS Grid specification as the source of truth. The indexed spec is
  available through `search-bs` as `css-grid`.
- For explicit-grid and auto-repeat changes, consult CSS Grid section 7.1
  (The Explicit Grid), section 7.2 (Explicit Track Sizing), and section 7.2.3.2
  (Repeat-to-fill: `auto-fill` and `auto-fit` repetitions).
- Auto-repeat bookkeeping must operate on expanded explicit tracks, not on the
  number of template syntax components. A template like
  `repeat(2, 10px) repeat(auto-fill, 20px)` contributes two explicit tracks
  before the auto-repeat clause is expanded.
- check the plan in `scratchpad/taffy-grid-test-analysis.md`

## Validation

- Validate layout and Taffy changes with `./mach build --release` and `./mach check`.
- For CSS Grid behavior changes, run targeted WPT coverage from
  `tests/wpt/tests/css/css-grid` with `./mach test-wpt --release /css/css-grid/...`.
- For a given change, if you've already ran `./mach build`, then there is no need for an additional `./mach check`.

## Layout Test Workflow

- When validating CSS Grid or Taffy behavior with release WPT runs, build first with
  `./mach build --release` and then run `./mach test-wpt --release ...` against the
  narrowest useful path.
- Do not mix `./mach check` into the same validation flow when the release build is
  already the prerequisite for the tests. Keep `./mach check` as a separate sanity
  pass.
- Prefer isolated test runs when a reftest hangs or crashes. Running a single file with
  `--processes=1 --log-raw=-` makes it much easier to distinguish a deterministic
  layout bug from suite noise.
- Prefer clean, rule-isolated subgrid tests over broad unexpected-result counts. Tests
  whose references depend only on features Servo already supports are better progress
  signals than mixed-feature reftests.
- For visual CSS Grid mismatches, rendering a single test page directly through
  `target/release/servoshell --headless -o ... file:///...` is useful for checking the
  actual geometry before chasing reftest infrastructure details.
- Avoid leaving screenshot or ad hoc WPT processes running while rerunning targeted
  tests. Stray background runs can keep the WPT ports bound and contaminate later test
  invocations.
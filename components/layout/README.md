# Layout Component Guidance

This component owns Servo's layout integration, including CSS Grid through the
workspace-patched Taffy crate in `scratchpad/taffy`.

## CSS Grid Work

- Treat the CSS Grid specification as the source of truth. The indexed spec is
  available through `search-bs` as `css-grid`, and the full spec is available at
  `scratchpad/css-grid-spec.html`.
- Keep the workspace `taffy` dependency patched to `scratchpad/taffy` when
  working on Servo's Taffy adapter. The code under
  `components/layout/taffy/stylo_taffy/` tracks the vendored crate API and can
  fail to build against the published crates.io release when that patch is
  missing.
- For explicit-grid and auto-repeat changes, consult CSS Grid section 7.1
  (The Explicit Grid), section 7.2 (Explicit Track Sizing), and section 7.2.3.2
  (Repeat-to-fill: `auto-fill` and `auto-fit` repetitions).
- Auto-repeat bookkeeping must operate on expanded explicit tracks, not on the
  number of template syntax components. A template like
  `repeat(2, 10px) repeat(auto-fill, 20px)` contributes two explicit tracks
  before the auto-repeat clause is expanded.
- check the plan in `scratchpad/taffy-grid-test-analysis.md`
- Keep the current subgrid bring-up status in `scratchpad/taffy-subgrid-plan.md`.
- For subgrid placement work, treat raw candidate spans and final clamped spans as
  different concepts. Bounds checks need to happen on the raw candidate before the
  placement is clamped back into the explicit subgrid span, otherwise auto-placement
  can probe the same occupied area indefinitely.
- When a CSS Grid failure is performance-sensitive, first determine whether the issue
  is search-path behavior or final geometry. Placement loops and visual sizing
  mismatches often need different debugging strategies.

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
- Raw `--log-raw` output is easier to reuse if it is written to a file and inspected
  with `scratchpad/reftest_log_tool.py` rather than pasting it into the old XHTML
  analyzer.
- Use `python3 scratchpad/reftest_log_tool.py summary <log>` to list reftest failures
  and `python3 scratchpad/reftest_log_tool.py extract <log> --output-dir <dir>` to
  decode embedded screenshots and manifests for later inspection.
- For visual CSS Grid mismatches, rendering a single test page directly through
  `target/release/servoshell --headless -o ... file:///...` is useful for checking the
  actual geometry before chasing reftest infrastructure details.
- Avoid leaving screenshot or ad hoc WPT processes running while rerunning targeted
  tests. Stray background runs can keep the WPT ports bound and contaminate later test
  invocations.
- When comparing reftest failures, use the smallest reproducible set first:
  isolated single-test reruns, then small paired reruns for regression checking, and
  only then broader directory coverage.
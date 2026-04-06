# Layout Component Guidance

This component owns Servo's layout integration, including CSS Grid through the
workspace-patched Taffy crate in `scratchpad/taffy`.

For all layout work, including edits in `scratchpad/taffy`, this README is the
authoritative component guide after `AGENTS.md`.

## CSS Grid Work

- Treat the CSS Grid specification as the source of truth. The full spec is available at
  `scratchpad/css-grid-spec.html`.
- For standalone `grid-template-columns` / `grid-template-rows` resolved values, serialize the
  used track sizes with the explicit line names authored in the track list, but do not inject
  line names synthesized from named areas or later placement. Spec: https://drafts.csswg.org/css-grid/#resolved-track-list-standalone
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
- Your goal is to get as many unexpected PASS results(for tests that currently expect FAIL), with as little unexpected FAIL as possible.
- At the end of each task, output a good commit message for the changes as part of your final response.
- For reftest, you need to also pay attention to the reference test implementation. Usually, the test uses some "new" tech, but the tech used in the ref might also lack some implementation in servo/taffy, which should be taken into account. Ideally, only pay attention to tests where the tech used in the ref is well supported, or add support for it first.
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
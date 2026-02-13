Agents — quick orientation

This file gives a minimal orientation to contributors/agents working in this
repo. It only points you at component READMEs — read those for implementation
conventions and deeper guidance.

Repo structure (high level)
- components/ — Rust crates that implement browser subsystems (DOM, layout,
  Web APIs, etc.). Check the README inside each component for specifics.
- third_party/, tools/, docs/, etc. — supporting materials and tooling.

Where to look next
- For DOM / Web API work: see `components/script/README.md`.
- For feature-specific guidance: open the README in the component you'll
  modify (each component directory may contain a README).

That's it — run the component README(s) for details relevant to your task.
Agents — quick orientation

This file gives a minimal orientation to contributors/agents working in this
repo. It only points you at component READMEs — read those for implementation
conventions and deeper guidance.

Repo structure (high level)
- components/ — Rust crates that implement browser subsystems (DOM, layout,
  Web APIs, etc.). Check the README inside each component for specifics.
- third_party/, tools/, docs/, etc. — supporting materials and tooling.

Do not use cargo, use `./mach` commands.

Where to look next
- For DOM / Web API work: see `components/script/README.md`.
- For feature-specific guidance: open the README in the component you'll modify (each component directory may contain a README).

README chain (read in order)
- `AGENTS.md` — top-level agent orientation and cross-cutting rules (this file).
- `components/<component>/README.md` — component-level guidance and the consolidation point for component-wide lessons (for example `components/script/README.md`).
- `components/<component>/dom/<subdir>/README.md` — subsystem-level guidance (keep *very* minimal; include only subsystem-specific notes, spec anchors, and TODOs).
- `specs/<spec_name>/index.bs` — the authoritative spec for algorithms and internal-slot definitions.

Agent pre-task checklist (MANDATORY)
- Step A: Read and add to your working context the README chain *for the task* in this order: `AGENTS.md` → `components/<component>/README.md` → `components/<component>/dom/<subdir>/README.md` (if present) → `specs/<spec_name>/index.bs`.
- Step B: Confirm in your first reply which README(s) you loaded and which `specs/.../index.bs` section(s) you will follow.
- Step C: If any required README or spec is missing or ambiguous, stop and ask a clarifying question before changing code.
- Step D: Follow the documented conventions in those READMEs (for example the `components/script/README.md` "Documenting your work" rules) when implementing and commenting code.

Principle: add lessons to the *lowest* README that makes sense. Do **not** duplicate or copy the same prose across multiple README files — put the short lesson where it belongs and, if broadly applicable, add a one-line pointer in the parent README.

Example — working on WebNN (in `script`)
- Read in order: `AGENTS.md` → `components/script/README.md` → `components/script/dom/webnn/README.md`.
- Implement code under `components/script/dom/webnn/` and add WebNN-specific lessons to `components/script/dom/webnn/README.md`.
- If a lesson applies to many Web APIs, add a single short line to `components/script/README.md` (do not copy the full prose into the subsystem README).

Guidance on adding documentation

Whenever the user corrects your code, besides fixing the code, if there
is a general lesson to document, add prose to the lowest-level possible `README.md` file. 
For example, if you learn something related only to the webnn implementation, then
add docs to `components/script/dom/webnn/README.md`, but if the lesson is relevant 
to any web api, then add docs to `components/script/README.md`. You can also add `README.md` files; ensure those are reffered-to from the `README.md` file at the level above. For example: `components/script/README.md` should refer to the `README.md` files found in `components/script/dom/`.

Prose & README style (for agents)

- Document the *current* design/state only — do **not** leave change‑history or "I did X" comments in source or README files (for example, avoid comments like "create a single sender"). Historical context belongs in the PR description or a changelog, not inline.
- Use neutral, factual language. Avoid subjective or minimizing words such as "small", "tiny", "minimal", "just", or "only" when describing a component or its responsibilities.
- Automated agents must not add conversational, transient, or change‑history text to committed files — write stable, forward‑looking prose suitable for future readers.
- Prefer importing shared message/types with `use` and referring to them unqualified in code (for example, `use webnn_traits::WebNNMsg;` then use `WebNNMsg`), rather than repeatedly using fully‑qualified paths like `webnn_traits::WebNNMsg`. (Keeps signatures and types concise.)
- If you need to capture why a change was made, put that explanation in the PR/commit message or an appropriate changelog entry.

- Always double-check your work with `./mach check`(without additional arguments). 
- Do not do anything with Git.

IMPORTANT: never use fully qualified imports. To check build, always use ./mach check with zero additional arguments

That's it — find and run the component README(s) for details relevant to your task.
components/script/dom/webnn — implementation notes (minimal)

- Read the README chain before changing WebNN code: start with `AGENTS.md` (top-level agent orientation), then `components/script/README.md` (component guidance), then this file for WebNN-specific notes. Do **not** duplicate content across README files; subsystem READMEs must be concise and only contain subsystem-specific notes, spec anchors, and TODOs.

- Check the normative spec at `specs/webnn/index.bs` before implementing behavior.

- The `MLContext` exposes a `timeline` concept in the spec. In this codebase model the timeline should be implemented as steps enqueued to a backend/task queue. This is work-in-progress; keep API implementations minimal and add TODOs for backend messaging. If an in-parallel TODO would resolve a Promise, do not resolve it in the stub — return the promise unresolved.

WebNN-specific expectations (short):

- Method-level doc: top doc-comment = canonical spec anchor only (no parenthetical/top-doc prose).
- Implementation comments: use `Step N:` comments for spec-mapped steps inside function bodies.
- Internal slots: document struct fields with the canonical `-slot` anchor when present.
- Tests: add WPT tests (do not add component-local tests).

Important: this README is subsystem-specific — read `components/script/README.md` and `AGENTS.md` first. Keep this file focused on WebNN-specific anchors and TODOs.
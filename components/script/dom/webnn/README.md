components/script/dom/webnn — implementation notes (minimal)

- Check the normative spec at `specs/webnn/index.bs` before implementing behavior.

- The `MLContext` exposes a `timeline` concept in the
  spec. In this codebase model the timeline should be implemented as steps
  enqueued to a backend/task queue (i.e. operations scheduled to run on a
  worker thread or device command queue). This is a work-in-progress;
  for now keep API implementations minimal and follow the spec algorithms
  when fleshing out the timeline behaviour. Just add TODO notes in the code for the actual messaging with the backend. 
  - Any steps meant to run on the timeline, or in-parallel, for now should just be left as TODOs. If the steps involve resolving a promise, no need to try to resolve the promise, just return the promise and let it never resolve for now. 

  The implementation will follow the below high-level plan:

  1. Implement most of the API surfarce here in script.
  2. Adding backend functionality for the timeline concept
  3. Integrating the two via messaging using the `GenericCallBack` and other mechanism. WebGPU and the Indexedbb factory and connection opening code are good examples for this part. 

  TODO: 

  - [x] Implement basic stub for MLContext and the navigator ML interface.
  - [x] Implement `CreateContext`.

  WebNN-specific expectations (short):

  - Method-level doc: the top doc-comment for an exposed method should contain *only* the canonical spec anchor (e.g. `/// <https://...#api-ml-createcontext>`).
  - Implementation comments: inside the function body annotate each relevant line of code with a single comment of the exact form `Step N: <spec prose>` (use `Step 5.1`, `Step 5.2` for sub-steps). Do not paste the entire algorithm block.
  - TODO behaviour: if you leave an in-parallel step as a TODO and that step would resolve a promise, do *not* resolve the promise in the stub — return the promise unresolved and leave resolution to the future queued ML task implementation.
  - Internal slots: document struct fields that back internal slots using a single-line doc-comment that contains only the canonical spec anchor in angle brackets (e.g. `/// <https://webmachinelearning.github.io/webnn/#api-mlcontext>`).
  - Add an in-code `TODO` (with spec anchor) for any intentionally minimal/stub behaviour (e.g. missing ML task queuing, backend work).
  - Tests: Do **not** add tests inside `components/script`. Add Web Platform Tests (WPT) for spec behavior instead — WPT is the canonical place for Web API conformance checks.

  Lessons learned (Accelerated & CreateContext):
  - `MLContext` internal-slot backing fields should link to the internal-slot anchor when present (for `[[accelerated]]` use `#dom-mlcontext-accelerated-slot`); if no `-slot` anchor exists, fall back to the attribute/getter anchor.
  - The `Accelerated()` getter must return the stored internal-slot value (see `MLContext::Accelerated`).
  - `CreateContext` must use the method-level spec anchor only in its top doc-comment, annotate implementation with per-line `Step N:` comments, create the `Promise` in the current realm, and leave in-parallel backend steps as TODOs. If a TODO'd step would resolve a promise, return the promise unresolved here and queue resolution via the ML task system later.
  - Always include a `// TODO (spec: #api-...)` comment for intentionally unimplemented timeline/backend work.

  Important: this README is *subsystem-specific*. Read `components/script/README.md` first and follow its rules — do NOT duplicate the parent README's content here. Keep this file limited to WebNN-specific notes, spec anchors, and TODOs.
https://webmachinelearning.github.io/webnn/

components/script/dom/webnn — implementation notes (minimal)

- Read the README chain before changing WebNN code: start with `AGENTS.md` (top-level agent orientation), then `components/script/README.md` (component guidance), then this file for WebNN-specific notes.

- Use the canonical WebNN spec via `search-bs` (do **not** rely on local `specs/` HTML copies). Recommended workflow for finding anchors and quoting spec prose:
  1. Index the spec (if not already indexed):
     `search-bs index https://github.com/webmachinelearning/webnn/blob/main/index.bs --name webnn`
  2. Locate the API/algorithm anchor and preview matches:
     `search-bs search --name webnn "cast" --around 2`
     - Note the returned `anchor` (for example `api-mlgraphbuilder-cast`) and the `line` number.
  3. Retrieve the exact lines you want to quote or paste into code comments:
     `search-bs get --name webnn --line <LINE> --count <N>`
  4. Use the anchor in top doc-comments exactly, for example:
     `/// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-cast>`
  5. Copy spec step prose with `search-bs get` and annotate code using `Step N:` comments quoting the spec text.

- The `MLContext` exposes a `timeline` concept in the spec. In this codebase model the timeline should be implemented as steps enqueued to a backend/task queue. See the `webnn` top-level component for more info. 

- Context identifiers (`ContextId`) are defined in `components/shared/base/id.rs` as
  `MLContextId` via the `namespace_id!` helper.  This makes them globally unique across
  worker threads even when they share the same `PipelineId`.  All other WebNN objects
  (graphs, tensors, etc.) are scoped to their context and never used on their own, so the
  system relies on the context id to guarantee global uniqueness.  Avoid generating
  additional global ids outside the context namespace; always propagate the owning
  `ContextId` along with any tensor/graph identifiers.

Important: this README is subsystem-specific — read `components/script/README.md` and `AGENTS.md` first. Keep this file focused on WebNN-specific anchors and TODOs.

- The `scratchpad/` top-level directory contains the source code for `rustnn`, as well as that of `pywebnn`. The latter is basically to Python what this crate is to the Web. Use this example. 

When implmenting methods of GraphBuilder, read how `PyMLGraphBuilder` in scratchpad uses rustnn, for all other details, look at existing method in this crate. The code in scratchpad is only an example on how to user rustnn, it is not an example on how to implement webnn in this crate. 

- **When adding a new `MLGraphBuilder` operand method:** update the full WebIDL surface from the spec section, not only the `partial interface MLGraphBuilder` method signature. If the spec section also defines companion structures (for example `partial dictionary MLOpSupportLimits` members), add them in `WebNN.webidl` at the same time. In Rust implementations, keep `Step N:` comments in the same order as the spec even if execution order differs for backend-id plumbing; add a `Note:` under the ordered step block explaining the implementation ordering, then repeat the verbatim step comment at the actual code site where it executes. For garbage-collection plumbing, add the method name to `components/script_bindings/codegen/Bindings.conf` (`canGc`) and use the generated `can_gc` argument instead of inserting `CanGc::note()` in new method bodies.

- **Method doc anchors:** for WebIDL methods (for example `MLGraphBuilder.abs()`), use the method-level anchor in top doc-comments (for example `#dom-mlgraphbuilder-abs`) rather than a section-level anchor such as `#api-mlgraphbuilder-unary`.

- **When the spec defines a named algorithm and another algorithm calls it:** implement the named algorithm as a separate Rust helper (`fn` if `self` is not required, method if `self` is required), link that helper's top doc-comment to the algorithm anchor, and call that helper from the caller algorithm. Name helpers as idiomatic Rust translations of the algorithm wording (for example, "create an element-wise binary operation" → `create_an_element_wise_binary_operation`) rather than anchor-like names. Keep helper and caller `Step N:` comments verbatim from `search-bs get` output.

- **Backend datatype support:** the CoreML executor now handles `Int32` outputs (output floats from CoreML are truncated to `i32`), and inputs of type `Int32` are promoted to `float32` before dispatch. The script-side read callback reconstructs little-endian `i32` values and returns an `Int32Array`; the conversion matches the backend logic. This mirrors the behaviour of the reference `dispatch_example` in the `skills/` directory.

Overview — how WebNN is wired in Servo

- Message type and API surface
  - `WebNNMsg` (IPC messages) lives in `components/shared/webnn` so DOM code, the Constellation, and the manager can share the type without creating dependency cycles. `WebNNMsg` is `Serialize`/`Deserialize` so it is safe to pass via `GenericSender/Receiver`.

- Manager thread / backend
  - The WebNN manager lives in `components/webnn`. It owns a worker thread, receives `WebNNMsg` messages, and routes work to the chosen backend (native library, `rustnn`, GPU driver, etc.). Currently the manager is a stub that handles `Exit`; replace the stub's loop with backend task routing when adding a backend.

- Constellation lifecycle
  - `Servo::new` constructs the manager (returns `(sender, JoinHandle)`), stores the sender in `InitialConstellationState`, and forwards the join handle so `Constellation` can `join()` the thread during shutdown.
  - `Constellation::handle_shutdown()` sends `WebNNMsg::Exit` and joins the manager thread — this ensures clean shutdown.

- Script/DOM access
  - The `GenericSender<WebNNMsg>` is threaded into `InitialScriptState` → `ScriptThread` → `GlobalScope` and exposed via `GlobalScope::webnn_sender()` for DOM/WebIDL implementations (e.g. `MLContext`, `MLGraphBuilder`) to send backend requests.
- Sending a message to the webnn backend should never fail, because the constellation keeps that component alive until all script has shut-down.  If a DOM-originated send ever fails it indicates a serious bug and should be logged at `error!`.
  
- Within the backend/ML thread itself the `manager_tx` is used to notify the manager of
  results or failures.  Those sends are best-effort; the current implementation logs
  `warn!` on failure (see the compile/compute handling above) rather than panicking the
  thread, because a manager shutdown race is non‑fatal and we prefer to drop the result
  quietly.

- Logging policy for DOM→manager sends: if sending a `WebNNMsg` from the DOM to the manager fails, log at `error!` (not `warn!`). The Constellation guarantees the manager thread exists until clean shutdown, so a send failure indicates a serious/unexpected problem and should be surfaced at error level.

Keep this README focused — see `components/script/README.md` for cross-cutting rules and `AGENTS.md` for contributor guidance.

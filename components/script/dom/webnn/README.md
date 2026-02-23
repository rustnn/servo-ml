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

Important: this README is subsystem-specific — read `components/script/README.md` and `AGENTS.md` first. Keep this file focused on WebNN-specific anchors and TODOs.

- The `scratchpad/` top-level directory contains the source code for `rustnn`, as well as that of `pywebnn`. The latter is basically to Python what this crate is to the Web. Use this example. 

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

- Logging policy for DOM→manager sends: if sending a `WebNNMsg` from the DOM to the manager fails, log at `error!` (not `warn!`). The Constellation guarantees the manager thread exists until clean shutdown, so a send failure indicates a serious/unexpected problem and should be surfaced at error level.

Keep this README focused — see `components/script/README.md` for cross-cutting rules and `AGENTS.md` for contributor guidance.

components/webnn — WebNN manager crate (runtime/backend stub)

Purpose
- Manager crate that owns the WebNN backend worker thread and
  exposes a factory for obtaining a `GenericSender<WebNNMsg>` the rest of
  the engine can use to send requests to the backend.

What this crate provides
- `new_webnn_manager() -> (GenericSender<WebNNMsg>, JoinHandle<()>)`
  - Spawns a worker thread that receives `WebNNMsg` messages and
    performs backend work (currently a minimal stub that handles `Exit`).
  - Returns the `sender` for use by `Servo` / `Constellation` / `Script` and
    the `JoinHandle` so the owner can join the worker on shutdown.
  - Note: the manager does **not** perform a default/naive execution of graphs; if no backend
    is available (for example CoreML is not enabled) `Dispatch` is a no-op and the manager
    will emit a warning. Implement a proper backend instead of relying on a fallback.
- Local runtime-only code; the message type is defined in
  `components/shared/webnn` (provides `WebNNMsg` in the `webnn_traits` crate) so message types are
  shared across crates without creating dependency cycles.
- **Timeline behaviour**: each `Context` now maintains a `VecDeque` of
  pending operations.  When a compute dispatch is in-flight subsequent create,
  read, write or dispatch requests are queued and replayed serially once the
  compute finishes.  This implements the WebNN timeline requirement without
  blocking the manager thread on ML work (see `Context::enqueue_or_run` and
  `PendingOp` in `src/lib.rs`).

Wiring & lifecycle (short)
- Construction: `Servo::new()` calls `webnn::new_webnn_manager()` to create
  the manager and obtains `(sender, join_handle)`.
- Wiring: `Servo` passes the `sender` into `create_constellation(...)` where
  the `Constellation` stores it in `InitialConstellationState` and threads it
  into `InitialScriptState` → `ScriptThread` → `GlobalScope` (exposed via
  `GlobalScope::webnn_sender()`).
- Shutdown: `Constellation::handle_shutdown()` sends `WebNNMsg::Exit` to
  the manager and joins the `JoinHandle` to ensure clean termination.

Design notes
- Message type location: `components/shared/webnn` (`webnn_traits`) —
  necessary because `GenericSender/Receiver<T>` require `T: Serialize +
  Deserialize` and different crates must share the same Rust type.
- Keep manager logic and backend adapters inside `components/webnn` (or in
  separate backend crates) — DOM/script should only depend on the shared
  message crate and the `sender`.
- Prefer importing synchronization primitives and shared graph/message types at
  module scope and using their short names in code rather than repeating
  fully-qualified paths throughout the manager implementation.
- The ML-thread cache should insert a per-graph entry before asynchronous
  compilation begins and transition that entry through explicit states
  (`Compiling`, `Compiled`, `Destroyed`).  Compute requests for a graph that is
  still compiling must wait on that entry rather than treating the cache as
  missing.
- Implementation note: prefer declaring manager helper types (for example
  `Context`) at module scope rather than inside `run_manager()` so types
  are discoverable, easier to test, and stable for future backend additions.
- When dispatching graphs we currently only support the macOS CoreML backend.
  Graphs with constant operands must have their bytes recorded (`constant-*`
  operations in `GraphInfo`) before conversion; the manager now auto-populates
  missing constant data from `tensor_store` to avoid CoreML parsing errors
  (previously some graphs failed with "Operations are expected to be topologically
  sorted"). 
- The manager makes a best-effort attempt to zero‑fill output tensors if no
  backend is available, preventing script-side reads from hanging.
- Recent refactors moved all model caching off the manager thread and
  into the ML worker itself.  The manager no longer retains any `GraphInfo`
  or compiled‑paths; it prepares pooled work items from the manager thread.
  The manager owns the rayon thread pool, self-sender, and shared compile
  cache, and each `Context` stores clones of those manager-owned execution
  handles so queue-draining code can spawn work at the lowest level without
  threading an extra execution-context parameter through unrelated paths. The
  cache keeps a per-graph entry with explicit `Compiling`,
  `Compiled`, and `Destroyed` states so repeated compile requests wait on the
  existing entry instead of launching duplicate compilation work. Dispatch
  messages now carry only the graph id and tensor data, significantly reducing
  manager memory pressure and avoiding cross-thread cloning of the graph
  description.
- Script-side callbacks for compilation (`ContextMessage::CompileResult`) only
  supply the `GraphId`.  Script code keeps a clone of the `GraphInfo` when
  `build()` is invoked so that it can validate dispatch arguments later; the
  manager never forwards the blob.
- In multiprocess mode manager threads started with Constellation run in
  the Constellation process (not the Script/content process).

Tests
- Do not add tests; everything should also be covered by WPT tests. 

See also
- `components/shared/webnn` — shared message types (canonical `WebNNMsg`).
- `components/script/dom/webnn/README.md` — WebNN DOM/implementation notes.

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
- Local runtime-only code; the message type is defined in
  `components/shared/webnn` (provides `WebNNMsg` in the `webnn_traits` crate) so message types are
  shared across crates without creating dependency cycles.

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
- In multiprocess mode manager threads started with Constellation run in
  the Constellation process (not the Script/content process).

Extending the backend
- Replace the manager's stub loop with task routing to a backend (GPU,
  rustnn, platform library, etc.).
- Use Promises on the JS side and route long-running work to the manager
  thread(s); do not block Script/Constellation event loops.

Tests
- Add WPT tests for the WebNN API surface.
- Optional: small unit/smoke tests inside `components/webnn` to validate
  message handling and Exit/join semantics.

See also
- `components/shared/webnn` — shared message types (canonical `WebNNMsg`).
- `components/script/dom/webnn/README.md` — WebNN DOM/implementation notes.

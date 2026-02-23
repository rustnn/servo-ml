components/script — README

**Purpose**
- `components/script` contains the implementation of the DOM and Web API
  bindings used by Servo's JavaScript engine layer.

**Structure overview**
- dom/ — DOM types and implementations (each WebIDL interface maps to a
  Rust `#[dom_struct]` type and a generated `*Methods` trait).
- bindings/ — the part of generated bindings glue code and helpers that hasn't been moved yet to `components/script_bindings/`.
- The codegen--like the WebIDL files and the config files--infra is found found in the top-level partner component `components/script_bindings/webidls`.

**Searching web specs (use search-bs)**
- To access web standars (specs) always use the `search-bs` command-line tool.
  - Index (one-time, if you do this you can add a note to the relevant README to remember it): `search-bs index https://github.com/webmachinelearning/webnn/blob/main/index.bs --name webnn`
  - Find the API/algorithm anchor and preview matches: `search-bs search --name webnn "cast" --around 2`
    - Look for the anchor id in the result (example: `api-mlgraphbuilder-cast`).
  - Fetch exact lines for quoting or copying spec steps: `search-bs get --name webnn --line <LINE> --count <N>`.
  - The tool is documented in `skills/search-bikeshed/README.md`.

  Example: to find the `MLGraphBuilder.cast()` anchor and extract the method steps:
  1. `search-bs search --name webnn "cast"` → note `dom-mlgraphbuilder-cast` and line number.
  2. `search-bs get --name webnn --line {insert line num} --count 40` → copy the algorithm prose you need.
  3. Make sure you obtain the full algorihtm: do not attempt to hallucinate parts not found in the initial response, instead, make a second `get` call to get the missing parts from the spec.

**Working on a Web API (tips)**
1. Find the WebIDL in `components/script_bindings/webidls/` (or add one if
   implementing a new API).
   - If your method can throw exceptions at runtime, mark it with `[Throws]` in the WebIDL.  The Servo bindings generator will then produce an implementation signature that returns `ErrorResult` (for void returns) or `Fallible<DomRoot<T>>` / `Result<..., Error>` (for methods that return DOM objects). Implementations should `return Err(Error::...)` instead of calling `throw_dom_exception(...)`. Example: mark `MLGraphBuilder.input` with `[Throws]` so `Input()` can return `Err(Error::Type(...))` and the binding will throw in JS.
2. Use the canonical spec as the authoritative source for algorithms and internal-slot definitions, and access it with the `search-bs` commmand line tool.
3. Document code by linking the generated method to the canonical spec anchor in the top-level doc-comment and add per-line `Step N:` comments inside the function body (see **Documenting your work** below for the exact format).
4. Add a `#[dom_struct]` with `Dom` members and implement the generated
   trait methods: start with `todo!()` bodies.
5. Sub-directories (e.g. `dom/webgpu`, `dom/xr`) often include their own
   README with subsystem-specific guidance. Always read the README chain for
   the area you're changing: start with `components/script/README.md`, then
   read any `components/script/dom/<subdir>/README.md` for subsystem rules,
   and finally consult the authoritative canonical online spec for
   algorithm and internal-slot details (link in the subsystem README).
6. Good quality examples of implementation and documentation patterns are found in `components/script/dom/stream`.
7. Prefer top-level `use` imports for types that appear in the file (for
   example `use crate::dom::webnn::mlcontext::MLContext;`) instead of using
   fully-qualified `crate::...` paths inside signatures or bodies.
   - Use short type names in method signatures and code; add the `use` at the
     top of the file. This improves readability and makes future refactors
     and reviews simpler.
8. Consolidate multiple calls to `global()` into one by assigning the global to a variable, and if needed passing it down by reference.

**Adding a backend for a Web API (how-to)**
Follow this checklist when you need a native/task-queue backend for a Web API (for example a manager thread that processes requests):

1. Shared message types
   - Put the cross-crate IPC/message types into `components/shared/<api>` (example: `components/shared/webnn`).
   - Derive `serde::Serialize` and `serde::Deserialize` for every message so `base::generic_channel::GenericSender<T>`/`GenericReceiver<T>` can be used across crate boundaries.

2. Manager component
   - Add a manager crate under `components/<api>` (example: `components/webnn`).
   - Expose a constructor `new_<api>_manager()` that returns `(GenericSender<T>, JoinHandle)` or at minimum a `GenericSender<T>`.
   - Keep the manager implementation local to the component; it should own its worker thread and its message receiver.

3. Wiring into Servo/Constellation
   - Create the manager in `components/servo/servo.rs` and obtain the `sender` (and optional `join_handle`).
   - Pass the `sender` into `create_constellation(...)` through `InitialConstellationState` (add fields there if needed).
   - Store the `JoinHandle` in `InitialConstellationState` as `Option<JoinHandle<()>>` (so Constellation can join on shutdown).
   - In `Constellation::handle_shutdown()` send an Exit/Shutdown message to the manager and join the handle.
   - Note (multi-process): when running in multi-process mode, threads started "alongside the Constellation" run in the Constellation process (i.e. separate from the Script/content process). Design your backend and IPC accordingly — message types must be IPC-safe and any assumptions about in-process access to script state are invalid.

4. Expose sender to script/DOM
   - Add the sender to `InitialScriptState` and thread it through `ScriptThread` → `GlobalScope` so DOM bindings can access it via `GlobalScope::webnn_sender()` (or similar accessor).

5. Avoid dependency cycles
   - Keep message types in `components/shared/<api>` so `script` only depends on the shared crate, not the manager crate.

6. Shutdown & lifecycle
   - The Constellation should be authoritative for thread shutdown: send an Exit message and `join()` the manager thread during `handle_shutdown()`.

7. Tests & spec
   - Behavior of the Web API must be covered by Web Platform Tests (WPT). Do not add or adjusts tests without first making sure it is necessary.
   - Never add unit-tests to `script`. 

8. Documentation
   - Add a short README under `components/<api>/` (and update the subsystem README under `components/script/dom/<api>/README.md`) describing the manager architecture and where message types live.

Example (wiring highlights):
- `let (sender, join_handle) = webnn::new_webnn_manager();` (in `Servo::new`)
- `create_constellation(..., webnn_sender.clone(), Some(join_handle));`
- `InitialConstellationState { webnn_sender, webnn_join_handle: Some(join_handle), ... }`
- `Constellation::handle_shutdown()` → `webnn_sender.send(WebNNMsg::Exit)` + `webnn_join_handle.take().unwrap().join()`

**Documenting your work:**
Follow these exact conventions so code <-> spec mapping is clear and reviewable.

- Method- & type-level doc
  - Method-level: the method's top doc-comment must contain *only* the canonical spec anchor (e.g. `/// <https://webmachinelearning.github.io/webnn/#dom-ml-createcontext>`).
    - Do NOT add parenthetical notes or extra prose in top doc-comments (for example, `(internal helper)`) — these add noise and are disallowed. Keep top-level doc-comments anchor-only.
    If the algorithm implementation is broken-up into multiple method or functions, you can add a note below the anchor to explain which part of the algo the current code corresponds to.

  - Functions & spec-algorithms
    - Follow the structure of the spec. For example, if the spec defines an interface method and then from it calls into another algorithm, then you should also implement that algorithm either with a seperate method(if you need to access state of the dom struct), or just a function.
    - Naming & visibility: name the function to reflect the spec algorithm (`create_an_mloperand`), keep it private by default, and move it to a shared module only if genuinely reused across components.

- In-body per-line spec mapping
  - Inside the function body annotate *each relevant line of code* with a
    single comment of the exact form `Step N: <spec prose>` (use `Step 5.1`,
    `Step 5.2` for sub-steps). Avoid pasting entire algorithm blocks.
    Quote the spec step verbatim in the code comment.
  - If the spec step does not map 1:1 to code, add `// Note: ...` explaining
    the divergence and reference the spec anchor. If the spec's preliminary
    steps (for example `Step 1`/`Step 2` that establish `global`/`realm`) are
    implicit in Rust (e.g. via `self.global()`), still include `Step 1:` and
    `Step 2:` comments and follow them with a `// Note:` explaining the
    implicit mapping.
  - Do *not* use shorthand/aggregation comments such as `Steps 1-5: same precondition
    checks as the non-BYOB variant.` — every algorithm step referenced in the
    spec must appear explicitly (Step N) in the implementation, even when the
    code is identical to another overload. This makes reviewer-to-spec
    mapping unambiguous and prevents accidental divergence.  - Internal slots / struct members: document the field with a single-line
    doc-comment that contains *only* the canonical spec anchor in angle
    brackets. Prefer an *internal-slot* anchor when the spec provides one
    (e.g. `#dom-foo-xyz-slot`). If no `-slot` anchor exists, link the field
    to the attribute getter or the interface/internal-slots section that
    documents the internal slot (for `MLContext.[[accelerated]]` prefer
    `#dom-mlcontext-accelerated-slot`; fall back to `#dom-mlcontext-accelerated`
    only when a `-slot` anchor is not present). Example:

      ```rust
      /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated-slot>
      accelerated: Cell<bool>,
      ```

    - Distinction: generated trait *methods/attributes* map to WebIDL anchors
      like `#dom-mlcontext-accelerated` (the attribute getter); struct fields
      that back *internal slots* should link to the internal-slot anchor where
      available, otherwise link to the attribute/getter or interface anchor.
    - Do not add additional prose when documenting internal-slot fields.

    - DOM struct fields must remain private. Always add `pub(crate)` accessor
      methods (getters/setters) on the `#[dom_struct]` type for other code to
      read or modify internal-slot values. Consumers outside the defining
      module must call these accessors — do *not* access struct fields
      directly from other modules.

    - When a stored value comes from a WebIDL dictionary (for example
      `MLTensorDescriptor`), link the field to the specific dictionary-member
      anchor (for example `#api-mltensordescriptor` / `{{MLTensorDescriptor/readable}}`) so
      the source of truth is obvious.

- TODOs and in-parallel steps
  - For any unimplemented spec step, add a `TODO: {optional short description}`
    immediately below the usual `Step N:` comment. 
  - IMPORTANT: if the TODO corresponds to an *in-parallel* step that would
    resolve a Promise, do *not* resolve the Promise in the stub — return
    the Promise unresolved and leave resolution to the future queued task.

  - Assertions & invariants
    - Do **not** use `panic!` for runtime checks in `components/script` code. Use
      `debug_assert!` for internal invariants that should only fire during
      development (for example `debug_assert!(false, "unexpected state")`).
    - If an invariant can be reached in release builds, return a `Result`/`Error`
      or provide a safe fallback rather than panicking. Library code should
      never abort the process in production.
    - When a helper function implements a spec algorithm and an impossible
      branch is present, prefer `debug_assert!` + a safe release fallback (see
      `mlgraphbuilder::create_an_mloperand` for an example).

- Formatting rules
  - Always leave a blank line after a `Step + code` or `Step + TODO` block.
    (Exception: you do not need to add an extra blank line before the method's
    closing brace solely to satisfy this rule.)
  - Keep comments short and place them on their own line above the code they
    document.

- Tests
  - Do **not** add tests inside `components/script`. All Web API behavior must be validated in Web Platform Tests (WPT); add new tests to the appropriate WPT tree instead.

- Examples & references
  - See `components/script/dom/readablestream.rs` for good examples of these
    conventions.

Correct example:
```rust
/// <https://example.spec/#api-foo-doThing>
fn do_thing(&self, options: &FooOptions) -> Rc<Promise> {
    // Step 1: Let |promise| be a new promise in |realm|.
    let p = Promise::new(&self.global(), CanGc::note());

    // Step 2: TODO — queue the backend work on the Foo task queue (do NOT resolve here).
    // TODO (spec: #api-foo-doThing): implement Foo task queuing and routing.

    // Step 3: Return |promise|.
    p
}
```
```

**Note in finding good examples of code patterns:**
If you struggle with implementing a concept from the spec, it can be useful to find existing code not by searching for code, but by searching for spec prose. 
For example, if the spec reads: "to resolve promise with", then searching for this text in the codebase is likely to surface other code implementing something similar. 

**Using `Bindings.conf` to add generated arguments:**
- The `components/script_bindings/codegen/Bindings.conf` file controls
  per-interface codegen options. Common fields inject additional
  parameters into generated method signatures so you don't need to change
  the WebIDL to get runtime helpers:
  - `inRealms`: causes the generated method to receive an `InRealm`
    argument (useful for methods that create promises or run async work).
  - `canGc`: causes the generated method to receive a `CanGc` argument.
  - `cx`: requests a `JSContext`-style argument for APIs that need a
    SpiderMonkey context.  

Example (Bindings.conf excerpt):

```properties
'Document': {
  'inRealms': ['ParseFromString'],
  'canGc': ['ParseFromString'],
},
```

Result: the generated trait method for `ParseFromString` will include
`comp: InRealm` and `can_gc: CanGc` parameters so your implementation can
use `Promise::new_in_current_realm(comp, can_gc)` and other runtime helpers.

Tip: inspect `components/script_bindings/codegen/Bindings.conf` and the
generated trait in `crate::dom::bindings::codegen::Bindings::{Interface}Binding` to
see the final method signature.

**Refcell re-borrow hazard**
Whenever the JS engine is called into, the GC could trace any member of a dom_struct, therefore, never hold a borrow when making a call to anything with a `CanGC` argument.  
components/script — README

**Purpose**
- `components/script` contains the implementation of the DOM and Web API
  bindings used by Servo's JavaScript engine layer.

**Structure overview**
- dom/ — DOM types and implementations (each WebIDL interface maps to a
  Rust `#[dom_struct]` type and a generated `*Methods` trait).
- bindings/ — the part of generated bindings glue code and helpers that hasn't been moved yet to `components/script_bindings/`.
- Various files, like the WebIDL files and the config files, are found in the top-level partner component `components/script_bindings/webidls`.

**Working on a Web API (tips)**
1. Find the WebIDL in `components/script_bindings/webidls/` (or add one if
   implementing a new API).
   - If your method can throw exceptions at runtime, mark it with `[Throws]` in the WebIDL.  The Servo bindings generator will then produce an implementation signature that returns `ErrorResult` (for void returns) or `Fallible<DomRoot<T>>` / `Result<..., Error>` (for methods that return DOM objects). Implementations should `return Err(Error::...)` instead of calling `throw_dom_exception(...)`. Example: mark `MLGraphBuilder.input` with `[Throws]` so `Input()` can return `Err(Error::Type(...))` and the binding will throw in JS.
2. Consult the spec for the API you are implementing — see the `specs/`
   directory for checked-out specs (if you don't know which spec to use,
   ask the user).
3. Use the corresponding `index.bs` file under `specs/` as authoritative
   guidance for algorithms and internal-slot definitions.
4. Document code by linking the generated method to the canonical spec anchor in the top-level doc-comment and add per-line `Step N:` comments inside the function body (see **Documenting your work** below for the exact format).
5. Add a `#[dom_struct]` with `Dom` members and implement the generated
   trait methods: start with `todo!()` bodies.
6. Sub-directories (e.g. `dom/webgpu`, `dom/xr`) often include their own
   README with subsystem-specific guidance. Always read the README chain for
   the area you're changing: start with `components/script/README.md`, then
   read any `components/script/dom/<subdir>/README.md` for subsystem rules,
   and finally consult the authoritative `specs/<spec_name>/index.bs` for
   algorithm and internal-slot details.
7. Good quality examples of implementation and documentation patterns are found in `components/script/dom/stream`.
8. Prefer top-level `use` imports for types that appear in the file (for
   example `use crate::dom::webnn::mlcontext::MLContext;`) instead of using
   fully-qualified `crate::...` paths inside signatures or bodies.
   - Use short type names in method signatures and code; add the `use` at the
     top of the file. This improves readability and makes future refactors
     and reviews simpler.

**Documenting your work:**
Follow these exact conventions so code <-> spec mapping is clear and reviewable.

- Method- & type-level doc
  - Method-level: the method's top doc-comment must contain *only* the canonical spec anchor (e.g. `/// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>`).
    - Do NOT add parenthetical notes or extra prose in top doc-comments (for example, `(internal helper)`) — these add noise and are disallowed. Keep top-level doc-comments anchor-only.

  - Functions & spec-algorithms
    - Use a **module-level free function** when you are implementing a *spec algorithm* (for example `create an MLOperand`) or a small utility that is shared by multiple generated methods and does **not** directly mutate the DOM object's internal slots. Prefer a free function when the spec algorithm is described independently of any single interface implementation.
    - Use an **impl method** on the `#[dom_struct]` type when the helper needs to access or mutate that struct's private internal slots (i.e. it logically belongs to the type and requires `self`).
    - Naming & visibility: name the function to reflect the spec algorithm (`create_an_mloperand`), keep it private by default, and move it to a shared module only if genuinely reused across components.
    - Documentation rules for functions that implement spec algorithms:
      - The function's top doc-comment must contain *only* the canonical *algorithm* anchor (for example `/// <https://webmachinelearning.github.io/webnn/#create-an-mloperand>`). Do **not** add extra prose in the top doc-comment.
      - Inside the function body annotate each implementation step with `Step N:` comments that quote the spec step verbatim, exactly as you would for generated methods.
      - IMPORTANT: if the spec's algorithm does **not** mutate the caller's internal slots, the function must **not** perform those mutations — the caller (e.g. the generated method) must run the spec steps that modify the object's internal state (this preserves 1:1 mapping between spec steps and code locations).
      - For small, internal utilities that are *not* direct implementations of a spec algorithm, use a brief one-line doc comment describing intent (no spec anchor).

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
  - For any unimplemented spec step, add a `Step N: TODO — <short reason>`
    immediately above the code or `todo!()` placeholder. Also add a second
    comment with the TODO tag including the spec anchor, e.g.
    `// TODO (spec: #api-ml-createcontext): implement ML task queuing`.
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

**Helpful files**
- `specs/` — check the spec `index.bs` files for API algorithms.
- `components/script_bindings/webidls/` — WebIDL source used by the
  generator.
- `components/script/dom/` — implementation files you will edit.
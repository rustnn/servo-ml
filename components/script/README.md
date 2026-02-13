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
2. Consult the spec for the API you are implementing — see the `specs/`
   directory for checked-out specs (if you don't know which spec to use,
   ask the user).
3. Use the corresponding `index.bs` file under `specs/` as authoritative
   guidance for algorithms and internal-slot definitions.
4. Document code by copy pasting the prose from the spec it implements. 
5. Add a `#[dom_struct]` with `Dom` members and implement the generated
   trait methods: start with `todo!()` bodies.
6. Sub-directories (e.g. `dom/webgpu`, `dom/xr`) often include their own
   README with subsystem-specific guidance.
7. Good quality examples of implementation and documentation patterns are found in `components/script/dom/stream`.

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

**General Notes**
- When implementing algorithms, prefer small, testable steps and document the
  spec text in comments as you go.
- Each piece of Rust code should be documented with the prose from the spec it implements. If the mapping between spec and implementation is not clear, add a note to explain. 
- If you add new WebIDL, the codegen will generate the trait that you must
  implement in this crate.
- As you implement a WebAPI, you can track TODO in the api specific `README.md`. 

**Helpful files**
- `specs/` — check the spec `index.bs` files for API algorithms.
- `components/script_bindings/webidls/` — WebIDL source used by the
  generator.
- `components/script/dom/` — implementation files you will edit.
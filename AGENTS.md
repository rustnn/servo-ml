Agents — quick orientation

This file gives a minimal orientation to contributors/agents working in this
repo. It only points you at component READMEs — read those for implementation
conventions and deeper guidance.

Repo structure (high level)
- components/ — Rust crates that implement browser subsystems (DOM, layout,
  Web APIs, etc.). Check the README inside each component for specifics.
- third_party/, tools/, docs/, etc. — supporting materials and tooling.

Where to look next
- For DOM / Web API work: see `components/script/README.md`.
- For feature-specific guidance: also open the README in the component you'll
  modify (each component directory may contain a README).
- Note: the `README.md` files form a nested architecture: ensure you read the entire chain all the way down to the lowest-level. So for example if working on webnn, read the readmes of `script`, and then `script/webnn`. Your guidance is the concentation of all relevant readmes and this agent file. 

Guidance on adding documentation

Whenever the user corrects your code, besides fixing the code, if there
is a general lesson to document, add prose to the lowest-level possible `README.md` file. 
For example, if you learn something related only to the webnn implementation, then
add docs to `components/script/dom/webnn/README.md`, but if the lesson is relevant 
to any web api, then add docs to `components/script/README.md`. You can also add `README.md` files; ensure those are reffered-to from the `README.md` file at the level above. For example: `components/script/README.md` should refer to the `README.md` files found in `components/script/dom/`.

- Always double-check your work with `./mach check`. 

That's it — find and run the component README(s) for details relevant to your task.
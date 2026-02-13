components/script/dom/webnn — implementation notes (minimal)

- Check the normative spec at `specs/webnn/index.bs` before implementing behavior.

- Implementation hint: the `MLContext` exposes a `timeline` concept in the
  spec. In this codebase model the timeline should be implemented as steps
  enqueued to a backend/task queue (i.e. operations scheduled to run on a
  worker thread or device command queue). This is a work-in-progress;
  for now keep API implementations minimal and follow the spec algorithms
  when fleshing out the timeline behaviour. Just add TODO notes in the code for the actual messaging with the backend. 

  The implementation will follow the below high-level plan:

  1. Implement most of the API surfarce here in script.
  2. Adding backend functionality for the timeline concept
  3. Integrating the two via messaging using the `GenericCallBack` and other mechanism. WebGPU and the Indexedbb factory and connection opening code are good examples for this part. 

  TODO: 

  - [x] Implement basic stub for MLContext and the navigator ML interface.
  - [ ] Implement `CreateContext`. 
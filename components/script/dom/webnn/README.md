components/script/dom/webnn — implementation notes (minimal)

- Check the normative spec at `specs/webnn/index.bs` (or the official
  WebNN spec) before implementing behavior.

- Implementation hint: the `MLContext` exposes a `timeline` concept in the
  spec. In this codebase model the timeline should be implemented as steps
  enqueued to a backend/task queue (i.e. operations scheduled to run on a
  worker thread or device command queue). This is a work-in-progress;
  for now keep API implementations minimal and follow the spec algorithms
  when fleshing out the timeline behaviour.
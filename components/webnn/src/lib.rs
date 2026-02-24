use std::collections::{HashMap, VecDeque};
use std::hash::Hasher;
use std::path::PathBuf;
use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use crossbeam::channel::{self, Receiver, Sender};
use log::{debug, warn};
use profile_traits::generic_callback::GenericCallback;
use rustnn::GraphConverter;
use rustnn::executors::coreml::prepare_compiled_model_with_weights;
// Required to materialize constant operands during dispatch.
use rustnn::graph::ConstantData;
use webnn_traits::{ContextId, ContextMessage, GraphId, WebNNMsg};

#[derive(Debug)]
/// A single operation that may be deferred on a context timeline.
enum PendingOp {
    CreateTensor(GenericCallback<ContextMessage>, ContextId, u32, usize),
    /// Allocate a tensor and initialize it with a provided byte vector.
    /// This mirrors the `WebNNMsg::CreateConstantTensor` message and is used
    /// by graph builders when they materialize a constant operand from a
    /// host buffer.
    CreateConstantTensor(ContextId, u32, Vec<u8>),
    ReadTensor(GenericCallback<ContextMessage>, ContextId, u32),
    WriteTensor(ContextId, u32, Vec<u8>),
    Dispatch(
        ContextId,
        GraphId, // graph id key used to look up cached GraphInfo
        HashMap<u32, u32>,
        HashMap<u32, u32>,
    ),
    /// A compile step for a given cache key.  The timeline holds this to
    /// prevent any subsequent operations from running until the compilation
    /// has completed (the `Compiled` message will clear `queue_blocked`).
    Compile(GraphId),
}

struct ModelCacheEntry {
    /// raw converted model bytes; used for dispatch and (in future) hashing
    model_bytes: Vec<u8>,
    weights: Option<Vec<u8>>,
    /// path to compiled model directory once compilation has finished
    compiled_path: Option<PathBuf>,
    /// original GraphInfo from the build/compile step.  Stored here so
    /// the compute path can look it up using just the graph id.
    graph_info: rustnn::graph::GraphInfo,
}

struct Context {
    // Backend-specific context state.
    tensor_store: HashMap<u32, Vec<u8>>,
    // Sender for offloading ML work to the dedicated thread.
    compute_tx: Sender<MlMsg>,

    /// Cache of previously-seen graphs keyed by the `GraphId` provided by
    /// script.  The entry contains the compiled path when available as well
    /// as the original GraphInfo needed for dispatch.
    model_cache: HashMap<GraphId, ModelCacheEntry>,

    /// When the script requests a compilation via `MLGraphBuilder.build()` we
    /// record the `GraphId` it generated together with the persistent
    /// callback supplied by the caller.  After the ML worker signals that a
    /// particular graph id has finished compiling we use this map to notify every
    /// waiting build (and correspondingly resolve the promise on the script
    /// side).
    script_build_request: HashMap<GraphId, Vec<(GraphId, GenericCallback<ContextMessage>)>>,

    // Implementation of the WebNN context "timeline".  When a compute or
    // compile is in-flight we must hold subsequent operations and replay them
    // after the work completes.  The queue holds operations that arrived while
    // `queue_blocked` was true.
    timeline: VecDeque<PendingOp>,
    queue_blocked: bool,
}

/// State handled by the manager thread.  Encapsulating the mutable maps
/// inside a struct lets us move the large match arms into helpers and
/// dramatically reduce nesting in `run_manager`.
struct WebNNManager {
    contexts: HashMap<ContextId, Context>,
    // channel used for compute work; cloned into each context
    ml_sender: Sender<MlMsg>,
}

impl WebNNManager {
    fn new(ml_sender: Sender<MlMsg>) -> Self {
        WebNNManager {
            contexts: HashMap::new(),
            ml_sender,
        }
    }

    fn run(&mut self, receiver: GenericReceiver<WebNNMsg>) {
        debug!("webnn manager started");
        while let Ok(msg) = receiver.recv() {
            if !self.handle_message(msg) {
                break;
            }
        }
        debug!("webnn manager stopped");
    }

    fn handle_message(&mut self, msg: WebNNMsg) -> bool {
        match msg {
            WebNNMsg::Exit => {
                debug!("webnn manager exiting");
                // notify ml thread so it can exit as well
                if let Err(e) = self.ml_sender.send(MlMsg::Exit) {
                    warn!("webnn manager: failed to send ML exit: {:?}", e);
                }
                false
            },
            WebNNMsg::NewContext(id) => {
                debug!("webnn manager: NewContext {:?}", id);
                self.contexts
                    .insert(id, Context::new(self.ml_sender.clone()));
                true
            },
            WebNNMsg::DestroyContext(id) => {
                debug!("webnn manager: DestroyContext {:?}", id);
                // dropping the context automatically discards its tensor store
                self.contexts.remove(&id);
                true
            },
            WebNNMsg::CreateTensor(callback, ctx_id, tensor_id, byte_length) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.enqueue_or_run(PendingOp::CreateTensor(
                        callback,
                        ctx_id,
                        tensor_id,
                        byte_length,
                    ));
                } else {
                    warn!(
                        "webnn manager: CreateTensor for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
            WebNNMsg::CreateConstantTensor(ctx_id, tensor_id, bytes) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.enqueue_or_run(PendingOp::CreateConstantTensor(ctx_id, tensor_id, bytes));
                } else {
                    warn!(
                        "webnn manager: CreateConstantTensor for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
            WebNNMsg::ReadTensor(callback, ctx_id, tensor_id) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.enqueue_or_run(PendingOp::ReadTensor(callback, ctx_id, tensor_id));
                } else {
                    warn!("webnn manager: ReadTensor for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::WriteTensor(ctx_id, tensor_id, bytes) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.enqueue_or_run(PendingOp::WriteTensor(ctx_id, tensor_id, bytes));
                } else {
                    warn!(
                        "webnn manager: WriteTensor for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
            WebNNMsg::Dispatch(ctx_id, graph_id, inputs_map, outputs_map) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_dispatch(ctx_id, graph_id, inputs_map, outputs_map);
                } else {
                    warn!("webnn manager: Dispatch for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::Compile(cb, graph_id, ctx_id, graph_info) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    // register graph (possibly compiling or caching) and record callback
                    let key =
                        ctx.get_or_compile(ctx_id, graph_id, graph_info, Some((graph_id, cb)));
                    // put a compile step on the timeline so subsequent ops wait
                    ctx.enqueue_or_run(PendingOp::Compile(key));
                } else {
                    warn!("webnn manager: Compile for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::Compiled(ctx_id, graph_id, path) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    if let Some(entry) = ctx.model_cache.get_mut(&graph_id) {
                        entry.compiled_path = Some(path.clone());
                    }
                    // notify any build callbacks waiting for this graph_id
                    if let Some(vec) = ctx.script_build_request.remove(&graph_id) {
                        for (bid, cb) in vec {
                            // send result back to script thread, including the
                            // GraphInfo that we still have cached in the model
                            // entry.  This frees the script from having to keep a copy.
                            if let Some(entry) = ctx.model_cache.get(&graph_id) {
                                let _ = cb.send(ContextMessage::CompileResult(
                                    ctx_id,
                                    graph_id,
                                    entry.graph_info.clone(),
                                ));
                            } else {
                                // fallback: should never happen
                                let _ = cb.send(ContextMessage::CompileResult(
                                    ctx_id,
                                    graph_id,
                                    rustnn::graph::GraphInfo {
                                        operands: Vec::new(),
                                        input_operands: Vec::new(),
                                        output_operands: Vec::new(),
                                        operations: Vec::new(),
                                        constant_operand_ids_to_handles: HashMap::new(),
                                        id_to_constant_tensor_operand_map: HashMap::new(),
                                        quantized: false,
                                    },
                                ));
                            }
                        }
                    }
                    // compilation counts as an in-flight operation; clear flag
                    if ctx.queue_blocked {
                        ctx.queue_blocked = false;
                        ctx.process_queue();
                    }
                } else {
                    warn!("webnn manager: Compiled for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::CompileFailed(ctx_id, key, reason) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    warn!(
                        "webnn manager: compilation failed for {:?}/{}: {}",
                        ctx_id, key, reason
                    );
                    // drop any pending build callbacks; we still call them so the
                    // script side can resolve the promises, though they won't
                    // be able to dispatch successfully later.
                    if let Some(vec) = ctx.script_build_request.remove(&key) {
                        for (_build_id, cb) in vec {
                            let _ = cb.send(ContextMessage::CompileResult(
                                ctx_id,
                                key,
                                rustnn::graph::GraphInfo {
                                    operands: Vec::new(),
                                    input_operands: Vec::new(),
                                    output_operands: Vec::new(),
                                    operations: Vec::new(),
                                    constant_operand_ids_to_handles: HashMap::new(),
                                    id_to_constant_tensor_operand_map: HashMap::new(),
                                    quantized: false,
                                },
                            ));
                        }
                    }
                    if ctx.queue_blocked {
                        ctx.queue_blocked = false;
                        ctx.process_queue();
                    }
                } else {
                    warn!(
                        "webnn manager: CompileFailed for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
            WebNNMsg::ComputeResult(ctx_id, outputs) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_compute_result(outputs);
                } else {
                    warn!(
                        "webnn manager: ComputeResult for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
        }
    }
}

// Methods that used to live on the manager have been pushed down into each
// context instance.  The manager now merely looks up the appropriate context
// and forwards messages.
impl Context {
    fn new(compute_tx: Sender<MlMsg>) -> Self {
        Context {
            tensor_store: HashMap::new(),
            compute_tx,
            model_cache: HashMap::new(),
            script_build_request: HashMap::new(),
            timeline: VecDeque::new(),
            queue_blocked: false,
        }
    }

    fn handle_create_tensor(
        &mut self,
        callback: GenericCallback<ContextMessage>,
        ctx_id: ContextId,
        tensor_id: u32,
        byte_length: usize,
    ) {
        debug!(
            "webnn manager: CreateTensor ctx={:?} id={} len={}",
            ctx_id, tensor_id, byte_length
        );
        let mut buffer: Vec<u8> = Vec::with_capacity(byte_length);
        let n_f32 = byte_length / 4;
        for _ in 0..n_f32 {
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
        }
        if byte_length % 4 != 0 {
            buffer.extend(std::iter::repeat(0u8).take(byte_length % 4));
        }
        self.tensor_store.insert(tensor_id, buffer);
        if let Err(e) = callback.send(ContextMessage::CreateTensorResult(
            ctx_id,
            tensor_id,
            Ok(()),
        )) {
            warn!("webnn manager: CreateTensor callback send failed: {:?}", e);
        }
    }

    fn handle_create_constant_tensor(&mut self, ctx_id: ContextId, tensor_id: u32, bytes: Vec<u8>) {
        debug!(
            "webnn manager: CreateConstantTensor ctx={:?} id={} len={}",
            ctx_id,
            tensor_id,
            bytes.len()
        );
        // create the buffer exactly as given; no zeroing required
        self.tensor_store.insert(tensor_id, bytes);
        // unlike CreateTensor there is no callback path; callers either
        // resolve their own promise synchronously or ignore the result.
    }

    fn handle_read_tensor(
        &mut self,
        callback: GenericCallback<ContextMessage>,
        ctx_id: ContextId,
        tensor_id: u32,
    ) {
        debug!(
            "webnn manager: ReadTensor ctx={:?} id={}",
            ctx_id, tensor_id
        );
        match self.tensor_store.get(&tensor_id) {
            Some(buf) => {
                if let Err(e) = callback.send(ContextMessage::ReadTensorResult(
                    ctx_id,
                    tensor_id,
                    Ok(buf.clone()),
                )) {
                    warn!("webnn manager: ReadTensor callback send failed: {:?}", e);
                }
            },
            None => {
                warn!(
                    "webnn manager: ReadTensor - missing buffer for {:?}/{}",
                    ctx_id, tensor_id
                );
                if let Err(e) =
                    callback.send(ContextMessage::ReadTensorResult(ctx_id, tensor_id, Err(())))
                {
                    warn!(
                        "webnn manager: ReadTensor error callback send failed: {:?}",
                        e
                    );
                }
            },
        }
    }

    fn handle_write_tensor(&mut self, ctx_id: ContextId, tensor_id: u32, bytes: Vec<u8>) {
        debug!(
            "webnn manager: WriteTensor ctx={:?} id={} len={}",
            ctx_id,
            tensor_id,
            bytes.len()
        );
        self.tensor_store.insert(tensor_id, bytes);
    }

    fn handle_dispatch(
        &mut self,
        ctx_id: ContextId,
        key: GraphId,
        inputs_map: Vec<(u32, u32)>,
        outputs_map: Vec<(u32, u32)>,
    ) {
        // Convert the flat vectors into maps before creating the queued
        // operation, matching the old behaviour.  (Doing this once now keeps
        // the `PendingOp` simple.)
        let inputs_map: HashMap<u32, u32> = inputs_map.into_iter().collect();
        let outputs_map: HashMap<u32, u32> = outputs_map.into_iter().collect();

        self.enqueue_or_run(PendingOp::Dispatch(ctx_id, key, inputs_map, outputs_map));
    }

    fn collect_input_bytes(&self, inputs_map: &HashMap<u32, u32>) -> HashMap<u32, Vec<u8>> {
        let mut inputs_bytes = HashMap::new();
        for (op_id, tensor_id) in inputs_map {
            if let Some(buf) = self.tensor_store.get(tensor_id) {
                inputs_bytes.insert(*op_id, buf.clone());
            }
        }
        inputs_bytes
    }

    fn resolve_constant_operands(&mut self, graph_info: &mut rustnn::graph::GraphInfo) {
        if graph_info.id_to_constant_tensor_operand_map.is_empty() {
            return;
        }
        for (op_id, tensor_id_str) in graph_info.id_to_constant_tensor_operand_map.iter() {
            if let Ok(tid) = tensor_id_str.parse::<u32>() {
                if let Some(buf) = self.tensor_store.get(&tid) {
                    graph_info.constant_operand_ids_to_handles.insert(
                        *op_id,
                        ConstantData {
                            data: buf.clone(),
                            label: None,
                        },
                    );
                }
            }
        }
    }

    // NOTE: the CoreML path has been extracted into a free function so that
    // it can be invoked from multiple entry points (e.g. the new `compute`
    // helper).  `compute` itself lives just below the helper declarations.

    /// Convenience helper used from `handle_dispatch` to run the graph and
    /// populate `tensor_store`.  It mirrors the former body of
    /// `handle_dispatch` but delegates the CoreML attempt to a free function.
    /// Run a graph compute on the ML thread.  Returns true if the compute
    /// was successfully dispatched and `queue_blocked` remains true.  A
    /// return value of `false` indicates we fell back synchronously (e.g. the
    /// ML channel was closed), in which case callers should continue draining
    /// the timeline immediately.
    /// Ensure the graph is compiled and return its cache key.
    ///
    /// This may queue a compilation on the ML thread if no entry exists yet.
    /// Compute/cache a key for `graph_info`, enqueue a compile on the
    /// ML thread if we haven't seen it before, and optionally record a script-side
    /// callback that should be invoked when the compilation completes.
    fn get_or_compile(
        &mut self,
        ctx_id: ContextId,
        graph_id: GraphId,
        graph_info: rustnn::graph::GraphInfo,
        build_request: Option<(GraphId, GenericCallback<ContextMessage>)>,
    ) -> GraphId {
        debug!("get_or_compile: ctx={:?}", ctx_id);
        // make a mutable copy so we can resolve constants without warning
        let mut gi = graph_info;
        // resolve any constant operands before conversion: this ensures the
        // compiled model includes them and we don't need to mutate the graph
        // on the compute path.
        self.resolve_constant_operands(&mut gi);
        use rustnn::converters::CoremlMlProgramConverter;
        // convert once; the resulting bytes are used by the backend for
        // compilation and dispatch.  The script-supplied `graph_id` acts as the
        // cache key instead of hashing the bytes.
        let converter = CoremlMlProgramConverter;
        if let Ok(converted) = converter.convert(&gi) {
            debug!(
                "get_or_compile: conversion succeeded for graph {:?}",
                graph_id
            );

            // record any build callback so we can notify when this graph
            // finishes compiling.
            if let Some((gid, cb)) = build_request {
                debug!(
                    "get_or_compile: recording build_request for graph {:?}",
                    gid
                );
                self.script_build_request
                    .entry(gid)
                    .or_default()
                    .push((gid, cb));
            }

            if !self.model_cache.contains_key(&graph_id) {
                debug!(
                    "get_or_compile: inserting new graph {:?} into cache",
                    graph_id
                );
                self.model_cache.insert(
                    graph_id,
                    ModelCacheEntry {
                        model_bytes: converted.data.clone(),
                        weights: converted.weights_data.clone(),
                        compiled_path: None,
                        graph_info: gi.clone(),
                    },
                );
                // send compile request to ML thread
                if let Err(e) = self.compute_tx.send(MlMsg::Compile {
                    ctx_id,
                    key: graph_id,
                    model_bytes: converted.data,
                    weights: converted.weights_data,
                    graph_info: gi.clone(),
                }) {
                    warn!("webnn manager: failed to send compile message: {:?}", e);
                }
            } else {
                debug!(
                    "get_or_compile: graph {:?} already present in cache",
                    graph_id
                );
            }
            graph_id
        } else {
            warn!("get_or_compile: conversion failed");
            graph_id
        }
    }

    fn compute(
        &mut self,
        _ctx_id: ContextId,
        key: GraphId,
        inputs_map: HashMap<u32, u32>,
        outputs_map: HashMap<u32, u32>,
        compute_tx: &Sender<MlMsg>,
    ) -> bool {
        debug!("compute: ctx={:?}", _ctx_id);

        // Look up the matching GraphInfo from our cache.  Dispatch no longer
        // sends the full graph, so we must retrieve it here using the previously
        // stored entry.  If it's missing we can't reasonably execute, so return
        // early with zeroed outputs.
        let mut graph_info = if let Some(entry) = self.model_cache.get(&key) {
            entry.graph_info.clone()
        } else {
            warn!("compute: missing graph_info for graph {:?}", key);
            self.zeroed_outputs(
                &rustnn::graph::GraphInfo {
                    operands: Vec::new(),
                    input_operands: Vec::new(),
                    output_operands: Vec::new(),
                    operations: Vec::new(),
                    constant_operand_ids_to_handles: HashMap::new(),
                    id_to_constant_tensor_operand_map: HashMap::new(),
                    quantized: false,
                },
                &outputs_map,
            );
            self.queue_blocked = false;
            return false;
        };

        // Materialize any constant operands now that we have the graph info.
        self.resolve_constant_operands(&mut graph_info);

        let inputs_bytes = self.collect_input_bytes(&inputs_map);
        debug!("compute: collected {} input buffers", inputs_bytes.len());

        // Ensure the graph has been compiled (a no-op if already cached).
        let _ = self.get_or_compile(_ctx_id, key, graph_info.clone(), None);
        debug!("compute: using graph {:?}", key);

        let work_graph = graph_info.clone();
        let work_outputs = outputs_map.clone();

        let msg = MlMsg::Compute {
            ctx_id: _ctx_id,
            key,
            inputs_map,
            inputs_bytes,
            outputs_map: work_outputs,
        };

        if let Err(e) = compute_tx.send(msg) {
            warn!("webnn manager: failed to send compute message: {:?}", e);
            self.zeroed_outputs(&work_graph, &outputs_map);
            self.queue_blocked = false;
            return false;
        }

        debug!("compute: message sent successfully");
        true
    }

    fn zeroed_outputs(
        &mut self,
        graph_info: &rustnn::graph::GraphInfo,
        outputs_map: &HashMap<u32, u32>,
    ) {
        for (op_id, tensor_id) in outputs_map.iter() {
            if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                let element_count: usize =
                    operand.descriptor.shape.iter().fold(1usize, |acc, d| {
                        acc.saturating_mul(rustnn::graph::get_static_or_max_size(d) as usize)
                    });
                let byte_len =
                    element_count.saturating_mul(operand.descriptor.data_type.bytes_per_element());
                self.tensor_store.insert(*tensor_id, vec![0u8; byte_len]);
            }
        }
    }

    // timeline helpers
    fn enqueue_or_run(&mut self, op: PendingOp) {
        if self.queue_blocked {
            self.timeline.push_back(op);
        } else {
            self.run_now(op);
        }
    }

    fn run_now(&mut self, op: PendingOp) {
        debug!("run_now: op={:?}", op);
        match op {
            PendingOp::CreateTensor(cb, ctx_id, tensor_id, len) => {
                debug!("run_now: CreateTensor");
                self.handle_create_tensor(cb, ctx_id, tensor_id, len);
            },
            PendingOp::CreateConstantTensor(ctx_id, tensor_id, bytes) => {
                debug!("run_now: CreateConstantTensor");
                self.handle_create_constant_tensor(ctx_id, tensor_id, bytes);
            },
            PendingOp::ReadTensor(cb, ctx_id, tensor_id) => {
                debug!("run_now: ReadTensor");
                self.handle_read_tensor(cb, ctx_id, tensor_id);
            },
            PendingOp::WriteTensor(ctx_id, tensor_id, bytes) => {
                debug!("run_now: WriteTensor");
                self.handle_write_tensor(ctx_id, tensor_id, bytes);
            },
            PendingOp::Dispatch(ctx_id, key, inputs_map, outputs_map) => {
                debug!("run_now: Dispatch");
                self.queue_blocked = true;
                let compute_chan = self.compute_tx.clone();
                let started = self.compute(ctx_id, key, inputs_map, outputs_map, &compute_chan);
                if !started {
                    debug!("run_now: compute failed to start, clearing queue_blocked");
                    self.queue_blocked = false;
                }
            },
            PendingOp::Compile(key) => {
                debug!("run_now: Compile graph={:?}", key);
                // block the timeline until the compilation result arrives.
                self.queue_blocked = true;
                if let Some(entry) = self.model_cache.get(&key) {
                    if entry.compiled_path.is_some() {
                        // already done; we can immediately resume
                        debug!("run_now: compile already done, unblocking");
                        self.queue_blocked = false;
                    }
                }
            },
        }
    }

    fn handle_compute_result(&mut self, outputs: HashMap<u32, Vec<u8>>) {
        debug!("handle_compute_result: received {} tensors", outputs.len());
        for (tensor_id, bytes) in outputs {
            debug!(
                "handle_compute_result: inserting tensor {} len={}",
                tensor_id,
                bytes.len()
            );
            self.tensor_store.insert(tensor_id, bytes);
        }
        self.queue_blocked = false;
        self.process_queue();
    }

    fn process_queue(&mut self) {
        debug!(
            "process_queue: starting, queue_blocked={} queue_len={}",
            self.queue_blocked,
            self.timeline.len()
        );
        while !self.queue_blocked {
            if let Some(op) = self.timeline.pop_front() {
                debug!("process_queue: running op {:?}", op);
                self.run_now(op);
                // loop again unless a dispatch set queue_blocked true
            } else {
                debug!("process_queue: queue empty");
                break;
            }
        }
    }
}

/// Create a new WebNN manager and return the `GenericSender<WebNNMsg>`
/// together with the `JoinHandle` for the manager thread.
///
/// Returning the join handle allows the caller (the `Constellation`) to
/// join the manager thread during shutdown.
pub fn new_webnn_manager() -> (GenericSender<WebNNMsg>, std::thread::JoinHandle<()>) {
    let (tx, rx): (GenericSender<WebNNMsg>, GenericReceiver<WebNNMsg>) =
        channel().expect("webnn channel");

    // we keep a clone of the sender so the ML thread can emit ComputeResult
    // back to the manager.
    let manager_tx = tx.clone();

    let handle = thread::Builder::new()
        .name("WebNNManager".into())
        .spawn(move || run_manager(rx, manager_tx))
        .expect("failed to spawn WebNN manager thread");

    (tx, handle)
}

fn run_manager(receiver: GenericReceiver<WebNNMsg>, manager_tx: GenericSender<WebNNMsg>) {
    debug!("run_manager: starting");
    // create dedicated channel for ML work
    let (ml_tx, ml_rx): (Sender<MlMsg>, Receiver<MlMsg>) = channel::unbounded();

    // spawn the ML worker thread; give it a copy of `manager_tx` so it
    // can post the ComputeResult message when work finishes.
    let ml_handle = thread::Builder::new()
        .name("WebNNML".into())
        .spawn(move || ml_loop(ml_rx, manager_tx.clone()))
        .expect("failed to spawn WebNN ML thread");

    // manager now owns a sender that it will clone into each context
    let mut manager = WebNNManager::new(ml_tx.clone());
    manager.run(receiver);

    debug!("run_manager: manager.run returned");
    // manager.run has returned (likely due to Exit).  make sure the ML thread
    // is told to exit in case we didn't already send the message above.
    if let Err(e) = ml_tx.send(MlMsg::Exit) {
        warn!("webnn manager: failed to notify ML thread of exit: {:?}", e);
    }

    // we want to join the ML thread but only block for a short time.
    // spawning a small helper thread lets us use a channel's recv_timeout
    // rather than busy‑polling.
    let (done_tx, done_rx) = channel::bounded(1);
    let joiner = thread::spawn(move || {
        if let Err(e) = ml_handle.join() {
            warn!("webnn manager: ML thread join panicked: {:?}", e);
        }
        if let Err(e) = done_tx.send(()) {
            warn!("webnn manager: failed to signal join completion: {:?}", e);
        }
    });

    if let Err(e) = done_rx.recv_timeout(std::time::Duration::from_millis(100)) {
        warn!("webnn manager: ML join helper timeout or error: {:?}", e);
    }
    // if the helper is still alive we don't care; dropping its handle will
    // detach it.  otherwise, join it to clean up.
    if let Err(e) = joiner.join() {
        warn!("webnn manager: join helper thread panicked: {:?}", e);
    }
}

// messages sent to the ml worker thread.  compute requests carry a
// copy of all data necessary to run CoreML (or compute zeroed outputs).  The
// worker no longer returns results directly; instead it sends a
// `WebNNMsg::ComputeResult` back to the manager thread.
#[derive(Debug)]
enum MlMsg {
    Compute {
        ctx_id: ContextId,
        key: GraphId,
        inputs_map: HashMap<u32, u32>,
        inputs_bytes: HashMap<u32, Vec<u8>>,
        outputs_map: HashMap<u32, u32>,
    },
    Compile {
        ctx_id: ContextId,
        key: GraphId,
        model_bytes: Vec<u8>,
        weights: Option<Vec<u8>>,
        graph_info: rustnn::graph::GraphInfo,
    },
    Exit,
}

// Worker loop run on the dedicated ML thread.  It waits for compute messages,
// executes `try_coreml_execute` (or generates zeroed outputs if that fails),
// and then sends the resulting tensor bytes back to the caller.
fn ml_loop(rx: Receiver<MlMsg>, manager_tx: GenericSender<WebNNMsg>) {
    debug!("ml_loop: starting");
    // map from graph id -> (cached compiled model directory, original GraphInfo)
    let mut compiled_map: HashMap<GraphId, (PathBuf, rustnn::graph::GraphInfo)> = HashMap::new();
    while let Ok(msg) = rx.recv() {
        debug!("ml_loop: received message {:?}", msg);
        match msg {
            MlMsg::Compute {
                ctx_id,
                key,
                inputs_map,
                inputs_bytes,
                outputs_map,
            } => {
                debug!("ml_loop: Compute ctx={:?} graph={:?}", ctx_id, key);
                let mut outputs = HashMap::new();
                // retrieve the stored graph_info and path from the cache
                let (cached_path, mut graph_info) = if let Some((path, gi)) = compiled_map.get(&key)
                {
                    (Some(path.as_path()), gi.clone())
                } else {
                    // missing entry; produce zeroed outputs and skip CoreML
                    for (op_id, tensor_id) in outputs_map.iter() {
                        // can't inspect operand shape without graph_info; just zero 4-byte elements
                        outputs.insert(*tensor_id, Vec::new());
                    }
                    if let Err(e) = manager_tx.send(WebNNMsg::ComputeResult(ctx_id, outputs)) {
                        warn!("webnn ML thread: failed to send ComputeResult: {:?}", e);
                    }
                    continue;
                };
                if !try_coreml_execute(
                    &mut graph_info,
                    &inputs_map,
                    &inputs_bytes,
                    &outputs_map,
                    cached_path,
                    &mut outputs,
                ) {
                    debug!("ml_loop: coreml execution failed, zeroing outputs");
                    // coreml either not available or failed, generate zeros
                    for (op_id, tensor_id) in outputs_map.iter() {
                        if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                            let element_count: usize =
                                operand.descriptor.shape.iter().fold(1usize, |acc, d| {
                                    acc.saturating_mul(
                                        rustnn::graph::get_static_or_max_size(d) as usize
                                    )
                                });
                            let byte_len = element_count
                                .saturating_mul(operand.descriptor.data_type.bytes_per_element());
                            outputs.insert(*tensor_id, vec![0u8; byte_len]);
                        }
                    }
                } else {
                    debug!("ml_loop: coreml execution succeeded");
                }
                // send the outputs back to the manager instead of using the
                // one‑shot response channel.
                if let Err(e) = manager_tx.send(WebNNMsg::ComputeResult(ctx_id, outputs)) {
                    warn!("webnn ML thread: failed to send ComputeResult: {:?}", e);
                }
            },
            MlMsg::Compile {
                ctx_id,
                key,
                model_bytes,
                weights,
                graph_info,
            } => {
                debug!("ml_loop: Compile ctx={:?} graph={:?}", ctx_id, key);
                // compile the model and record the returned path
                // we no longer need to pre‑allocate a cache directory; the
                // helper will give us the location of the compiled model.
                // sanity‑check the model bytes in debug builds; an empty buffer
                // would trip the CoreML FFI and is almost certainly a bug.
                debug_assert!(!model_bytes.is_empty(), "compile: empty model bytes");

                // The `prepare_compiled_model_with_weights` helper is marked
                // `unsafe` because it crosses the CoreML FFI boundary.  Exercise
                // extra care by catching panics so a corrupted model can't abort
                // the ML thread and bring down the browser process.
                let compile_result = std::panic::catch_unwind(|| unsafe {
                    prepare_compiled_model_with_weights(&model_bytes, weights.as_deref(), None)
                });

                match compile_result {
                    Ok(Ok((_compiled_url, compiled_path, _temp_mlmodel))) => {
                        compiled_map.insert(key, (compiled_path.clone(), graph_info));
                        // notify manager so the context can mark the entry ready
                        if let Err(e) =
                            manager_tx.send(WebNNMsg::Compiled(ctx_id, key, compiled_path))
                        {
                            warn!("webnn ML thread: failed to send Compiled: {:?}", e);
                        }
                    },
                    Ok(Err(e)) => {
                        warn!("webnn ML thread: compile failed for key {}: {:?}", key, e);
                        if let Err(e) = manager_tx.send(WebNNMsg::CompileFailed(
                            ctx_id,
                            key,
                            format!("{:?}", e),
                        )) {
                            warn!("webnn ML thread: failed to send CompileFailed: {:?}", e);
                        }
                    },
                    Err(panic_payload) => {
                        // Conversion from panic payload to string is best-effort.
                        let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        warn!("webnn ML thread: compile panicked for key {}: {}", key, msg);
                        if let Err(e) = manager_tx.send(WebNNMsg::CompileFailed(
                            ctx_id,
                            key,
                            format!("panic: {}", msg),
                        )) {
                            warn!("webnn ML thread: failed to send CompileFailed: {:?}", e);
                        }
                    },
                }
            },
            MlMsg::Exit => {
                debug!("ml_loop: Exit received, breaking");
                break;
            },
        }
    }
    debug!("ml_loop: exiting");
}

// Common implementation for attempting a CoreML execution.  If it succeeds
// the `outputs_store` map is populated and `true` is returned.  In the
// `#[cfg(not(target_os = "macos"))]` case we simply return false.

#[cfg(target_os = "macos")]
fn try_coreml_execute(
    graph_info: &mut rustnn::graph::GraphInfo,
    inputs_map: &HashMap<u32, u32>,
    inputs_bytes: &HashMap<u32, Vec<u8>>,
    outputs_map: &HashMap<u32, u32>,
    cached_compiled: Option<&std::path::Path>,
    outputs_store: &mut HashMap<u32, Vec<u8>>,
) -> bool {
    debug!("try_coreml_execute: entry");
    use rustnn::GraphConverter;
    use rustnn::converters::CoremlMlProgramConverter;
    use rustnn::executors::coreml::{
        CoremlInput, run_coreml_with_inputs_cached, run_coreml_with_inputs_with_weights,
    };

    debug!("webnn manager: Dispatch — attempting CoreML execution");

    let mut coreml_inputs: Vec<CoremlInput> = Vec::new();
    for (op_id, _) in inputs_map.iter() {
        if let Some(op) = graph_info.operands.get(*op_id as usize) {
            let default_name = format!("input_{}", op_id);
            let input_name = op.name.as_deref().unwrap_or(&default_name).to_string();

            if let Some(buf) = inputs_bytes.get(op_id) {
                let data: Vec<f32> = match op.descriptor.data_type {
                    rustnn::graph::DataType::Float32 => {
                        let mut out = Vec::with_capacity(buf.len() / 4);
                        let mut i = 0usize;
                        while i + 4 <= buf.len() {
                            let mut b = [0u8; 4];
                            b.copy_from_slice(&buf[i..i + 4]);
                            out.push(f32::from_le_bytes(b));
                            i += 4;
                        }
                        out
                    },
                    rustnn::graph::DataType::Float16 => {
                        let mut out = Vec::with_capacity(buf.len() / 2);
                        let mut i = 0usize;
                        while i + 2 <= buf.len() {
                            let mut b = [0u8; 2];
                            b.copy_from_slice(&buf[i..i + 2]);
                            let bits = u16::from_le_bytes(b);
                            out.push(half::f16::from_bits(bits).to_f32());
                            i += 2;
                        }
                        out
                    },
                    _other => Vec::new(),
                };

                if !data.is_empty() {
                    let shape: Vec<usize> = op
                        .descriptor
                        .shape
                        .iter()
                        .map(|d| rustnn::graph::get_static_or_max_size(d) as usize)
                        .collect();
                    coreml_inputs.push(CoremlInput {
                        name: input_name,
                        shape,
                        data,
                    });
                }
            }
        }
    }

    let converter = CoremlMlProgramConverter;
    if let Ok(converted) = converter.convert(graph_info) {
        debug!("try_coreml_execute: conversion succeeded");
        // at this point the graph should already have been compiled on a
        // previous path; ensure we have a cached model directory.  If this
        // assertion ever trips in release it indicates a programming error
        // where dispatch was allowed to run without compilation.
        debug_assert!(
            cached_compiled.is_some(),
            "dispatch without cached compiled model"
        );
        let path = cached_compiled.expect("cached model path");
        debug!(
            "try_coreml_execute: running cached compiled model at {:?}",
            path
        );
        let run_result = run_coreml_with_inputs_cached(&converted.data, coreml_inputs, Some(path));
        if let Ok(attempts) = run_result {
            if let Some(outputs) = attempts
                .iter()
                .find_map(|a| a.result.as_ref().ok().cloned())
            {
                debug!("try_coreml_execute: got outputs from coreml");
                for (op_id, tensor_id) in outputs_map.iter() {
                    if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                        let default_name = format!("output_{}", op_id);
                        let output_name = operand.name.as_deref().unwrap_or(&default_name);

                        if let Some(coreml_out) = outputs.iter().find(|o| o.name == output_name) {
                            match operand.descriptor.data_type {
                                rustnn::graph::DataType::Float32 => {
                                    let mut bytes = Vec::with_capacity(coreml_out.data.len() * 4);
                                    for &v in coreml_out.data.iter() {
                                        bytes.extend_from_slice(&v.to_le_bytes());
                                    }
                                    outputs_store.insert(*tensor_id, bytes);
                                },
                                rustnn::graph::DataType::Float16 => {
                                    let mut bytes = Vec::with_capacity(coreml_out.data.len() * 2);
                                    for &v in coreml_out.data.iter() {
                                        let bits = half::f16::from_f32(v).to_bits();
                                        bytes.extend_from_slice(&bits.to_le_bytes());
                                    }
                                    outputs_store.insert(*tensor_id, bytes);
                                },
                                _other => {
                                    let byte_length = operand
                                        .descriptor
                                        .shape
                                        .iter()
                                        .fold(1usize, |acc, d| {
                                            acc.saturating_mul(
                                                rustnn::graph::get_static_or_max_size(d) as usize,
                                            )
                                        })
                                        .saturating_mul(4usize);
                                    outputs_store.insert(*tensor_id, vec![0u8; byte_length]);
                                },
                            }
                        } else {
                            debug!(
                                "try_coreml_execute: missing output {} from coreml",
                                output_name
                            );
                            let byte_length = operand
                                .descriptor
                                .shape
                                .iter()
                                .fold(1usize, |acc, d| {
                                    acc.saturating_mul(
                                        rustnn::graph::get_static_or_max_size(d) as usize
                                    )
                                })
                                .saturating_mul(4usize);
                            outputs_store.insert(*tensor_id, vec![0u8; byte_length]);
                        }
                    }
                }
                return true;
            }
        }
    }
    debug!("try_coreml_execute: falling back, returning false");
    false
}

#[cfg(not(target_os = "macos"))]
fn try_coreml_execute(
    _graph_info: &mut rustnn::graph::GraphInfo,
    _inputs_map: &HashMap<u32, u32>,
    _inputs_bytes: &HashMap<u32, Vec<u8>>,
    _outputs_map: &HashMap<u32, u32>,
    _outputs_store: &mut HashMap<u32, Vec<u8>>,
) -> bool {
    false
}

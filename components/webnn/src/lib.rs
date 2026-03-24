/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use servo_base::generic_channel::{channel, GenericReceiver, GenericSender};
use log::{debug, warn};
use parking_lot::{Condvar, Mutex, RwLock};
use profile_traits::generic_callback::GenericCallback;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rustnn::GraphConverter;
use rustnn::executors::coreml::{CoremlOutput, prepare_compiled_model_with_weights};
use rustnn::graph::{ConstantData, DataType, GraphInfo, Operand, get_static_or_max_size};

fn operand_byte_length(operand: &Operand) -> usize {
    operand
        .descriptor
        .shape
        .iter()
        .fold(1usize, |acc, d| {
            acc.saturating_mul(get_static_or_max_size(d) as usize)
        })
        .saturating_mul(operand.descriptor.data_type.bytes_per_element())
}

// helper for converting a CoreML output (or lack thereof) into the byte
// buffer we store in the manager's tensor store.  Handles all supported
// data types and falls back to a zeroed buffer when the output is missing.
fn process_coreml_outputs(operand: &Operand, coreml_out: Option<CoremlOutput>) -> Vec<u8> {
    if let Some(coreml_out) = coreml_out {
        match operand.descriptor.data_type {
            DataType::Float32 => {
                let mut bytes = Vec::with_capacity(coreml_out.data.len() * 4);
                for &v in coreml_out.data.iter() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            },
            DataType::Float16 => {
                let mut bytes = Vec::with_capacity(coreml_out.data.len() * 2);
                for &v in coreml_out.data.iter() {
                    let bits = half::f16::from_f32(v).to_bits();
                    bytes.extend_from_slice(&bits.to_le_bytes());
                }
                bytes
            },
            DataType::Int32 => {
                // CoreML emits a float array even for integer outputs.  In
                // practice the values appear as tiny denormals whose underlying
                // bit pattern encodes the integer we want (1e-45 == 0x0000_0001).
                // Casting to i32 would convert numerically (producing 0); instead
                // reinterpret the bits directly.
                let mut bytes = Vec::with_capacity(coreml_out.data.len() * 4);
                for &v in coreml_out.data.iter() {
                    // reinterpret the raw bits as a signed integer
                    let bits = v.to_bits();
                    let iv = i32::from_ne_bytes(bits.to_ne_bytes());
                    bytes.extend_from_slice(&iv.to_le_bytes());
                }
                bytes
            },
            DataType::Uint8 => coreml_out.data.iter().map(|&v| v as u8).collect(),
            _other => {
                let byte_length = operand_byte_length(operand);
                vec![0u8; byte_length]
            },
        }
    } else {
        // no CoreML output at all -> zero buffer
        let byte_length = operand_byte_length(operand);
        vec![0u8; byte_length]
    }
}

use webnn_traits::{ContextId, ContextMessage, GraphId, WebNNMsg};

#[derive(Debug)]
/// A single operation that may be deferred on a context timeline.
enum PendingOp {
    CreateTensor(GenericCallback<ContextMessage>, ContextId, u32, usize),
    CreateConstantTensor(ContextId, u32, Vec<u8>),
    ReadTensor(GenericCallback<ContextMessage>, ContextId, u32),
    WriteTensor(ContextId, u32, Vec<u8>),
    Dispatch(ContextId, GraphId, HashMap<u32, u32>, HashMap<u32, u32>),
    Compile(ContextId, GraphId, GraphInfo),
}

struct Context {
    // Backend-specific context state.
    tensor_store: HashMap<u32, Arc<Vec<u8>>>,
    pool: Rc<ThreadPool>,
    compiled_map: SharedCompiledMap,
    manager_tx: GenericSender<WebNNMsg>,

    /// When the script requests a compilation via `MLGraphBuilder.build()` we
    /// record the `GraphId` it generated together with the persistent
    /// callback supplied by the caller.  After the ML worker signals that a
    /// particular graph id has finished compiling we use this map to notify every
    /// waiting build (and correspondingly resolve the promise on the script
    /// side).  The key already identifies the graph so we only store the
    /// callback in the value vector.
    script_build_request: HashMap<GraphId, Vec<GenericCallback<ContextMessage>>>,

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
    manager_tx: GenericSender<WebNNMsg>,
    pool: Rc<ThreadPool>,
    compiled_map: SharedCompiledMap,
}

impl WebNNManager {
    fn new(manager_tx: GenericSender<WebNNMsg>) -> Self {
        WebNNManager {
            contexts: HashMap::new(),
            manager_tx,
            pool: Rc::new(
                ThreadPoolBuilder::new()
                    .thread_name(|index| format!("WebNNMLWorker{index}"))
                    .build()
                    .expect("failed to build WebNN ML thread pool"),
            ),
            compiled_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn run(&mut self, receiver: GenericReceiver<WebNNMsg>) {
        while let Ok(msg) = receiver.recv() {
            if !self.handle_message(msg) {
                break;
            }
        }
    }

    fn handle_message(&mut self, msg: WebNNMsg) -> bool {
        match msg {
            WebNNMsg::Exit => false,
            WebNNMsg::NewContext(id) => {
                self.contexts.insert(
                    id,
                    Context::new(
                        Rc::clone(&self.pool),
                        Arc::clone(&self.compiled_map),
                        self.manager_tx.clone(),
                    ),
                );
                true
            },
            WebNNMsg::DestroyContext(id) => {
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
                    ctx.handle_compile(ctx_id, graph_id, graph_info, cb);
                } else {
                    warn!("webnn manager: Compile for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::Compiled(ctx_id, graph_id, _path) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_compiled(ctx_id, graph_id);
                } else {
                    warn!("webnn manager: Compiled for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::CompileFailed(ctx_id, key, reason) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_compile_failed(ctx_id, key, reason);
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

impl Context {
    fn new(
        pool: Rc<ThreadPool>,
        compiled_map: SharedCompiledMap,
        manager_tx: GenericSender<WebNNMsg>,
    ) -> Self {
        Context {
            tensor_store: HashMap::new(),
            pool,
            compiled_map,
            manager_tx,
            script_build_request: HashMap::new(),
            timeline: VecDeque::new(),
            queue_blocked: false,
        }
    }

    fn spawn_work(&self, work: WorkItem) {
        let manager_tx = self.manager_tx.clone();
        let compiled_map = Arc::clone(&self.compiled_map);
        self.pool.spawn(move || match work {
            WorkItem::Compute {
                ctx_id,
                key,
                inputs_map,
                inputs_bytes,
                outputs_map,
            } => handle_ml_compute(
                ctx_id,
                key,
                inputs_map,
                inputs_bytes,
                outputs_map,
                compiled_map,
                manager_tx,
            ),
            WorkItem::Compile {
                ctx_id,
                key,
                graph_info,
            } => handle_ml_compile(ctx_id, key, graph_info, compiled_map, manager_tx),
        });
    }

    fn handle_create_tensor(
        &mut self,
        callback: GenericCallback<ContextMessage>,
        ctx_id: ContextId,
        tensor_id: u32,
        byte_length: usize,
    ) {
        // The backend stores tensors as raw bytes.  All data types are
        // represented as little-endian numerical values when interpreted by
        // the consumer, but from the manager's point of view the buffer is
        // untyped.  Zeroing can therefore be done with a simple byte
        // sequence – there is no need to assume float32 or any other element
        // size.
        let buffer = vec![0u8; byte_length];
        self.tensor_store.insert(tensor_id, Arc::new(buffer));
        if let Err(e) = callback.send(ContextMessage::CreateTensorResult(
            ctx_id,
            tensor_id,
            Ok(()),
        )) {
            warn!("webnn manager: CreateTensor callback send failed: {:?}", e);
        }
    }

    fn handle_create_constant_tensor(
        &mut self,
        _ctx_id: ContextId,
        tensor_id: u32,
        bytes: Vec<u8>,
    ) {
        // create the buffer exactly as given; no zeroing required
        self.tensor_store.insert(tensor_id, Arc::new(bytes));
        // unlike CreateTensor there is no callback path; callers either
        // resolve their own promise synchronously or ignore the result.
    }

    fn handle_read_tensor(
        &mut self,
        callback: GenericCallback<ContextMessage>,
        ctx_id: ContextId,
        tensor_id: u32,
    ) {
        match self.tensor_store.get(&tensor_id) {
            Some(buf) => {
                if let Err(e) = callback.send(ContextMessage::ReadTensorResult(
                    ctx_id,
                    tensor_id,
                    Ok(buf.as_ref().clone()),
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

    fn handle_write_tensor(&mut self, _ctx_id: ContextId, tensor_id: u32, bytes: Vec<u8>) {
        self.tensor_store.insert(tensor_id, Arc::new(bytes));
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

        self.enqueue_or_run(PendingOp::Dispatch(ctx_id, key, inputs_map, outputs_map))
    }

    fn collect_input_bytes(&self, inputs_map: &HashMap<u32, u32>) -> HashMap<u32, Vec<u8>> {
        let mut inputs_bytes = HashMap::new();
        for (op_id, tensor_id) in inputs_map {
            if let Some(buf) = self.tensor_store.get(tensor_id) {
                inputs_bytes.insert(*op_id, buf.as_ref().clone());
            }
        }
        inputs_bytes
    }

    fn resolve_constant_operands(&mut self, graph_info: &mut GraphInfo) {
        if graph_info.id_to_constant_tensor_operand_map.is_empty() {
            return;
        }
        for (op_id, tensor_id_str) in graph_info.id_to_constant_tensor_operand_map.iter() {
            if let Ok(tid) = tensor_id_str.parse::<u32>() {
                if let Some(buf) = self.tensor_store.get(&tid) {
                    graph_info.constant_operand_ids_to_handles.insert(
                        *op_id,
                        ConstantData {
                            data: buf.as_ref().clone(),
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

    fn handle_compile(
        &mut self,
        ctx_id: ContextId,
        graph_id: GraphId,
        mut graph_info: GraphInfo,
        callback: GenericCallback<ContextMessage>,
    ) {
        // caller already gave us ownership of the graph info; mutate it
        // directly to resolve constants before sending the graph to the
        // compute thread.
        self.resolve_constant_operands(&mut graph_info);

        // record any build callback so we can notify when this graph
        // finishes compiling.
        self.script_build_request
            .entry(graph_id)
            .or_default()
            .push(callback);

        self.enqueue_or_run(PendingOp::Compile(ctx_id, graph_id, graph_info));
    }

    fn build_compute_work(
        &self,
        ctx_id: ContextId,
        key: GraphId,
        inputs_map: HashMap<u32, u32>,
        outputs_map: HashMap<u32, u32>,
    ) -> WorkItem {
        // The manager no longer keeps a cache; the compute thread owns all
        // graph metadata.  We simply forward the inputs to the pooled worker
        // and let it handle missing/uncached graphs (it will zero the outputs).
        let inputs_bytes = self.collect_input_bytes(&inputs_map);

        WorkItem::Compute {
            ctx_id,
            key,
            inputs_map,
            inputs_bytes,
            outputs_map,
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
        match op {
            PendingOp::CreateTensor(cb, ctx_id, tensor_id, len) => {
                self.handle_create_tensor(cb, ctx_id, tensor_id, len);
            },
            PendingOp::CreateConstantTensor(ctx_id, tensor_id, bytes) => {
                self.handle_create_constant_tensor(ctx_id, tensor_id, bytes);
            },
            PendingOp::ReadTensor(cb, ctx_id, tensor_id) => {
                self.handle_read_tensor(cb, ctx_id, tensor_id);
            },
            PendingOp::WriteTensor(ctx_id, tensor_id, bytes) => {
                self.handle_write_tensor(ctx_id, tensor_id, bytes);
            },
            PendingOp::Dispatch(ctx_id, key, inputs_map, outputs_map) => {
                self.queue_blocked = true;
                self.spawn_work(self.build_compute_work(ctx_id, key, inputs_map, outputs_map));
            },
            PendingOp::Compile(ctx_id, key, graph_info) => {
                // block the timeline until the compilation result arrives.
                self.queue_blocked = true;
                self.spawn_work(WorkItem::Compile {
                    ctx_id,
                    key,
                    graph_info,
                });
            },
        }
    }

    fn handle_compiled(&mut self, ctx_id: ContextId, graph_id: GraphId) {
        // notify any build callbacks waiting for this graph_id
        if let Some(vec) = self.script_build_request.remove(&graph_id) {
            for cb in vec {
                // send result back to script thread.  The script stored its
                // own clone of the GraphInfo when the build request was
                // issued, so the callback can construct the MLGraph with that
                // information.  The manager itself does not send the info in
                // the compile result message.
                let _ = cb.send(ContextMessage::CompileResult(ctx_id, graph_id));
            }
        }

        if self.queue_blocked {
            self.queue_blocked = false;
            self.process_queue();
        }
    }

    fn handle_compile_failed(&mut self, ctx_id: ContextId, key: GraphId, reason: String) {
        warn!(
            "webnn manager: compilation failed for {:?}/{}: {}",
            ctx_id, key, reason
        );

        // drop any pending build callbacks; we still call them so the script
        // side can resolve the promises, though they won't be able to dispatch
        // successfully later.
        if let Some(vec) = self.script_build_request.remove(&key) {
            for cb in vec {
                let _ = cb.send(ContextMessage::CompileResult(ctx_id, key));
            }
        }

        if self.queue_blocked {
            self.queue_blocked = false;
            self.process_queue();
        }
    }

    fn handle_compute_result(&mut self, outputs: HashMap<u32, Vec<u8>>) {
        for (tensor_id, bytes) in outputs {
            self.tensor_store.insert(tensor_id, Arc::new(bytes));
        }
        self.queue_blocked = false;
        self.process_queue()
    }

    fn process_queue(&mut self) {
        while !self.queue_blocked {
            if let Some(op) = self.timeline.pop_front() {
                self.run_now(op);
                // loop again unless a dispatch set queue_blocked true
            } else {
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
pub fn new_webnn_manager() -> (GenericSender<WebNNMsg>, JoinHandle<()>) {
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
    let mut manager = WebNNManager::new(manager_tx);
    manager.run(receiver);
}

enum WorkItem {
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
        graph_info: GraphInfo,
    },
}

enum MlCacheEntry {
    Compiling,
    Compiled {
        graph_info: GraphInfo,
        compiled_path: PathBuf,
        // bytes produced by conversion; reused during compute to avoid
        // repeating the convert step.  Always present once entry exists.
        converted_program: Vec<u8>,
    },
    Destroyed(String),
}

type SharedCacheEntry = Arc<(Mutex<MlCacheEntry>, Condvar)>;
type SharedCompiledMap = Arc<RwLock<HashMap<(ContextId, GraphId), SharedCacheEntry>>>;

fn handle_ml_compute(
    ctx_id: ContextId,
    key: GraphId,
    inputs_map: HashMap<u32, u32>,
    inputs_bytes: HashMap<u32, Vec<u8>>,
    outputs_map: HashMap<u32, u32>,
    compiled_map: SharedCompiledMap,
    manager_tx: GenericSender<WebNNMsg>,
) {
    let mut outputs = HashMap::new();

    let cache_entry = {
        let compiled_map = compiled_map.read();
        let entry = compiled_map.get(&(ctx_id, key)).cloned();
        debug_assert!(
            entry.is_some(),
            "compute without cache entry for {:?}/{}",
            ctx_id,
            key
        );
        entry
    };

    // if we haven't seen a compile for this graph yet the ML pool
    // can't execute anything; generate zeroed outputs and bail.  Note
    // that we key the cache by context as well as graph id.
    let Some(cache_entry) = cache_entry else {
        warn!("webnn ML thread: missing cache entry during compute");
        for tensor_id in outputs_map.values() {
            outputs.insert(*tensor_id, Vec::new());
        }
        if let Err(e) = manager_tx.send(WebNNMsg::ComputeResult(ctx_id, outputs)) {
            warn!("webnn ML thread: failed to send ComputeResult: {:?}", e);
        }
        return;
    };

    let (entry_mutex, entry_condvar) = &*cache_entry;
    let mut entry = entry_mutex.lock();

    loop {
        match &*entry {
            MlCacheEntry::Compiling => {
                entry_condvar.wait(&mut entry);
            },
            MlCacheEntry::Destroyed(reason) => {
                warn!(
                    "webnn ML thread: compute observed destroyed cache entry: {}",
                    reason
                );
                for tensor_id in outputs_map.values() {
                    outputs.insert(*tensor_id, Vec::new());
                }
                drop(entry);
                if let Err(e) = manager_tx.send(WebNNMsg::ComputeResult(ctx_id, outputs)) {
                    warn!("webnn ML thread: failed to send ComputeResult: {:?}", e);
                }
                return;
            },
            MlCacheEntry::Compiled {
                graph_info,
                compiled_path,
                converted_program,
            } => {
                let cached_path = Some(compiled_path.as_path());
                let program_bytes = converted_program.as_slice();

                debug!("Starg coreml for {:?} {:?}", ctx_id, inputs_bytes.len());

                if !try_coreml_execute(
                    graph_info,
                    &inputs_map,
                    &inputs_bytes,
                    &outputs_map,
                    program_bytes,
                    cached_path,
                    &mut outputs,
                ) {
                    warn!("CoreML failed");
                    // coreml either not available or failed, generate zeros
                    for (op_id, tensor_id) in &outputs_map {
                        if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                            let element_count: usize =
                                operand.descriptor.shape.iter().fold(1usize, |acc, d| {
                                    acc.saturating_mul(get_static_or_max_size(d) as usize)
                                });
                            let byte_len = element_count
                                .saturating_mul(operand.descriptor.data_type.bytes_per_element());
                            outputs.insert(*tensor_id, vec![0u8; byte_len]);
                        }
                    }
                }
                break;
            },
        }
    }

    debug!("Compute result for {:?}", ctx_id);
    if let Err(e) = manager_tx.send(WebNNMsg::ComputeResult(ctx_id, outputs)) {
        warn!("webnn ML thread: failed to send ComputeResult: {:?}", e);
    }
}

fn handle_ml_compile(
    ctx_id: ContextId,
    key: GraphId,
    graph_info: GraphInfo,
    compiled_map: SharedCompiledMap,
    manager_tx: GenericSender<WebNNMsg>,
) {
    let (cache_entry, should_compile) = {
        let mut compiled_map = compiled_map.write();
        match compiled_map.entry((ctx_id, key)) {
            Entry::Occupied(entry) => (entry.get().clone(), false),
            Entry::Vacant(entry) => {
                let cache_entry = Arc::new((Mutex::new(MlCacheEntry::Compiling), Condvar::new()));
                entry.insert(Arc::clone(&cache_entry));
                (cache_entry, true)
            },
        }
    };

    if !should_compile {
        let (entry_mutex, entry_condvar) = &*cache_entry;
        loop {
            let mut entry = entry_mutex.lock();
            match &*entry {
                MlCacheEntry::Compiling => {
                    entry_condvar.wait(&mut entry);
                },
                MlCacheEntry::Compiled { compiled_path, .. } => {
                    let compiled_path = compiled_path.clone();
                    drop(entry);
                    if let Err(e) = manager_tx.send(WebNNMsg::Compiled(ctx_id, key, compiled_path))
                    {
                        warn!("webnn ML thread: failed to send Compiled: {:?}", e);
                    }
                    return;
                },
                MlCacheEntry::Destroyed(reason) => {
                    let reason = reason.clone();
                    drop(entry);
                    if let Err(e) = manager_tx.send(WebNNMsg::CompileFailed(ctx_id, key, reason)) {
                        warn!("webnn ML thread: failed to send CompileFailed: {:?}", e);
                    }
                    return;
                },
            }
        }
    }

    use rustnn::converters::CoremlMlProgramConverter;

    let converter = CoremlMlProgramConverter;
    match converter.convert(&graph_info) {
        Ok(converted) => {
            let compile_result = std::panic::catch_unwind(|| unsafe {
                prepare_compiled_model_with_weights(
                    &converted.data,
                    converted.weights_data.as_deref(),
                    None,
                )
            });

            match compile_result {
                Ok(Ok((_compiled_url, compiled_path, _temp_mlmodel))) => {
                    let (entry_mutex, entry_condvar) = &*cache_entry;
                    let mut entry = entry_mutex.lock();
                    *entry = MlCacheEntry::Compiled {
                        graph_info,
                        compiled_path: compiled_path.clone(),
                        converted_program: converted.data,
                    };
                    drop(entry);
                    entry_condvar.notify_all();

                    if let Err(e) = manager_tx.send(WebNNMsg::Compiled(ctx_id, key, compiled_path))
                    {
                        warn!("webnn ML thread: failed to send Compiled: {:?}", e);
                    }
                },
                Ok(Err(e)) => {
                    let reason = format!("{:?}", e);
                    let (entry_mutex, entry_condvar) = &*cache_entry;
                    let mut entry = entry_mutex.lock();
                    *entry = MlCacheEntry::Destroyed(reason.clone());
                    drop(entry);
                    entry_condvar.notify_all();
                    warn!("webnn ML thread: compile failed for key {}: {:?}", key, e);
                    if let Err(e) = manager_tx.send(WebNNMsg::CompileFailed(ctx_id, key, reason)) {
                        warn!("webnn ML thread: failed to send CompileFailed: {:?}", e);
                    }
                },
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    let reason = format!("panic: {}", msg);
                    let (entry_mutex, entry_condvar) = &*cache_entry;
                    let mut entry = entry_mutex.lock();
                    *entry = MlCacheEntry::Destroyed(reason.clone());
                    drop(entry);
                    entry_condvar.notify_all();
                    warn!("webnn ML thread: compile panicked for key {}: {}", key, msg);
                    if let Err(e) = manager_tx.send(WebNNMsg::CompileFailed(ctx_id, key, reason)) {
                        warn!("webnn ML thread: failed to send CompileFailed: {:?}", e);
                    }
                },
            }
        },
        Err(err) => {
            let reason = "conversion failed".to_string();
            let (entry_mutex, entry_condvar) = &*cache_entry;
            let mut entry = entry_mutex.lock();
            *entry = MlCacheEntry::Destroyed(reason.clone());
            drop(entry);
            entry_condvar.notify_all();
            warn!(
                "webnn ML thread: conversion failed for graph: {:?} with error {:?}",
                graph_info, err
            );
            if let Err(e) = manager_tx.send(WebNNMsg::CompileFailed(ctx_id, key, reason)) {
                warn!("webnn ML thread: failed to send CompileFailed: {:?}", e);
            }
        },
    }
}

// Common implementation for attempting a CoreML execution.  If it succeeds
// the `outputs_store` map is populated and `true` is returned.  In the
// `#[cfg(not(target_os = "macos"))]` case we simply return false.

#[cfg(target_os = "macos")]
fn try_coreml_execute(
    graph_info: &GraphInfo,
    inputs_map: &HashMap<u32, u32>,
    inputs_bytes: &HashMap<u32, Vec<u8>>,
    outputs_map: &HashMap<u32, u32>,
    program_bytes: &[u8],
    cached_compiled: Option<&Path>,
    outputs_store: &mut HashMap<u32, Vec<u8>>,
) -> bool {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_cached};

    // Perfomance TODO: way too many allocations in the below loops; we
    // alleviate a few of them here by reusing buffers and preallocating
    // capacity.  The biggest allocation still comes from converting the
    // byte buffers into `Vec<f32>` for CoreML; reducing that would require a
    // more invasive API change.
    let mut coreml_inputs: Vec<CoremlInput> = Vec::with_capacity(inputs_map.len());
    // reusable temporary for shape computation
    let mut shape_buf: Vec<usize> = Vec::new();

    for (op_id, _) in inputs_map.iter() {
        if let Some(op) = graph_info.operands.get(*op_id as usize) {
            let input_name = op
                .name
                .as_deref()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("input_{}", op_id));

            if let Some(buf) = inputs_bytes.get(op_id) {
                let data: Vec<f32> = match op.descriptor.data_type {
                    DataType::Float32 => {
                        // convert in one pass using chunks; avoids the inner
                        // byte buffer and manually advancing index.
                        buf.chunks_exact(4)
                            .map(|b| {
                                let mut arr = [0u8; 4];
                                arr.copy_from_slice(b);
                                f32::from_le_bytes(arr)
                            })
                            .collect()
                    },
                    DataType::Float16 => buf
                        .chunks_exact(2)
                        .map(|b| {
                            let bits = u16::from_le_bytes([b[0], b[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect(),
                    // Int32 inputs are promoted to float32 for CoreML.  We
                    // simply cast each i32 element to f32 here rather than
                    // attempting to preserve integer semantics; CoreML
                    // typically works in floating point anyway.
                    DataType::Int32 => buf
                        .chunks_exact(4)
                        .map(|b| {
                            let mut arr = [0u8; 4];
                            arr.copy_from_slice(b);
                            let iv = i32::from_le_bytes(arr);
                            iv as f32
                        })
                        .collect(),
                    DataType::Uint8 => buf.iter().map(|&b| b as f32).collect(),
                    _other => Vec::new(),
                };

                if !data.is_empty() {
                    shape_buf.clear();
                    shape_buf.extend(
                        op.descriptor
                            .shape
                            .iter()
                            .map(|d| get_static_or_max_size(d) as usize),
                    );
                    coreml_inputs.push(CoremlInput {
                        name: input_name,
                        shape: shape_buf.clone(),
                        data,
                    });
                }
            }
        }
    }

    // at this point the graph should already have been compiled on a
    // previous path; ensure we have a cached model directory.  If this
    // assertion ever trips in release it indicates a programming error
    // where dispatch was allowed to run without compilation.
    debug_assert!(
        cached_compiled.is_some(),
        "dispatch without cached compiled model"
    );
    let path = cached_compiled.expect("cached model path");
    let run_result = run_coreml_with_inputs_cached(program_bytes, coreml_inputs, Some(path));
    if let Ok(attempts) = run_result {
        if let Some(outputs) = attempts
            .iter()
            .find_map(|a| a.result.as_ref().ok().cloned())
        {
            // Build a list of outputs we'll consume as we iterate.  The
            // outer block above already performed validation; we just need a
            // single loop that handles name mismatches gracefully.
            let mut remaining_outputs: Vec<CoremlOutput> = outputs.clone();
            for (op_id, tensor_id) in outputs_map.iter() {
                if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                    let default_name = format!("output_{}", op_id);
                    let output_name = operand.name.as_deref().unwrap_or(&default_name);

                    debug!("Output name: {:?}", output_name);

                    let maybe_coreml = remaining_outputs
                        .iter()
                        .position(|o| o.name == output_name)
                        .map(|idx| remaining_outputs.remove(idx));

                    let coreml_out = if let Some(o) = maybe_coreml {
                        Some(o)
                    } else {
                        if remaining_outputs.is_empty() {
                            None
                        } else {
                            debug!(
                                "CoreML output name '{}' not found, using next positional output",
                                output_name
                            );
                            Some(remaining_outputs.remove(0))
                        }
                    };

                    let bytes = process_coreml_outputs(operand, coreml_out);
                    outputs_store.insert(*tensor_id, bytes);
                }
            }
            return true;
        }
    }
    false
}

#[cfg(not(target_os = "macos"))]
fn try_coreml_execute(
    _graph_info: &mut GraphInfo,
    _inputs_map: &HashMap<u32, u32>,
    _inputs_bytes: &HashMap<u32, Vec<u8>>,
    _outputs_map: &HashMap<u32, u32>,
    _program_bytes: &[u8],
    _outputs_store: &mut HashMap<u32, Vec<u8>>,
) -> bool {
    false
}

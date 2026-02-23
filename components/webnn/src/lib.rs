use std::collections::HashMap;
use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use crossbeam::channel::{self, Sender, Receiver};
use log::{debug, warn};
use profile_traits::generic_callback::GenericCallback;
// Required to materialize constant operands during dispatch.
use rustnn::graph::ConstantData;
use webnn_traits::{ContextId, ContextMessage, WebNNMsg};

#[derive(Debug)]
struct Context {
    // Backend-specific context state.
    tensor_store: HashMap<u32, Vec<u8>>,
    // Sender for offloading ML work to the dedicated thread.
    compute_tx: Sender<MlMsg>,
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
                self.contexts.insert(id, Context::new(self.ml_sender.clone()));
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
                    ctx.handle_create_tensor(callback, ctx_id, tensor_id, byte_length);
                } else {
                    warn!(
                        "webnn manager: CreateTensor for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
            WebNNMsg::ReadTensor(callback, ctx_id, tensor_id) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_read_tensor(callback, ctx_id, tensor_id);
                } else {
                    warn!("webnn manager: ReadTensor for unknown context {:?}", ctx_id);
                }
                true
            },
            WebNNMsg::WriteTensor(ctx_id, tensor_id, bytes) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_write_tensor(ctx_id, tensor_id, bytes);
                } else {
                    warn!(
                        "webnn manager: WriteTensor for unknown context {:?}",
                        ctx_id
                    );
                }
                true
            },
            WebNNMsg::Dispatch(ctx_id, graph_info, inputs_map, outputs_map) => {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.handle_dispatch(ctx_id, graph_info, inputs_map, outputs_map);
                } else {
                    warn!("webnn manager: Dispatch for unknown context {:?}", ctx_id);
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
                if let Err(e) = callback.send(ContextMessage::ReadTensorResult(ctx_id, tensor_id, Err(()))) {
                    warn!("webnn manager: ReadTensor error callback send failed: {:?}", e);
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
        graph_info: rustnn::graph::GraphInfo,
        inputs_map: Vec<(u32, u32)>,
        outputs_map: Vec<(u32, u32)>,
    ) {
        // convert to HashMap for easier lookup inside the dispatch logic
        let inputs_map: HashMap<u32, u32> = inputs_map.into_iter().collect();
        let outputs_map: HashMap<u32, u32> = outputs_map.into_iter().collect();
        debug!(
            "webnn manager: Dispatch ctx={:?} graph_ops={} inputs={} outputs={}",
            ctx_id,
            graph_info.operations.len(),
            inputs_map.len(),
            outputs_map.len()
        );

        if graph_info.output_operands.is_empty() {
            debug!(
                "webnn manager: Dispatch ctx={:?} has no outputs; skipping execution",
                ctx_id
            );
            return;
        }

        // Pass ownership of the maps into compute directly.  we also hand
        // over a reference to the compute channel so the call below can send
        // work to the ml thread.
        // clone sender to avoid borrowing `self` mutably while we still
        // need to access other fields.
        let compute_chan = self.compute_tx.clone();
        self.compute(
            ctx_id,
            graph_info,
            inputs_map,
            outputs_map,
            &compute_chan,
        );
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
    fn compute(
        &mut self,
        _ctx_id: ContextId,
        mut graph_info: rustnn::graph::GraphInfo,
        inputs_map: HashMap<u32, u32>,
        outputs_map: HashMap<u32, u32>,
        compute_tx: &Sender<MlMsg>,
    ) {
        let inputs_bytes = self.collect_input_bytes(&inputs_map);
        self.resolve_constant_operands(&mut graph_info);

        // send work to the ml thread, then wait for the outputs mapping.  we always
        // get a complete set of bytes back (even if CoreML failed, the worker will
        // generate zeroed outputs), so we can just apply whatever we receive.
        let (response_tx, response_rx) = channel::bounded(1);

        // clone the graph info and outputs_map for the worker; we still need
        // the originals for fallback if something goes wrong.
        let work_graph = graph_info.clone();
        let work_outputs = outputs_map.clone();

        let msg = MlMsg::Compute {
            graph_info: work_graph,
            inputs_map,
            inputs_bytes,
            outputs_map: work_outputs,
            response: response_tx,
        };

        // this send should never fail unless the worker has already exited;
        // in that case we'll simply fall back to zeroed outputs below.
        if let Err(e) = compute_tx.send(msg) {
            warn!("webnn manager: failed to send compute message: {:?}", e);
        }
        if let Ok(outputs) = response_rx.recv() {
            for (tensor_id, bytes) in outputs {
                self.tensor_store.insert(tensor_id, bytes);
            }
            return;
        }

        // channel closed or error, do local zeroed fallback using the original
        // graph_info we still own.
        self.zeroed_outputs(&graph_info, &outputs_map);
    }

    fn zeroed_outputs(
        &mut self,
        graph_info: &rustnn::graph::GraphInfo,
        outputs_map: &HashMap<u32, u32>,
    ) {
        for (op_id, tensor_id) in outputs_map.iter() {
            if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                let element_count: usize = operand
                    .descriptor
                    .shape
                    .iter()
                    .fold(1usize, |acc, &d| acc.saturating_mul(d as usize));
                let byte_len =
                    element_count.saturating_mul(operand.descriptor.data_type.bytes_per_element());
                self.tensor_store.insert(*tensor_id, vec![0u8; byte_len]);
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

    let handle = thread::Builder::new()
        .name("WebNNManager".into())
        .spawn(move || run_manager(rx))
        .expect("failed to spawn WebNN manager thread");

    (tx, handle)
}

fn run_manager(receiver: GenericReceiver<WebNNMsg>) {
    // create dedicated channel for ML work
    let (ml_tx, ml_rx): (Sender<MlMsg>, Receiver<MlMsg>) = channel::unbounded();

    // spawn the ML worker thread
    let ml_handle = thread::Builder::new()
        .name("WebNNML".into())
        .spawn(move || ml_loop(ml_rx))
        .expect("failed to spawn WebNN ML thread");

    // manager now owns a sender that it will clone into each context
    let mut manager = WebNNManager::new(ml_tx.clone());
    manager.run(receiver);

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
// copy of all data necessary to run CoreML (or compute zeroed outputs) and
// a one-shot channel on which the worker returns the resulting bytes map.
#[derive(Debug)]
enum MlMsg {
    Compute {
        graph_info: rustnn::graph::GraphInfo,
        inputs_map: HashMap<u32, u32>,
        inputs_bytes: HashMap<u32, Vec<u8>>,
        outputs_map: HashMap<u32, u32>,
        response: Sender<HashMap<u32, Vec<u8>>>,
    },
    Exit,
}

// Worker loop run on the dedicated ML thread.  It waits for compute messages,
// executes `try_coreml_execute` (or generates zeroed outputs if that fails),
// and then sends the resulting tensor bytes back to the caller.
fn ml_loop(rx: Receiver<MlMsg>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            MlMsg::Compute {
                mut graph_info,
                inputs_map,
                inputs_bytes,
                outputs_map,
                response,
            } => {
                let mut outputs = HashMap::new();
                if !try_coreml_execute(
                    &mut graph_info,
                    &inputs_map,
                    &inputs_bytes,
                    &outputs_map,
                    &mut outputs,
                ) {
                    // coreml either not available or failed, generate zeros
                    for (op_id, tensor_id) in outputs_map.iter() {
                        if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                            let element_count: usize = operand
                                .descriptor
                                .shape
                                .iter()
                                .fold(1usize, |acc, &d| acc.saturating_mul(d as usize));
                            let byte_len =
                                element_count.saturating_mul(operand.descriptor.data_type.bytes_per_element());
                            outputs.insert(*tensor_id, vec![0u8; byte_len]);
                        }
                    }
                }
                if let Err(e) = response.send(outputs) {
                    warn!("webnn ML thread: response send failed: {:?}", e);
                }
            }
            MlMsg::Exit => break,
        }
    }
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
    outputs_store: &mut HashMap<u32, Vec<u8>>,
) -> bool {
    use rustnn::GraphConverter;
    use rustnn::converters::CoremlMlProgramConverter;
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};

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
                    let shape: Vec<usize> =
                        op.descriptor.shape.iter().map(|&d| d as usize).collect();
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
        let weights_ref = converted.weights_data.as_deref();
        if let Ok(attempts) =
            run_coreml_with_inputs_with_weights(&converted.data, weights_ref, coreml_inputs)
        {
            if let Some(outputs) = attempts
                .iter()
                .find_map(|a| a.result.as_ref().ok().cloned())
            {
                for (op_id, tensor_id) in outputs_map.iter() {
                    if let Some(operand) = graph_info.operands.get(*op_id as usize) {
                        let default_name = format!("output_{}", op_id);
                        let output_name = operand.name.as_deref().unwrap_or(&default_name);

                        if let Some(coreml_out) = outputs.iter().find(|o| o.name == output_name)
                        {
                            match operand.descriptor.data_type {
                                rustnn::graph::DataType::Float32 => {
                                    let mut bytes =
                                        Vec::with_capacity(coreml_out.data.len() * 4);
                                    for &v in coreml_out.data.iter() {
                                        bytes.extend_from_slice(&v.to_le_bytes());
                                    }
                                    outputs_store.insert(*tensor_id, bytes);
                                },
                                rustnn::graph::DataType::Float16 => {
                                    let mut bytes =
                                        Vec::with_capacity(coreml_out.data.len() * 2);
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
                                        .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
                                        .saturating_mul(4usize);
                                    outputs_store
                                        .insert(*tensor_id, vec![0u8; byte_length]);
                                },
                            }
                        } else {
                            let byte_length = operand
                                .descriptor
                                .shape
                                .iter()
                                .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
                                .saturating_mul(4usize);
                            outputs_store.insert(*tensor_id, vec![0u8; byte_length]);
                        }
                    }
                }
                return true;
            }
        }
    }
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

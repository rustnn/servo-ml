use std::collections::HashMap;
use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use log::{debug, warn};
use profile_traits::generic_callback::GenericCallback;
// Required to materialize constant operands during dispatch.
use rustnn::graph::ConstantData;
use webnn_traits::{ContextId, ContextMessage, WebNNMsg};

#[derive(Debug)]
struct ContextInfo {
    // Placeholder for future backend-specific context state.
}

/// State handled by the manager thread.  Encapsulating the mutable maps
/// inside a struct lets us move the large match arms into helpers and
/// dramatically reduce nesting in `run_manager`.
struct WebNNManager {
    contexts: HashMap<ContextId, ContextInfo>,
    tensor_store: HashMap<(ContextId, u32), Vec<u8>>,
}

impl WebNNManager {
    fn new() -> Self {
        WebNNManager {
            contexts: HashMap::new(),
            tensor_store: HashMap::new(),
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
                false
            },
            WebNNMsg::NewContext(id) => {
                debug!("webnn manager: NewContext {:?}", id);
                self.contexts.insert(id, ContextInfo {});
                true
            },
            WebNNMsg::DestroyContext(id) => {
                debug!("webnn manager: DestroyContext {:?}", id);
                self.tensor_store.retain(|(ctx, _), _| ctx != &id);
                self.contexts.remove(&id);
                true
            },
            WebNNMsg::CreateTensor(callback, ctx_id, tensor_id, byte_length) => {
                self.handle_create_tensor(callback, ctx_id, tensor_id, byte_length);
                true
            },
            WebNNMsg::ReadTensor(callback, ctx_id, tensor_id) => {
                self.handle_read_tensor(callback, ctx_id, tensor_id);
                true
            },
            WebNNMsg::WriteTensor(ctx_id, tensor_id, bytes) => {
                self.handle_write_tensor(ctx_id, tensor_id, bytes);
                true
            },
            WebNNMsg::Dispatch(ctx_id, graph_info, inputs_map, outputs_map) => {
                self.handle_dispatch(ctx_id, graph_info, inputs_map, outputs_map);
                true
            },
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
        self.tensor_store.insert((ctx_id, tensor_id), buffer);
        let _ = callback.send(ContextMessage::CreateTensorResult(
            ctx_id,
            tensor_id,
            Ok(()),
        ));
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
        match self.tensor_store.get(&(ctx_id, tensor_id)) {
            Some(buf) => {
                let _ = callback.send(ContextMessage::ReadTensorResult(
                    ctx_id,
                    tensor_id,
                    Ok(buf.clone()),
                ));
            },
            None => {
                warn!(
                    "webnn manager: ReadTensor - missing buffer for {:?}/{}",
                    ctx_id, tensor_id
                );
                let _ = callback.send(ContextMessage::ReadTensorResult(ctx_id, tensor_id, Err(())));
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
        self.tensor_store.insert((ctx_id, tensor_id), bytes);
    }

    fn handle_dispatch(
        &mut self,
        ctx_id: ContextId,
        mut graph_info: rustnn::graph::GraphInfo,
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

        let inputs_bytes = self.collect_input_bytes(ctx_id, &inputs_map);
        self.resolve_constant_operands(ctx_id, &mut graph_info);

        // Attempt CoreML execution; on macOS this may succeed and write outputs.
        if self.try_coreml(
            ctx_id,
            &mut graph_info,
            &inputs_map,
            &inputs_bytes,
            &outputs_map,
        ) {
            return;
        }

        // Fall back to zeroed outputs if we didn't already write something.
        self.zeroed_outputs(ctx_id, &graph_info, &outputs_map);
    }

    fn collect_input_bytes(
        &self,
        ctx_id: ContextId,
        inputs_map: &HashMap<u32, u32>,
    ) -> HashMap<u32, Vec<u8>> {
        let mut inputs_bytes = HashMap::new();
        for (op_id, tensor_id) in inputs_map {
            if let Some(buf) = self.tensor_store.get(&(ctx_id, *tensor_id)) {
                inputs_bytes.insert(*op_id, buf.clone());
            }
        }
        inputs_bytes
    }

    fn resolve_constant_operands(
        &mut self,
        ctx_id: ContextId,
        graph_info: &mut rustnn::graph::GraphInfo,
    ) {
        if graph_info.id_to_constant_tensor_operand_map.is_empty() {
            return;
        }
        for (op_id, tensor_id_str) in graph_info.id_to_constant_tensor_operand_map.iter() {
            if let Ok(tid) = tensor_id_str.parse::<u32>() {
                if let Some(buf) = self.tensor_store.get(&(ctx_id, tid)) {
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

    #[cfg(target_os = "macos")]
    fn try_coreml(
        &mut self,
        ctx_id: ContextId,
        graph_info: &mut rustnn::graph::GraphInfo,
        inputs_map: &HashMap<u32, u32>,
        inputs_bytes: &HashMap<u32, Vec<u8>>,
        outputs_map: &HashMap<u32, u32>,
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
                                        self.tensor_store.insert((ctx_id, *tensor_id), bytes);
                                    },
                                    rustnn::graph::DataType::Float16 => {
                                        let mut bytes =
                                            Vec::with_capacity(coreml_out.data.len() * 2);
                                        for &v in coreml_out.data.iter() {
                                            let bits = half::f16::from_f32(v).to_bits();
                                            bytes.extend_from_slice(&bits.to_le_bytes());
                                        }
                                        self.tensor_store.insert((ctx_id, *tensor_id), bytes);
                                    },
                                    _other => {
                                        let byte_length = operand
                                            .descriptor
                                            .shape
                                            .iter()
                                            .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
                                            .saturating_mul(4usize);
                                        self.tensor_store
                                            .insert((ctx_id, *tensor_id), vec![0u8; byte_length]);
                                    },
                                }
                            } else {
                                let byte_length = operand
                                    .descriptor
                                    .shape
                                    .iter()
                                    .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
                                    .saturating_mul(4usize);
                                self.tensor_store
                                    .insert((ctx_id, *tensor_id), vec![0u8; byte_length]);
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
    fn try_coreml(
        &mut self,
        _ctx_id: ContextId,
        _graph_info: &mut rustnn::graph::GraphInfo,
        _inputs_map: &HashMap<u32, u32>,
        _inputs_bytes: &HashMap<u32, Vec<u8>>,
        _outputs_map: &HashMap<u32, u32>,
    ) -> bool {
        false
    }

    fn zeroed_outputs(
        &mut self,
        ctx_id: ContextId,
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
                self.tensor_store
                    .insert((ctx_id, *tensor_id), vec![0u8; byte_len]);
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
    let mut manager = WebNNManager::new();
    manager.run(receiver);
}

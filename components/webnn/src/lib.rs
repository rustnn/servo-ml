use std::collections::HashMap;
use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use log::{debug, warn};
// CoreML runtime / converters (only compiled on macOS with the feature enabled)
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
use rustnn::converters::CoremlMlProgramConverter;
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
use webnn_traits::{ContextId, ContextMessage, WebNNMsg};

#[derive(Debug)]
struct ContextInfo {
    // Placeholder for future backend-specific context state.
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
    debug!("webnn manager started");

    let mut contexts: HashMap<ContextId, ContextInfo> = HashMap::new();
    // Store backend-side tensor buffers keyed by (context_id, tensor_id).
    let mut tensor_store: HashMap<(ContextId, u32), Vec<u8>> = HashMap::new();

    loop {
        match receiver.recv() {
            Ok(msg) => {
                match msg {
                    WebNNMsg::Exit => {
                        debug!("webnn manager exiting");
                        break;
                    },
                    WebNNMsg::NewContext(id) => {
                        debug!("webnn manager: NewContext {:?}", id);
                        contexts.insert(id, ContextInfo {});
                    },
                    WebNNMsg::DestroyContext(id) => {
                        debug!("webnn manager: DestroyContext {:?}", id);
                        // Remove any stored tensors associated with that context.
                        tensor_store.retain(|(ctx, _), _| ctx != &id);
                        contexts.remove(&id);
                    },
                    WebNNMsg::CreateTensor(callback, ctx_id, tensor_id, byte_length) => {
                        debug!(
                            "webnn manager: CreateTensor ctx={:?} id={} len={}",
                            ctx_id, tensor_id, byte_length
                        );
                        // Simple stub backend: create a Vec<u8> whose element bytes represent
                        // IEEE-754 float32 zeros (explicitly treat tensor storage as f32-packed).
                        // This makes the backing bytes explicitly f32 (4-byte) elements while
                        // preserving the requested byte length.
                        let mut buffer: Vec<u8> = Vec::with_capacity(byte_length);
                        let n_f32 = byte_length / 4;
                        for _ in 0..n_f32 {
                            buffer.extend_from_slice(&0.0f32.to_le_bytes());
                        }
                        if byte_length % 4 != 0 {
                            buffer.extend(std::iter::repeat(0u8).take(byte_length % 4));
                        }
                        tensor_store.insert((ctx_id, tensor_id), buffer);
                        // Send a ContextMessage so the ML-level persistent callback can
                        // route the reply by ContextId.
                        let _ = callback.send(ContextMessage::CreateTensorResult(
                            ctx_id,
                            tensor_id,
                            Ok(()),
                        ));
                    },

                    WebNNMsg::ReadTensor(callback, ctx_id, tensor_id) => {
                        debug!(
                            "webnn manager: ReadTensor ctx={:?} id={}",
                            ctx_id, tensor_id
                        );
                        match tensor_store.get(&(ctx_id, tensor_id)) {
                            Some(buf) => {
                                // Return a copy of the bytes to the caller via callback.
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
                                let _ = callback.send(ContextMessage::ReadTensorResult(
                                    ctx_id,
                                    tensor_id,
                                    Err(()),
                                ));
                            },
                        }
                    },

                    WebNNMsg::WriteTensor(ctx_id, tensor_id, bytes) => {
                        debug!(
                            "webnn manager: WriteTensor ctx={:?} id={} len={}",
                            ctx_id,
                            tensor_id,
                            bytes.len()
                        );
                        // Overwrite or create the backend buffer for the tensor.
                        tensor_store.insert((ctx_id, tensor_id), bytes);
                    },

                    WebNNMsg::Dispatch(ctx_id, graph_info, inputs_map, outputs_map) => {
                        debug!(
                            "webnn manager: Dispatch ctx={:?} graph_ops={} inputs={} outputs={}",
                            ctx_id,
                            graph_info.operations.len(),
                            inputs_map.len(),
                            outputs_map.len()
                        );

                        // Collect input buffers keyed by operand id.
                        let mut inputs_bytes: std::collections::HashMap<u32, Vec<u8>> =
                            std::collections::HashMap::new();
                        for (op_id, tensor_id) in inputs_map.iter() {
                            if let Some(buf) = tensor_store.get(&(ctx_id, *tensor_id)) {
                                inputs_bytes.insert(*op_id, buf.clone());
                            } else {
                                warn!(
                                    "webnn manager: Dispatch - missing input buffer for {:?}/{}",
                                    ctx_id, tensor_id
                                );
                            }
                        }

                        // Try CoreML execution (macOS + coreml-runtime). If unavailable or execution fails,
                        // fall back to the previous naive behavior.
                        #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
                        {
                            use rustnn::converters::CoremlMlProgramConverter;
                            use rustnn::executors::coreml::{
                                CoremlInput, run_coreml_with_inputs_with_weights,
                            };

                            debug!("webnn manager: Dispatch — attempting CoreML execution");

                            // Build CoreML inputs from stored tensor bytes (supports Float32/Float16).
                            let mut coreml_inputs: Vec<CoremlInput> = Vec::new();
                            for (op_id, _tensor_id) in inputs_map.iter() {
                                if let Some(op) = graph_info.operands.get(*op_id as usize) {
                                    let default_name = format!("input_{}", op_id);
                                    let input_name =
                                        op.name.as_deref().unwrap_or(&default_name).to_string();

                                    if let Some(buf) = inputs_bytes.get(op_id) {
                                        let data: Vec<f32> = match op.descriptor.data_type {
                                            rustnn::graph::DataType::Float32 => {
                                                // interpret bytes as little-endian f32
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
                                                // interpret bytes as little-endian u16 half-floats -> f32
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
                                            other => {
                                                warn!(
                                                    "webnn manager: Dispatch CoreML — unsupported input dtype {:?} for operand {} (skipping)",
                                                    other, op_id
                                                );
                                                Vec::new()
                                            },
                                        };

                                        // Only push inputs with non-empty data and known shapes
                                        if !data.is_empty() {
                                            let shape: Vec<usize> = op
                                                .descriptor
                                                .shape
                                                .iter()
                                                .map(|&d| d as usize)
                                                .collect();
                                            coreml_inputs.push(CoremlInput {
                                                name: input_name,
                                                shape,
                                                data,
                                            });
                                        }
                                    } else {
                                        warn!(
                                            "webnn manager: Dispatch CoreML - missing input buffer for {:?}/{}",
                                            ctx_id, op_id
                                        );
                                    }
                                }
                            }

                            // Convert GraphInfo -> CoreML model
                            let converter = CoremlMlProgramConverter;
                            match converter.convert(&graph_info) {
                                Ok(converted) => {
                                    let weights_ref = converted.weights_data.as_deref();
                                    match run_coreml_with_inputs_with_weights(
                                        &converted.data,
                                        weights_ref,
                                        coreml_inputs,
                                    ) {
                                        Ok(attempts) => {
                                            if let Some(outputs) = attempts
                                                .iter()
                                                .find_map(|a| a.result.as_ref().ok().cloned())
                                            {
                                                // Map CoreML outputs back into tensor_store using outputs_map
                                                for (op_id, tensor_id) in outputs_map.iter() {
                                                    if let Some(operand) =
                                                        graph_info.operands.get(*op_id as usize)
                                                    {
                                                        let default_name =
                                                            format!("output_{}", op_id);
                                                        let output_name = operand
                                                            .name
                                                            .as_deref()
                                                            .unwrap_or(&default_name);

                                                        if let Some(coreml_out) = outputs
                                                            .iter()
                                                            .find(|o| o.name == output_name)
                                                        {
                                                            // write bytes according to operand dtype
                                                            match operand.descriptor.data_type {
                                                            rustnn::graph::DataType::Float32 => {
                                                                let mut bytes = Vec::with_capacity(coreml_out.data.len() * 4);
                                                                for &v in coreml_out.data.iter() {
                                                                    bytes.extend_from_slice(&v.to_le_bytes());
                                                                }
                                                                tensor_store.insert((ctx_id, *tensor_id), bytes);
                                                            }
                                                            rustnn::graph::DataType::Float16 => {
                                                                let mut bytes = Vec::with_capacity(coreml_out.data.len() * 2);
                                                                for &v in coreml_out.data.iter() {
                                                                    let bits = half::f16::from_f32(v).to_bits();
                                                                    bytes.extend_from_slice(&bits.to_le_bytes());
                                                                }
                                                                tensor_store.insert((ctx_id, *tensor_id), bytes);
                                                            }
                                                            other => {
                                                                warn!(
                                                                    "webnn manager: Dispatch CoreML — unsupported output dtype {:?} for operand {}; storing zeroed buffer",
                                                                    other, op_id
                                                                );
                                                                // fall through to zeroed buffer below
                                                                let byte_length = operand
                                                                    .descriptor
                                                                    .shape
                                                                    .iter()
                                                                    .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
                                                                    .saturating_mul(4usize);
                                                                tensor_store.insert((ctx_id, *tensor_id), vec![0u8; byte_length]);
                                                            }
                                                        }
                                                        } else {
                                                            warn!(
                                                                "webnn manager: Dispatch CoreML — no output named '{}' returned by CoreML; storing zeroed buffer",
                                                                output_name
                                                            );
                                                            let byte_length = operand
                                                                .descriptor
                                                                .shape
                                                                .iter()
                                                                .fold(1usize, |acc, &d| {
                                                                    acc.saturating_mul(d as usize)
                                                                })
                                                                .saturating_mul(4usize);
                                                            tensor_store.insert(
                                                                (ctx_id, *tensor_id),
                                                                vec![0u8; byte_length],
                                                            );
                                                        }
                                                    } else {
                                                        warn!(
                                                            "webnn manager: Dispatch CoreML — unknown operand id {}",
                                                            op_id
                                                        );
                                                    }
                                                }

                                                // CoreML succeeded; skip fallback.
                                                continue;
                                            } else {
                                                warn!(
                                                    "webnn manager: Dispatch CoreML — no successful CoreML attempt"
                                                );
                                            }
                                        },
                                        Err(e) => {
                                            warn!(
                                                "webnn manager: Dispatch CoreML execution failed: {:?}",
                                                e
                                            );
                                        },
                                    }
                                },
                                Err(e) => {
                                    warn!(
                                        "webnn manager: Dispatch CoreML conversion failed: {:?}",
                                        e
                                    );
                                },
                            }
                            // If we reach here, CoreML path failed — fall through to fallback.
                        }

                        // If CoreML is not available
                        // (or Dispatch fails), do not perform a heuristic/default execution here.
                        // Instead warn so the lack of a backend is visible; outputs will not be
                        // produced by the manager. Implement a real backend in `components/webnn`
                        // or enable the `coreml-runtime` feature on macOS.
                        warn!(
                            "webnn manager: Dispatch received but no backend available — graph execution is a no-op"
                        );
                    },
                }
            },
            Err(_) => break,
        }
    }
    debug!("webnn manager stopped");
}

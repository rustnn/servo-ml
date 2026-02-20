use std::collections::HashMap;
use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use log::{debug, warn};
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
            Ok(msg) => match msg {
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
                    match tensor_store.get_mut(&(ctx_id, tensor_id)) {
                        Some(buf) => {
                            // If sizes match, overwrite in-place; otherwise replace.
                            if buf.len() == bytes.len() {
                                buf.copy_from_slice(&bytes);
                            } else {
                                *buf = bytes;
                            }
                        },
                        None => {
                            warn!(
                                "webnn manager: WriteTensor - missing buffer for {:?}/{}",
                                ctx_id, tensor_id
                            );
                        },
                    }
                },
            },
            Err(_) => break,
        }
    }
    debug!("webnn manager stopped");
}

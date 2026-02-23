//! Shared WebNN message types used across the process boundary.

use std::collections::HashMap;

use base::id::PipelineId;
use serde::{Deserialize, Serialize};

/// A simple identifier for a WebNN context. Contains the originating
/// `PipelineId` and a per-pipeline context counter.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ContextId {
    pub pipeline_id: PipelineId,
    pub counter: u32,
}

impl malloc_size_of::MallocSizeOf for ContextId {
    fn size_of(&self, _ops: &mut malloc_size_of::MallocSizeOfOps) -> usize {
        0
    }
}

/// Messages addressed to the WebNN manager.
use profile_traits::generic_callback::GenericCallback;
use rustnn::graph::GraphInfo;

/// Messages delivered through the ML-level persistent callback.  Keeping a
/// small enum here makes it easy to extend the ML callback for other
/// context-scoped replies (not just CreateTensor results).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContextMessage {
    /// Reply for a CreateTensor request: (context id, tensor id, result).
    CreateTensorResult(ContextId, u32, Result<(), ()>),

    /// Reply for a ReadTensor request: (context id, tensor id, result bytes).
    /// The Result contains the requested byte vector on success, or an Err(()) on failure.
    ReadTensorResult(ContextId, u32, Result<Vec<u8>, ()>),
}

impl malloc_size_of::MallocSizeOf for ContextMessage {
    fn size_of(&self, _ops: &mut malloc_size_of::MallocSizeOfOps) -> usize {
        0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WebNNMsg {
    /// Graceful shutdown.
    Exit,

    /// Inform the backend that a new context has been created.
    NewContext(ContextId),

    /// Inform the backend that a context was destroyed and may be freed.
    DestroyContext(ContextId),

    /// Request the backend to create a tensor for `context_id`.
    ///
    /// Arguments: callback, context id, tensor id (script-side u32), byte length to allocate.
    /// The backend should allocate a Vec<u8> of the given length, store it keyed by
    /// (context_id, tensor_id), and invoke `callback` with a `ContextMessage::CreateTensorResult`
    /// containing the `ContextId` and `tensor_id` and Ok(()) / Err(()) to indicate
    /// success or failure. Using `ContextMessage` lets the same persistent ML-level
    /// callback be reused for other context-level replies in the future.
    CreateTensor(GenericCallback<ContextMessage>, ContextId, u32, usize),

    /// Request the backend to return the bytes for the tensor identified by
    /// `(context_id, tensor_id)`. The backend should look up its stored buffer
    /// and invoke `callback` with `ContextMessage::ReadTensorResult` containing
    /// the copied bytes (or an error).
    ReadTensor(GenericCallback<ContextMessage>, ContextId, u32),

    /// Request the backend to write bytes into an existing tensor buffer.
    /// Arguments: ContextId, tensor id, bytes to write.
    WriteTensor(ContextId, u32, Vec<u8>),

    /// Dispatch a graph execution request to the backend.
    ///
    /// Arguments:
    /// - ContextId: originating context
    /// - GraphInfo: the implementation-defined graph description (rustnn::GraphInfo)
    /// - Vec<(operand_id, tensor_id)>: mapping of graph input operand ids -> backend tensor ids
    /// - Vec<(operand_id, tensor_id)>: mapping of graph output operand ids -> backend tensor ids
    Dispatch(ContextId, GraphInfo, Vec<(u32, u32)>, Vec<(u32, u32)>),

    /// Response from the backend containing output tensor bytes for a
    /// previously-dispatched graph.  The context id allows the manager thread
    /// to look up the correct `Context` instance and merge the results into
    /// its tensor store.
    ComputeResult(ContextId, HashMap<u32, Vec<u8>>),
}

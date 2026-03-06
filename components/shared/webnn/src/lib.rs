//! Shared WebNN message types used across the process boundary.

use std::collections::HashMap;
use std::path::PathBuf;

/// An MLContext identifier.  Defined in `base::id` alongside other
/// pipeline-scoped ids; it uses the usual namespace machinery to stay unique
/// across workers.
/// All other ids useed in webnn are unique only in so far as they are used
/// in a way that is nested to a context id.
pub use base::id::MLContextId as ContextId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct GraphId(pub u32);

impl std::fmt::Display for GraphId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl malloc_size_of::MallocSizeOf for GraphId {
    fn size_of(&self, _ops: &mut malloc_size_of::MallocSizeOfOps) -> usize {
        0
    }
}

/// Messages addressed to the WebNN manager.
use profile_traits::generic_callback::GenericCallback;
use rustnn::graph::GraphInfo;

/// Messages delivered through the ML-level persistent callback.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContextMessage {
    /// Reply for a CreateTensor request: (context id, tensor id, result).
    CreateTensorResult(ContextId, u32, Result<(), ()>),

    /// Reply for a ReadTensor request: (context id, tensor id, result bytes).
    /// The Result contains the requested byte vector on success, or an Err(()) on failure.
    ReadTensorResult(ContextId, u32, Result<Vec<u8>, ()>),

    /// Notification that a graph build has completed compilation.  Arguments
    /// are the originating context and the `GraphId` that was provided by
    /// script when `build()` was called.  Script-side consumers do not receive
    /// a `GraphInfo` because the ML thread retains the authoritative copy; new
    /// `MLGraph` objects created after compile start with an empty info.
    CompileResult(ContextId, GraphId),
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

    /// A new context has been created.
    NewContext(ContextId),

    /// A context was destroyed and may be freed.
    DestroyContext(ContextId),

    /// Request to create a tensor for `context_id`.
    ///
    /// Arguments: callback, context id, tensor id (script-side u32), byte length to allocate.
    /// The worker thread should allocate a Vec<u8> of the given length, store it keyed by
    /// (context_id, tensor_id), and invoke `callback` with a `ContextMessage::CreateTensorResult`
    /// containing the `ContextId` and `tensor_id` and Ok(()) / Err(()) to indicate
    /// success or failure. Using `ContextMessage` lets the same persistent ML-level
    /// callback be reused for other context-level replies in the future.
    CreateTensor(GenericCallback<ContextMessage>, ContextId, u32, usize),

    /// Request to create a tensor and initialize it with the provided
    /// bytes in one step.  This is used by `MLGraphBuilder.constant()`; the
    /// caller only needs an id and does not wait for a reply.
    ///
    /// Arguments: context id, tensor id, initial bytes for the tensor.
    CreateConstantTensor(ContextId, u32, Vec<u8>),

    /// Request to return the bytes for the tensor identified by
    /// `(context_id, tensor_id)`. The worker thread should look up its stored buffer
    /// and invoke `callback` with `ContextMessage::ReadTensorResult` containing
    /// the copied bytes (or an error).
    ReadTensor(GenericCallback<ContextMessage>, ContextId, u32),

    /// Request to write bytes into an existing tensor buffer.
    /// Arguments: ContextId, tensor id, bytes to write.
    WriteTensor(ContextId, u32, Vec<u8>),

    /// Dispatch a graph execution request to the ML worker thread.
    ///
    /// Arguments:
    /// - ContextId: originating context
    /// - GraphId: identifier for the graph; the worker thread uses this as the cache
    ///   key and looks up the associated `GraphInfo` from its internal cache.
    /// - Vec<(operand_id, tensor_id)>: mapping of graph input operand ids -> worker thread tensor ids
    /// - Vec<(operand_id, tensor_id)>: mapping of graph output operand ids -> worker thread tensor ids
    Dispatch(ContextId, GraphId, Vec<(u32, u32)>, Vec<(u32, u32)>),

    /// Response from the ML worker thread containing output tensor bytes for a
    /// previously-dispatched graph.  The context id allows the manager thread
    /// to look up the correct `Context` instance and merge the results into
    /// its tensor store.
    ComputeResult(ContextId, HashMap<u32, Vec<u8>>),

    /// Request the ML worker thread to asynchronously compile.
    Compile(
        GenericCallback<ContextMessage>,
        GraphId,
        ContextId,
        GraphInfo,
    ),

    /// Response from the ML worker thread signalling that a prior `Compile` request has
    /// finished successfully.  The `GraphId` corresponds to the identifier that
    /// was supplied by script.  Contexts use this to unblock queued dispatches.
    Compiled(ContextId, GraphId, PathBuf),

    /// Response from ML the worker thread signalling that a prior `Compile` request has
    /// failed. Arguments are the originating context, the graph id, and a
    /// human-readable error description.  The manager will unblock the
    /// timeline and propagate the failure to any waiting script build
    /// callbacks, allowing the upstream code to reject the associated promise.
    CompileFailed(ContextId, GraphId, String),
}

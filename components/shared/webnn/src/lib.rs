//! Shared WebNN message types used across the process boundary.

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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WebNNMsg {
    /// Graceful shutdown.
    Exit,

    /// Inform the backend that a new context has been created.
    NewContext(ContextId),

    /// Inform the backend that a context was destroyed and may be freed.
    DestroyContext(ContextId),
}

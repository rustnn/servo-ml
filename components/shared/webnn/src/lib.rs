//! Shared WebNN message types used across the process boundary.

use serde::{Deserialize, Serialize};

/// Messages addressed to the WebNN manager.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WebNNMsg {
    /// Graceful shutdown.
    Exit,
}

//! Error types for patchwork.

use thiserror::Error;

/// Errors that can occur during patchwork operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Error from the underlying ACP connection.
    #[error("connection error: {0}")]
    Connection(#[from] agent_client_protocol::Error),

    /// Error deserializing the LLM's response.
    #[error("failed to deserialize response: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// The LLM did not call return_result.
    #[error("LLM did not return a result")]
    NoResult,

    /// Error from a tool invocation.
    #[error("tool error: {0}")]
    Tool(String),

    /// The connection was closed before we could get the context.
    #[error("connection closed")]
    ConnectionClosed,
}

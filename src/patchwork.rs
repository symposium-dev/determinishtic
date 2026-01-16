//! The main Patchwork struct that wraps a Component.

use sacp::{
    ClientToAgent, Component, JrConnectionCx, NullResponder,
    link::AgentToClient,
    schema::{InitializeRequest, InitializeResponse, ProtocolVersion},
};
use sacp_conductor::{AgentOnly, Conductor, McpBridgeMode};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument};

use crate::ThinkBuilder;

/// The main entry point for patchwork operations.
///
/// Wraps a sacp [`Component`] and provides the [`think`](Self::think) method
/// for creating LLM-powered reasoning blocks.
///
/// The connection runs in a background task and is cancelled when `Patchwork`
/// is dropped.
pub struct Patchwork {
    cx: JrConnectionCx<ClientToAgent>,
    task: JoinHandle<Result<(), sacp::Error>>,
}

impl Patchwork {
    /// Create a new Patchwork instance from a sacp Component.
    ///
    /// This spawns a background task to run the connection to the agent.
    /// The component will be used to communicate with an LLM agent.
    #[instrument(name = "Patchwork::new", skip_all)]
    pub async fn new(
        component: impl Component<AgentToClient> + 'static,
    ) -> Result<Self, crate::Error> {
        debug!("spawning connection task");
        let (tx, rx) = oneshot::channel();

        let task = tokio::spawn(async move {
            ClientToAgent::builder()
                .with_spawned(|cx| async move {
                    // Send the connection context back to the caller
                    let _ = tx.send(cx);
                    // Keep running until the connection closes
                    std::future::pending::<Result<(), sacp::Error>>().await
                })
                .serve(Conductor::new_agent(
                    "patchwork-conductor",
                    AgentOnly(component),
                    McpBridgeMode::default(),
                ))
                .await
        });

        let cx = rx.await.map_err(|_| crate::Error::ConnectionClosed)?;
        info!("connection established");

        // FIXME: we should check that it supports MCP-over-ACP
        let InitializeResponse { .. } = cx
            .send_request(InitializeRequest::new(ProtocolVersion::LATEST))
            .block_task()
            .await?;

        Ok(Self { cx, task })
    }

    /// Start building a think block.
    ///
    /// Returns a [`ThinkBuilder`] that can be used to compose the prompt
    /// and register tools. The builder is consumed when awaited.
    pub fn think<'bound, Output>(&self) -> ThinkBuilder<'bound, NullResponder, Output>
    where
        Output: Send + JsonSchema + DeserializeOwned + 'static,
    {
        ThinkBuilder::new(self.cx.clone())
    }
}

impl Drop for Patchwork {
    fn drop(&mut self) {
        self.task.abort();
    }
}

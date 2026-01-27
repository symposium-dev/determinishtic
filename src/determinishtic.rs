//! The main Determinishtic struct that wraps a ConnectTo component.

use sacp::{
    Agent, Client, ConnectionTo, ConnectTo,
    schema::{InitializeRequest, InitializeResponse, ProtocolVersion},
};
use sacp_conductor::{AgentOnly, ConductorImpl, McpBridgeMode};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument};

use crate::ThinkBuilder;

/// The main entry point for determinishtic operations.
///
/// Wraps a sacp [`ConnectTo`] component and provides the [`think`](Self::think) method
/// for creating LLM-powered reasoning blocks.
///
/// The connection runs in a background task and is cancelled when `Determinishtic`
/// is dropped.
pub struct Determinishtic {
    cx: ConnectionTo<Agent>,
    task: JoinHandle<Result<(), sacp::Error>>,
}

impl Determinishtic {
    /// Create a new Determinishtic instance from a sacp ConnectTo component.
    ///
    /// This spawns a background task to run the connection to the agent.
    /// The component will be used to communicate with an LLM agent.
    #[instrument(name = "Determinishtic::new", skip_all)]
    pub async fn new(
        component: impl ConnectTo<Client> + 'static,
    ) -> Result<Self, crate::Error> {
        debug!("spawning connection task");
        let (tx, rx) = oneshot::channel();

        let task = tokio::spawn(async move {
            Client
                .builder()
                .with_spawned(|cx| async move {
                    // Send the connection context back to the caller
                    let _ = tx.send(cx);
                    // Keep running until the connection closes
                    std::future::pending::<Result<(), sacp::Error>>().await
                })
                .connect_to(ConductorImpl::new_agent(
                    "determinishtic-conductor",
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
    pub fn think<'bound, Output>(&self) -> ThinkBuilder<'bound, Output>
    where
        Output: Send + JsonSchema + DeserializeOwned + 'static,
    {
        ThinkBuilder::new(self.cx.clone())
    }
}

impl Drop for Determinishtic {
    fn drop(&mut self) {
        self.task.abort();
    }
}

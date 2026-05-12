//! The main Determinishtic struct that wraps a ConnectTo component.

use agent_client_protocol::{
    Agent, Client, ConnectionTo, ConnectTo,
    role::{HasPeer, Role},
    schema::{InitializeRequest, InitializeResponse, ProtocolVersion},
};
use agent_client_protocol_conductor::{AgentOnly, ConductorImpl, McpBridgeMode};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument};

use std::sync::Arc;
use crate::think::ThinkObserver;
use crate::ThinkBuilder;

/// The main entry point for determinishtic operations.
///
/// Wraps an agent-client-protocol [`ConnectionTo`] component and provides the [`think`](Self::think) method
/// for creating LLM-powered reasoning blocks.
///
/// The type parameter `R` is the role whose connection we hold. It defaults to `Agent`
/// for backwards compatibility with code that creates its own connection via [`new`](Self::new).
/// When using an existing connection (e.g., from inside a proxy), use [`from_connection`](Self::from_connection)
/// which accepts any role that can communicate with an agent.
///
/// When created via `new`, the connection runs in a background task and is cancelled
/// when `Determinishtic` is dropped.
pub struct Determinishtic<R: Role = Agent>
where
    R: HasPeer<Agent>,
{
    cx: ConnectionTo<R>,
    task: Option<JoinHandle<Result<(), agent_client_protocol::Error>>>,
    observer: Option<Arc<dyn ThinkObserver>>,
}

impl<R: Role> Determinishtic<R>
where
    R: HasPeer<Agent>,
{
    /// Create from an existing connection.
    ///
    /// Use this when you already have a connection to an agent (e.g., from inside
    /// an ACP proxy or MCP tool handler). The connection is borrowed - no background
    /// task is spawned, and the connection lifecycle is managed by the caller.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Inside an MCP tool handler
    /// async fn my_tool(cx: McpConnectionTo<Conductor>) -> Result<Output, Error> {
    ///     let d = Determinishtic::from_connection(cx.connection_to());
    ///     let result: MyOutput = d.think()
    ///         .text("Do something")
    ///         .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn from_connection(cx: ConnectionTo<R>) -> Self {
        Self { cx, task: None, observer: None }
    }

    /// Start building a think block.
    ///
    /// Returns a [`ThinkBuilder`] that can be used to compose the prompt
    /// and register tools. The builder is consumed when awaited.
    pub fn think<'bound, Output>(&self) -> ThinkBuilder<'bound, Output, R>
    where
        Output: Send + JsonSchema + DeserializeOwned + 'static,
    {
        ThinkBuilder::new(self.cx.clone(), self.observer.clone())
    }

    /// Attach an observer that will receive all session updates
    /// from every `think()` call made through this instance.
    pub fn set_observer(&mut self, observer: Arc<dyn ThinkObserver>) {
        self.observer = Some(observer);
    }
}

impl Determinishtic<Agent> {
    /// Create a new Determinishtic instance from an agent-client-protocol ConnectTo component.
    ///
    /// This spawns a background task to run the connection to the agent.
    /// The component will be used to communicate with an LLM agent.
    ///
    /// For use inside proxies where you already have a connection, use
    /// [`from_connection`](Self::from_connection) instead.
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
                    std::future::pending::<Result<(), agent_client_protocol::Error>>().await
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

        Ok(Self { cx, task: Some(task), observer: None })
    }
}

impl<R: Role> Drop for Determinishtic<R>
where
    R: HasPeer<Agent>,
{
    fn drop(&mut self) {
        if let Some(ref task) = self.task {
            task.abort();
        }
    }
}

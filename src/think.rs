//! ThinkBuilder for composing prompts with tools.

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use sacp::mcp_server::{McpConnectionTo, McpServer, McpServerBuilder};
use sacp::schema::{
    PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionNotification,
};
use sacp::util::MatchDispatch;
use sacp::{Agent, BoxFuture, ConnectionTo, NullRun, RunWithConnectionTo};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, info, trace, warn};

use crate::Error;

/// Builder for composing LLM prompts with embedded tools.
///
/// Created via [`Determinishtic::think`](crate::Determinishtic::think).
///
/// The type parameter `Run` tracks the responder chain for registered tools,
/// allowing tools to capture references from the stack frame.
pub struct ThinkBuilder<'bound, Output, Run: RunWithConnectionTo<Agent> = NullRun> {
    cx: ConnectionTo<Agent>,
    segments: Vec<Segment>,
    server: McpServerBuilder<Agent, Run>,
    explicit_spacing: bool,
    phantom: PhantomData<fn(&'bound Run) -> Output>,
}

/// A segment of the prompt being built.
enum Segment {
    Text(String),
    ToolReference(String),
}

impl<'bound, Output> ThinkBuilder<'bound, Output, NullRun>
where
    Output: Send + JsonSchema + DeserializeOwned + 'static,
{
    pub(crate) fn new(cx: ConnectionTo<Agent>) -> Self {
        Self {
            cx,
            segments: Vec::new(),
            server: McpServer::builder("patchwork".to_string())
                .instructions("You have access to tools. Call return_result when done."),
            explicit_spacing: false,
            phantom: PhantomData,
        }
        .textln("Please complete the following task to the best of your ability,")
        .textln("No further instructions will be given,")
        .textln(
            "so do your best to interpret the instructions without further feedback from the user,",
        )
        .textln("making use of the tools you have available.")
        .textln("")
        .textln(
            "IMPORTANT: When complete, invoke the `return_result` tool with the requested result.",
        )
        .textln("")
    }
}

impl<'bound, Output, Run: RunWithConnectionTo<Agent>> ThinkBuilder<'bound, Output, Run>
where
    Output: Send + JsonSchema + DeserializeOwned + 'static,
{
    /// Add literal text to the prompt.
    pub fn text(mut self, text: &str) -> Self {
        self.segments.push(Segment::Text(text.to_string()));
        self
    }

    /// Add literal text to the prompt followed by a newline.
    pub fn textln(mut self, text: &str) -> Self {
        self.segments.push(Segment::Text(format!("{text}\n")));
        self
    }

    /// Interpolate a value using its [`Display`] implementation.
    pub fn display(mut self, value: &impl Display) -> Self {
        self.segments.push(Segment::Text(value.to_string()));
        self
    }

    /// Interpolate a value using its [`Debug`] implementation.
    ///
    /// Useful for paths, complex types, or when you want to see the
    /// debug representation.
    pub fn debug(mut self, value: &impl Debug) -> Self {
        self.segments.push(Segment::Text(format!("{:?}", value)));
        self
    }

    /// Disable automatic spacing between segments.
    ///
    /// By default, the builder inserts spaces between segments unless
    /// the next segment starts with punctuation. Call this to require
    /// explicit spacing.
    pub fn explicit_spacing(mut self) -> Self {
        self.explicit_spacing = true;
        self
    }

    /// Build the final prompt string with smart spacing.
    fn build_prompt(&self) -> String {
        let mut result = String::new();

        for (i, segment) in self.segments.iter().enumerate() {
            let text = match segment {
                Segment::Text(t) => t.as_str(),
                Segment::ToolReference(name) => {
                    // Tool references are embedded inline
                    result.push_str(&format!("<mcp_tool>{}</mcp_tool>", name));
                    continue;
                }
            };

            // Smart spacing: insert space before this segment if needed
            if !self.explicit_spacing && i > 0 && !result.is_empty() {
                let needs_space = !result.ends_with([' ', '\t', '\n', '(', '[', '{'])
                    && !text.starts_with(['.', ',', ':', ';', '!', '?']);
                if needs_space {
                    result.push(' ');
                }
            }

            result.push_str(text);
        }

        result
    }

    /// Register a tool and embed a reference to it in the prompt.
    ///
    /// The tool closure receives the input and an [`McpConnectionTo`], and returns
    /// the output. Both input and output must implement [`JsonSchema`] for
    /// the LLM to understand the expected types.
    ///
    /// Tools can capture references from the stack frame, enabling them to
    /// access and mutate local data during the session. Tool invocations are
    /// serialized (one at a time) because the closure is `AsyncFnMut`.
    ///
    /// Due to Rust compiler limitations, you must pass `sacp::tool_fn_mut!()`
    /// as the final argument.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = Mutex::new(Vec::new());
    /// patchwork.think()
    ///     .text("Process the data using")
    ///     .tool(
    ///         "transform",
    ///         "Transform the input data",
    ///         async |input: TransformInput, _cx| {
    ///             let output = transform_data(&input);
    ///             results.lock().unwrap().push(output.clone());
    ///             Ok(output)
    ///         },
    ///         sacp::tool_fn_mut!(),
    ///     )
    ///     .run()
    ///     .await?
    /// ```
    pub fn tool<I, O, F, H>(
        mut self,
        name: &str,
        description: &str,
        func: F,
        tool_future_hack: H,
    ) -> ThinkBuilder<'bound, Output, impl RunWithConnectionTo<Agent>>
    where
        I: JsonSchema + DeserializeOwned + Send + 'static,
        O: JsonSchema + Serialize + Send + 'static,
        F: AsyncFnMut(I, McpConnectionTo<Agent>) -> Result<O, sacp::Error> + Send,
        H: for<'a> Fn(
                &'a mut F,
                I,
                McpConnectionTo<Agent>,
            ) -> BoxFuture<'a, Result<O, sacp::Error>>
            + Send
            + 'static,
    {
        debug!(tool_name = name, "registering tool");
        self.segments.push(Segment::ToolReference(name.to_string()));
        ThinkBuilder {
            cx: self.cx,
            segments: self.segments,
            server: self
                .server
                .tool_fn_mut(name, description, func, tool_future_hack),
            explicit_spacing: self.explicit_spacing,
            phantom: PhantomData,
        }
    }

    /// Register a tool without embedding a reference in the prompt.
    ///
    /// Use this when you want the tool to be available but don't want to
    /// explicitly mention it at this point in the prompt.
    ///
    /// Due to Rust compiler limitations, you must pass `sacp::tool_fn_mut!()`
    /// as the final argument.
    pub fn define_tool<I, O, F, H>(
        self,
        name: &str,
        description: &str,
        func: F,
        tool_future_hack: H,
    ) -> ThinkBuilder<'bound, Output, impl RunWithConnectionTo<Agent>>
    where
        I: JsonSchema + DeserializeOwned + Send + 'static,
        O: JsonSchema + Serialize + Send + 'static,
        F: AsyncFnMut(I, McpConnectionTo<Agent>) -> Result<O, sacp::Error> + Send,
        H: for<'a> Fn(
                &'a mut F,
                I,
                McpConnectionTo<Agent>,
            ) -> BoxFuture<'a, Result<O, sacp::Error>>
            + Send
            + 'static,
    {
        debug!(tool_name = name, "defining tool (hidden from prompt)");
        ThinkBuilder {
            cx: self.cx,
            segments: self.segments,
            server: self
                .server
                .tool_fn_mut(name, description, func, tool_future_hack),
            explicit_spacing: self.explicit_spacing,
            phantom: PhantomData,
        }
    }

}

impl<'bound, Output, Run: RunWithConnectionTo<Agent>> IntoFuture for ThinkBuilder<'bound, Output, Run>
where
    Output: Send + JsonSchema + DeserializeOwned + 'static,
    Run: Send,
{
    type Output = Result<Output, Error>;

    type IntoFuture = BoxFuture<'bound, Result<Output, Error>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            // Build prompt before consuming server
            let prompt = self.build_prompt();
            let cx = self.cx;

            // Use a cell to store the result from the return_result tool
            let mut output: Option<Output> = None;

            // Add the return_result tool
            let server = self.server.tool_fn_mut(
                "return_result",
                "Return the final result. Call this when you have completed the task.",
                async |input: ReturnResultInput<Output>, _cx| {
                    debug!("return_result tool invoked");
                    output = Some(input.result);
                    Ok(ReturnResultOutput { success: true })
                },
                sacp::tool_fn_mut!(),
            );

            info!(prompt_len = prompt.len(), "executing think block");
            trace!(prompt = %prompt, "full prompt");

            // Create a session with the MCP server and run it
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));

            cx.build_session(&cwd)
                .with_mcp_server(server.build())?
                .block_task()
                .run_until(async |mut session| {
                    session.send_prompt(&prompt)?;
                    tracing::info!(?prompt, "sending prompt");

                    // Wait for updates until we get a stop reason
                    loop {
                        let update = session.read_update().await?;
                        trace!(?update, "received session update");
                        match update {
                            sacp::SessionMessage::StopReason(reason) => {
                                debug!(?reason, "session stopped");
                                break;
                            }
                            sacp::SessionMessage::SessionMessage(dispatch) => {
                                MatchDispatch::new(dispatch)
                                    .if_notification(async |notification: SessionNotification| {
                                        tracing::debug!(?notification, "received session notification");
                                        Ok(())
                                    })
                                    .await
                                    .if_request(
                                        async |request: RequestPermissionRequest, responder| {
                                            tracing::debug!(
                                                ?request,
                                                "received tool use permission request"
                                            );
                                            // approve all tool usage
                                            let option =
                                                request.options.iter().find(|o| match o.kind {
                                                    PermissionOptionKind::AllowOnce
                                                    | PermissionOptionKind::AllowAlways => true,
                                                    PermissionOptionKind::RejectOnce
                                                    | PermissionOptionKind::RejectAlways => false,
                                                    _ => false,
                                                });
                                            let outcome = option
                                                .map(|o| {
                                                    RequestPermissionOutcome::Selected(
                                                        SelectedPermissionOutcome::new(
                                                            o.option_id.clone(),
                                                        ),
                                                    )
                                                })
                                                .unwrap_or(RequestPermissionOutcome::Cancelled);
                                            responder.respond(RequestPermissionResponse::new(outcome))
                                        },
                                    )
                                    .await
                                    .otherwise_ignore()?
                            }
                            _ => continue,
                        }
                    }
                    Ok(())
                })
                .await?;

            if output.is_some() {
                info!("think block completed successfully");
            } else {
                warn!("think block completed but no result was returned");
            }

            output.ok_or(Error::NoResult)
        })
    }
}

/// Input schema for the return_result tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ReturnResultInput<T> {
    /// The result value to return.
    result: T,
}

/// Output schema for the return_result tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ReturnResultOutput {
    /// Whether the result was successfully recorded.
    success: bool,
}

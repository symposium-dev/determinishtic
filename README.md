# determinishtic

Blend deterministic Rust code with LLM-powered reasoning.

> *Hat tip to [Dave Herman](https://github.com/dherman) for the name.*

## Philosophy

**Do things deterministically that are deterministic.** File discovery, iteration, and I/O happen in Rust. Summarization, analysis, and judgment happen via the LLM.

```rust
use determinishtic::Determinishtic;
use agent_client_protocol_tokio::AcpAgent;

#[tokio::main]
async fn main() -> Result<(), determinishtic::Error> {
    let d = Determinishtic::new(AcpAgent::zed_claude_code()).await?;

    // Rust handles the deterministic parts
    let files = std::fs::read_dir("./docs")?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension() == Some("md".as_ref()))
        .collect::<Vec<_>>();

    for entry in files {
        let contents = std::fs::read_to_string(entry.path())?;

        // LLM handles the non-deterministic reasoning
        let summary: String = d.think()
            .text("Summarize in one sentence:")
            .display(&contents)
            .await?;

        println!("{}: {}", entry.path().display(), summary);
    }

    Ok(())
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
determinishtic = "0.1"
agent-client-protocol = "0.11"

[dev-dependencies]
agent-client-protocol-tokio = "0.11"
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
```

## Core Concepts

### `Determinishtic`

The main entry point. Wraps a connection to an LLM agent and provides the `think()` method.

```rust
// Connect to an agent
let d = Determinishtic::new(AcpAgent::zed_claude_code()).await?;

// Or use an existing connection (e.g., inside a proxy)
let d = Determinishtic::from_connection(cx.connection_to());
```

### `ThinkBuilder`

A builder for composing prompts with embedded tools. Created via `d.think()`.

```rust
let result: MyOutput = d.think()
    .text("Analyze this data:")
    .display(&data)
    .textln("")
    .text("Focus on trends and anomalies.")
    .await?;
```

The output type must implement `JsonSchema` and `Deserialize` - the LLM returns structured data by calling a `return_result` tool.

### Tools

Register tools that the LLM can call during reasoning:

```rust
use agent_client_protocol::tool_fn_mut;

let mut results = Vec::new();

let output: Summary = d.think()
    .text("Process each item using the provided tool")
    .tool(
        "process_item",
        "Process a single item and return the result",
        async |input: ItemInput, _cx| {
            let output = process(&input);
            results.push(output.clone());
            Ok(output)
        },
        tool_fn_mut!(),
    )
    .await?;
```

Tools can capture references from the stack frame - no `'static` requirement. The `tool_fn_mut!()` macro is required due to Rust async closure limitations.

- `.tool()` - Register a tool and mention it in the prompt
- `.define_tool()` - Register a tool without mentioning it in the prompt

## Examples

Run the summarize_docs example:

```bash
cargo run --example summarize_docs -- --agent claude-code ./docs
```

Available agents: `claude-code`, `gemini`, `codex`

## Documentation

See the [mdbook](./md) for detailed documentation and RFCs.

## License

MIT OR Apache-2.0

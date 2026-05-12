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

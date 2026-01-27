# determinishtic

An easy API to integrate deterministic computation with LLM-powered reasoning.

```rust
use determinishtic::Determinishtic;
use sacp_tokio::AcpAgent;

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

See the [mdbook](./md) for more details.

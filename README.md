# patchwork-rs

An easy API to integrate determinishtic computation with LLM-powered reasoning.

```rust
use patchwork::Patchwork;
use sacp_tokio::AcpAgent;

#[tokio::main]
async fn main() -> Result<(), patchwork::Error> {
    let patchwork = Patchwork::new(AcpAgent::zed_claude_code()).await?;

    // Rust handles the deterministic parts
    let files = std::fs::read_dir("./docs")?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension() == Some("md".as_ref()))
        .collect::<Vec<_>>();

    for entry in files {
        let contents = std::fs::read_to_string(entry.path())?;

        // LLM handles the non-deterministic reasoning
        let summary: String = patchwork.think()
            .text("Summarize in one sentence:")
            .display(&contents)
            .run()
            .await?;

        println!("{}: {}", entry.path().display(), summary);
    }

    Ok(())
}
```

See the [mdbook](./md) for more details.

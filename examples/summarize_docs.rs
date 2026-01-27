//! Example: Summarize all markdown files in a directory.
//!
//! This example demonstrates how to use determinishtic to blend deterministic
//! Rust code with LLM-powered reasoning. The Rust code handles file discovery
//! and reading (deterministic), while the LLM summarizes content (non-deterministic).
//!
//! Usage:
//!   cargo run --example summarize_docs -- --agent claude-code <directory>
//!   cargo run --example summarize_docs -- --agent gemini ./docs
//!   cargo run --example summarize_docs -- --agent codex ./docs

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use determinishtic::Determinishtic;
use sacp_tokio::AcpAgent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;
use walkdir::WalkDir;

/// Summarize markdown files in a directory using an LLM agent.
#[derive(Parser, Debug)]
#[command(name = "summarize_docs")]
#[command(about = "Summarize all markdown files in a directory using an LLM")]
struct Args {
    /// The LLM agent to use
    #[arg(short, long, value_enum, default_value = "claude-code")]
    agent: Agent,

    /// The directory containing markdown files to summarize
    #[arg(default_value = ".")]
    directory: PathBuf,
}

/// Available LLM agents
#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum Agent {
    /// Claude Code (via Zed Industries ACP bridge)
    ClaudeCode,
    /// Google Gemini CLI
    Gemini,
    /// OpenAI Codex (via Zed Industries ACP bridge)
    Codex,
}

impl Agent {
    fn to_acp_agent(self) -> AcpAgent {
        match self {
            Agent::ClaudeCode => AcpAgent::zed_claude_code(),
            Agent::Gemini => AcpAgent::google_gemini(),
            Agent::Codex => AcpAgent::zed_codex(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FileSummary {
    /// A one-line summary of the file
    summary: String,
    /// Key topics or concepts covered
    topics: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber with env filter
    // Use RUST_LOG=determinishtic=debug or RUST_LOG=determinishtic=trace for more output
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    println!("Agent: {:?}", args.agent);
    println!("Directory: {}", args.directory.display());

    // Deterministic: Find all markdown files using walkdir
    let md_files: Vec<PathBuf> = WalkDir::new(&args.directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("\nFound {} markdown files", md_files.len());

    if md_files.is_empty() {
        println!("No markdown files found.");
        return Ok(());
    }

    // Create the determinishtic instance connected to the agent
    let agent = args.agent.to_acp_agent();
    let d = Determinishtic::new(agent).await?;

    // Deterministic loop, LLM-powered summarization
    let mut summaries = Vec::new();
    for path in &md_files {
        println!("\nSummarizing: {}", path.display());

        // Deterministic: Read the file
        let contents = std::fs::read_to_string(path)?;

        // LLM-powered: Summarize the contents
        let summary: FileSummary = d
            .think()
            .text("Summarize this markdown file in one sentence and list the key topics:")
            .text("\n\n")
            .display(&contents)
            .await?;

        println!("  Summary: {}", summary.summary);
        println!("  Topics: {}", summary.topics.join(", "));

        summaries.push((path.clone(), summary));
    }

    // Print final report
    println!("\n=== Summary Report ===\n");
    for (path, summary) in &summaries {
        println!("{}", path.display());
        println!("  {}", summary.summary);
        println!("  Topics: {}", summary.topics.join(", "));
        println!();
    }

    Ok(())
}

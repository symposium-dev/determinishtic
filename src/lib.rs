//! Determinishtic: Blend deterministic Rust code with LLM-powered reasoning.
//!
//! This library provides a builder-style API for constructing prompts that can
//! invoke Rust closures as MCP tools, enabling seamless interleaving of structured
//! code and natural language processing.
//!
//! # Example
//!
//! ```rust,ignore
//! use determinishtic::Determinishtic;
//! use sacp::ConnectTo;
//!
//! let d = Determinishtic::new(component).await?;
//!
//! let name = "Alice";
//! let result: String = d.think()
//!     .text("Say hello to")
//!     .display(&name)
//!     .text("in a friendly way.")
//!     .await?;
//! ```

mod determinishtic;
mod error;
mod think;

pub use determinishtic::Determinishtic;
pub use error::Error;
pub use think::ThinkBuilder;

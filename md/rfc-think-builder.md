# RFC: ThinkBuilder API

## Summary

Patchwork-rs is a Rust library for blending deterministic programming with LLM-powered reasoning. It provides a builder-style API for constructing prompts that can invoke Rust closures as MCP tools, enabling seamless interleaving of structured code and natural language processing.

## Motivation

Modern applications increasingly benefit from LLM capabilities, but integrating them into typed, deterministic codebases is awkward. You either:

1. **String templates** - Lose type safety, no compile-time checking of interpolations
2. **Separate prompt files** - Context switch between code and prompts, hard to pass runtime values
3. **Framework lock-in** - Heavy abstractions that obscure what's actually happening

Patchwork-rs takes a different approach: LLM interactions are first-class Rust expressions. The `think` builder composes prompts programmatically while allowing the LLM to call back into typed Rust closures via MCP tools.

This is inspired by the [Patchwork programming language](https://github.com/patchwork-lang/patchwork), which pioneered the idea of `think` blocks that blend imperative code with LLM reasoning.

### Philosophy: Deterministic Code, Non-Deterministic Reasoning

A core principle of patchwork is that **deterministic operations belong in Rust code, while non-deterministic reasoning goes to the LLM**. File I/O, iteration, data transformation—these happen in your Rust code. Summarization, analysis, judgment—these happen via `think()`.

```rust
// Deterministic: Rust finds and reads files
let files: Vec<PathBuf> = WalkDir::new(&directory)
    .into_iter()
    .filter_map(|e| e.ok())
    .filter(|e| e.path().extension() == Some("md".as_ref()))
    .map(|e| e.path().to_path_buf())
    .collect();

// Deterministic loop, LLM-powered summarization
for path in &files {
    let contents = std::fs::read_to_string(path)?;
    
    // Non-deterministic: LLM reasons about content
    let summary: Summary = patchwork.think()
        .text("Summarize this file:")
        .display(&contents)
        .run()
        .await?;
}
```

## Guide-level design

### Basic usage

```rust
use patchwork::Patchwork;
use sacp_tokio::AcpAgent;

#[tokio::main]
async fn main() -> Result<(), patchwork::Error> {
    let agent = AcpAgent::zed_claude_code();
    let patchwork = Patchwork::new(agent).await?;
    
    let name = "Alice";
    let result: String = patchwork.think()
        .text("Say hello to")
        .display(&name)
        .text("in a friendly way.")
        .run()
        .await?;
    
    println!("{}", result);  // "Hello Alice! Great to meet you!"
    Ok(())
}
```

### Composing prompts

The `ThinkBuilder` provides methods for building up prompts piece by piece:

- `.text("...")` - Add literal text
- `.textln("...")` - Add literal text followed by a newline
- `.display(&value)` - Interpolate a value using its `Display` impl
- `.debug(&value)` - Interpolate a value using its `Debug` impl (useful for paths, complex types)

```rust
let file_path = Path::new("data/input.txt");
let contents = std::fs::read_to_string(&file_path)?;

let summary: String = patchwork.think()
    .text("Summarize the following file")
    .debug(&file_path)
    .text(":\n\n")
    .display(&contents)
    .run()
    .await?;
```

### Smart spacing

By default, the builder automatically inserts spaces between segments to reduce visual noise. A space is inserted before a segment when:

- The previous segment didn't end with whitespace or opening brackets `(`, `[`, `{`

**Unless** the current segment begins with punctuation like `.`, `,`, `:`, `;`, `!`, `?`.

This means you can write:

```rust
.text("Hello,")
.display(&name)
.text(". How are you?")
```

And get `"Hello, Alice. How are you?"` — space auto-inserted before the name, but not before the period.

If you need precise control, disable smart spacing:

```rust
patchwork.think()
    .explicit_spacing()  // disable auto-spacing for this builder
    .text("No")
    .text("Spaces")
    .text("Here")
    // produces "NoSpacesHere"
```

### Tools: calling Rust from the LLM

The real power comes from `.tool()`, which registers a Rust closure as an MCP tool the LLM can invoke:

```rust
let result: String = patchwork.think()
    .text("Process the transcript and invoke")
    .tool(
        "rephrase",
        "Rephrase a mean-spirited phrase to be nicer",
        async |input: RephraseInput, _cx| {
            Ok(make_it_nicer(&input.phrase))
        },
        sacp::tool_fn_mut!(),
    )
    .text("on each mean-spirited phrase.")
    .run()
    .await?;
```

When you call `.tool(name, description, closure, sacp::tool_fn_mut!())`:
1. The closure is registered as an MCP tool with the given name and description
2. The text `<mcp_tool>name</mcp_tool>` is embedded in the prompt

The closure receives the tool input as its first argument, followed by an `McpContext<ClientToAgent>`. It returns `Result<O, sacp::Error>` where `O` is the output type.

**Important**: Due to Rust compiler limitations with async closures ([rust-lang/rust#109417](https://github.com/rust-lang/rust/issues/109417), [#110338](https://github.com/rust-lang/rust/issues/110338)), you must pass `sacp::tool_fn_mut!()` as the final argument. This macro generates a shim that helps the compiler understand the async closure's lifetime.

### Stack-local captures

Tools can capture mutable references from the enclosing stack frame, enabling powerful patterns like accumulating results:

```rust
let mut results = Vec::new();

let _: () = patchwork.think()
    .text("Process each item and record it")
    .tool(
        "record",
        "Record a processed item",
        async |input: RecordInput, _cx| {
            results.push(input.item);
            Ok(RecordOutput { success: true })
        },
        sacp::tool_fn_mut!(),
    )
    .run()
    .await?;

// After the think block, `results` contains all recorded items
println!("Recorded: {:?}", results);
```

This works because:
1. Patchwork uses `run_session()` internally, which avoids `'static` bounds on tool closures
2. Tools are `AsyncFnMut`, so invocations are serialized (one at a time), giving exclusive `&mut` access

### Defining tools without embedding

Sometimes you want to make a tool available without embedding a reference in the prompt at that point:

```rust
let result: String = patchwork.think()
    .text("Analyze the sentiment of each paragraph.")
    .text("Use the classify tool for ambiguous cases.")
    .define_tool(
        "classify",
        "Classify sentiment of ambiguous text",
        async |text: ClassifyInput, _cx| Ok(classify_sentiment(&text)),
        sacp::tool_fn_mut!(),
    )
    .tool(
        "summarize",
        "Summarize multiple paragraphs",
        async |paras: SummarizeInput, _cx| Ok(summarize_all(&paras)),
        sacp::tool_fn_mut!(),
    )
    .run()
    .await?;
```

Here `classify` is available but not explicitly referenced with `<mcp_tool>` tags—the prompt mentions it in natural language. The `summarize` tool is both defined and referenced.

### Structured output

The return type of `.run()` can be any type that implements `JsonSchema + DeserializeOwned`:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct Analysis {
    sentiment: String,
    confidence: f64,
    key_phrases: Vec<String>,
}

let analysis: Analysis = patchwork.think()
    .text("Analyze the sentiment of: ")
    .display(&text)
    .run()
    .await?;
```

The LLM is instructed to return its result by calling a `return_result` MCP tool with the appropriate JSON schema.

## Available agents

Patchwork works with any sacp `Component`. The `sacp-tokio` crate provides convenient constructors for common agents:

```rust
use sacp_tokio::AcpAgent;

// Claude Code via Zed Industries ACP bridge
let agent = AcpAgent::zed_claude_code();

// Google Gemini CLI
let agent = AcpAgent::google_gemini();

// OpenAI Codex via Zed Industries ACP bridge
let agent = AcpAgent::zed_codex();

let patchwork = Patchwork::new(agent).await?;
```

## Frequently asked questions

### Why a builder instead of a macro?

We plan to add a `think!` macro eventually, but starting with a builder has advantages:

1. **Easier to iterate** - Runtime API is simpler to evolve than proc-macro
2. **Better error messages** - Proc-macro errors are notoriously hard to debug
3. **Transparent** - You can see exactly what the builder does

The macro will likely expand to builder calls (or something equivalent).

### Why MCP tools?

We use MCP tools both for invoking user-defined closures and for returning results. The key advantage is that MCP tools provide an explicit, deterministic output structure—the LLM must call the `return_result` tool with JSON matching the expected schema. This avoids the need to parse free-form text output and ensures type safety end-to-end.

### How does the LLM know to return a result?

The `Patchwork` runtime automatically:

1. Adds a `return_result` MCP tool with a schema matching your expected output type
2. Includes instructions telling the LLM to call this tool when done
3. Waits for the tool call and deserializes the result

### What's with the `tool_fn_mut!()` macro?

The Rust compiler currently has limitations with async closures that capture references. The `sacp::tool_fn_mut!()` macro generates a shim that helps the compiler understand the relationship between the closure and its future. This is a workaround until async closures are fully stabilized in Rust.

See [rust-lang/rust#109417](https://github.com/rust-lang/rust/issues/109417) and [rust-lang/rust#110338](https://github.com/rust-lang/rust/issues/110338) for details.

### What about nested think blocks?

A tool closure can contain another `think()` call, enabling multi-agent patterns:

```rust
.tool(
    "deep_analysis",
    "Perform deep analysis of a topic",
    async |input: AnalysisInput, _cx| {
        let result = patchwork.think()
            .text("Provide deep analysis of:")
            .display(&input.topic)
            .run()
            .await?;
        Ok(result)
    },
    sacp::tool_fn_mut!(),
)
```

Nested `think()` calls just work—they create independent sessions.

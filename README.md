<div align="center">

# ReasonKit Web (Rust Edition)

**High-Performance Web Sensing & Browser Automation Layer**
**Rust-Native Implementation**

[![Crates.io](https://img.shields.io/crates/v/reasonkit-web?style=flat-square&color=%2306b6d4)](https://crates.io/crates/reasonkit-web)
[![docs.rs](https://img.shields.io/docsrs/reasonkit-web?style=flat-square&color=%2310b981)](https://docs.rs/reasonkit-web)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square&color=%23a855f7)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.74%2B-orange?style=flat-square&logo=rust&color=%23f97316)](https://www.rust-lang.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green?style=flat-square&color=%2310b981)](https://modelcontextprotocol.io)

*The Eyes and Ears of AI Reasoning - Now Blazingly Fast*

[Documentation](https://docs.rs/reasonkit-web) | [Crates.io](https://crates.io/crates/reasonkit-web) | [ReasonKit Core](https://github.com/ReasonKit/reasonkit-core) | [Website](https://reasonkit.sh)

</div>

---

> **Note:** This is the **Rust implementation** of the ReasonKit Web Sensing layer. It supersedes the legacy Python prototype for performance-critical deployments.

Web sensing and browser automation layer for ReasonKit. Implements the Model Context Protocol (MCP) for seamless web interactions with AI reasoning systems, powered by Rust and ChromiumOxide.

## Features

![ReasonKit Web Features](./brand/readme/features.png)

## Use Cases

![ReasonKit Web Use Cases](./brand/readme/use_cases.png)

## Performance

![ReasonKit Web Performance Benchmark](./brand/readme/performance_benchmark.png)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
reasonkit-web = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

### As a Library

```rust,ignore
use reasonkit_web::BrowserController;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create browser controller
    let controller = BrowserController::new().await?;

    // Navigate to a page
    let page = controller.navigate("https://example.com").await?;

    // Extract content
    let content = page.content().await?;
    println!("Page content: {}", content);

    Ok(())
}
```

### As an MCP Server

```bash
# Build the MCP server binary
cargo build --release

# Run the MCP server
./target/release/reasonkit-web

# Or with cargo
cargo run --release
```

### Content Extraction

```rust,ignore
use reasonkit_web::{ContentExtractor, MetadataExtractor};

// Extract structured content from HTML
let html = "<html><body><h1>Title</h1><p>Content</p></body></html>";
let extractor = ContentExtractor::new();
let content = extractor.extract(html)?;

// Extract metadata
let meta_extractor = MetadataExtractor::new();
let metadata = meta_extractor.extract(html)?;
```

### API Workflow

![ReasonKit Web API Workflow](./brand/readme/api_workflow.png)

![ReasonKit Web API Sequence](./brand/readme/api_flow.png)

## Architecture

The ReasonKit Web layer implements a high-performance web sensing architecture designed for AI reasoning systems:

### System Topology

![ReasonKit Web MCP Topology](./brand/readme/mcp_topology.png)

### Core Architecture

![ReasonKit Web Architecture Diagram](./brand/readme/architecture.png)

### The Rust Engine

![ReasonKit Web Rust Engine](./brand/readme/rust_engine.png)

### Key Design Principles

**Performance-First**: Built in Rust with async/await for maximum throughput
**Protocol-Driven**: Implements Model Context Protocol (MCP) for AI integration
**Modular Design**: Separates capture, extraction, and memory for flexibility
**Security-Focused**: Headless browser isolation and content sanitization

### Integration Flow

![ReasonKit Web Integration Flow](./brand/readme/integration_flow.png)

## Security & Privacy

ReasonKit Web is built with a "GDPR by Default" philosophy, ensuring that AI reasoning systems can interact with the web safely and compliantly.

![ReasonKit Web Browser Isolation Layer](./brand/readme/browser_isolation_layer.svg)
![ReasonKit Web Security Shield](./brand/readme/security_shield.png)

### Key Security Features
- **PII Redaction Engine**: Automatically detects and redacts sensitive information (emails, phone numbers, secrets) before it leaves the secure runtime.
- **Headless Isolation**: Browser sessions run in isolated containers with no persistent state.
- **Content Sanitization**: Malicious scripts and trackers are stripped at the DOM level.

For detailed technical specifications, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

</div>

## Technology Stack

![ReasonKit Web Technology Stack](./brand/readme/tech_stack.png)

## License

Apache License 2.0 - see [LICENSE](LICENSE)

---

<div align="center">

![ReasonKit Ecosystem Connection](./brand/readme/ecosystem_connection.png)

**Part of the ReasonKit Ecosystem**

[ReasonKit Core](https://github.com/ReasonKit/reasonkit-core) • [ReasonKit Mem](https://github.com/ReasonKit/reasonkit-mem) • [Website](https://reasonkit.sh)

*"See How Your AI Thinks"*

</div>

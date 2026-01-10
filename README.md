<div align="center">

# ReasonKit Web (Rust Edition)

**High-Performance Web Sensing & Browser Automation Layer**
**Rust-Native Implementation**

[![CI](https://img.shields.io/github/actions/workflow/status/reasonkit/reasonkit-web/ci.yml?branch=main&style=flat-square&logo=github&label=CI&color=06b6d4&logoColor=06b6d4)](https://github.com/reasonkit/reasonkit-web/actions/workflows/ci.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/reasonkit/reasonkit-web/security.yml?branch=main&style=flat-square&logo=github&label=Security&color=10b981&logoColor=10b981)](https://github.com/reasonkit/reasonkit-web/actions/workflows/security.yml)
[![Crates.io](https://img.shields.io/crates/v/reasonkit-web?style=flat-square&logo=rust&color=10b981&logoColor=f9fafb)](https://crates.io/crates/reasonkit-web)
[![docs.rs](https://img.shields.io/docsrs/reasonkit-web?style=flat-square&logo=docs.rs&color=06b6d4&logoColor=f9fafb)](https://docs.rs/reasonkit-web)
[![Downloads](https://img.shields.io/crates/d/reasonkit-web?style=flat-square&color=ec4899&logo=rust&logoColor=f9fafb)](https://crates.io/crates/reasonkit-web)
[![License](https://img.shields.io/badge/license-Apache%202.0-a855f7?style=flat-square&labelColor=030508)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-f97316?style=flat-square&logo=rust&logoColor=f9fafb)](https://www.rust-lang.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-10b981?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmOWZhZmIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTIgMnYyME0yIDEyaDIwIi8+PC9zdmc+)](https://modelcontextprotocol.io)

_The Eyes and Ears of AI Reasoning - Now Blazingly Fast_

[Documentation](https://docs.rs/reasonkit-web) | [Crates.io](https://crates.io/crates/reasonkit-web) | [ReasonKit Core](https://github.com/ReasonKit/reasonkit-core) | [Website](https://reasonkit.sh)

</div>

---

> **Note:** This is the **Rust implementation** of the ReasonKit Web Sensing layer. It supersedes the legacy Python prototype for performance-critical deployments.

Web sensing and browser automation layer for ReasonKit. Implements the Model Context Protocol (MCP) for seamless web interactions with AI reasoning systems, powered by Rust and ChromiumOxide.

## Features

![ReasonKit Web Features](https://raw.githubusercontent.com/Lenvanderhof/ReasonKit/main/reasonkit-web/brand/readme/features.png)

## Use Cases

![ReasonKit Web Use Cases](https://raw.githubusercontent.com/Lenvanderhof/ReasonKit/main/reasonkit-web/brand/readme/use_cases.png)

## Performance

![ReasonKit Web Performance Benchmark](https://raw.githubusercontent.com/Lenvanderhof/ReasonKit/main/reasonkit-web/brand/readme/performance_benchmark.png)

## Installation

### Universal Installer (Recommended)

**Installs all 4 ReasonKit projects together:**

```bash
curl -fsSL https://get.reasonkit.sh | bash -s -- --with-web
```

**Platform & Shell Support:**

- ✅ All platforms (Linux/macOS/Windows/WSL)
- ✅ All shells (Bash/Zsh/Fish/Nu/PowerShell/Elvish)
- ✅ Auto-detects shell and configures PATH
- ✅ Beautiful progress visualization

### Cargo (Rust Library)

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

## Version & Maturity

| Component              | Status    | Notes                                      |
| ---------------------- | --------- | ------------------------------------------ |
| **MCP Server**         | ✅ Stable | Model Context Protocol integration         |
| **CDP Client**         | ✅ Stable | Chrome DevTools Protocol via chromiumoxide |
| **Content Extraction** | ✅ Stable | HTML parsing, metadata extraction          |
| **PII Redaction**      | ✅ Stable | GDPR-compliant data handling               |
| **WASM Support**       | 🔶 Beta   | Browser-native execution                   |

**Current Version:** v0.1.3 | [CHANGELOG](CHANGELOG.md) | [Releases](https://github.com/reasonkit/reasonkit-web/releases)

### Verify Installation

```bash
# Build and run MCP server
cargo build --release
./target/release/reasonkit-web --help

# Or test as a library
cargo test
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)

---

<div align="center">

![ReasonKit Ecosystem Connection](./brand/readme/ecosystem_connection.png)

**Part of the ReasonKit Ecosystem**

[ReasonKit Core](https://github.com/reasonkit/reasonkit-core) • [ReasonKit Mem](https://github.com/reasonkit/reasonkit-mem) • [Website](https://reasonkit.sh)

_"See How Your AI Thinks"_

</div>

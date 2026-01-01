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

- **Headless Browser Automation** - Full browser control via ChromiumOxide (CDP)
- **MCP Server** - Model Context Protocol for AI integration
- **Web Capture** - Screenshot, PDF, and HTML capture
- **Content Extraction** - Intelligent content parsing and metadata extraction
- **High Performance** - Async Rust runtime (Tokio) for low-latency operations

## Quick Start

```bash
# Build the server
cargo build --release

# Run the MCP server
./target/release/reasonkit-web
```

## Architecture

The ReasonKit Web layer implements a high-performance web sensing architecture designed for AI reasoning systems:

![ReasonKit Web Architecture Diagram](./brand/readme-assets/architecture.png)

### Key Design Principles

**Performance-First**: Built in Rust with async/await for maximum throughput
**Protocol-Driven**: Implements Model Context Protocol (MCP) for AI integration
**Modular Design**: Separates capture, extraction, and memory for flexibility
**Security-Focused**: Headless browser isolation and content sanitization

### Integration Flow

1. **AI Agent** → **MCP Server** → Initiates web sensing operations
2. **Browser Controller** → Navigates and captures web content
3. **Extraction Engine** → Processes content for structured data
4. **Memory Integration** → Sends results to `reasonkit-mem` for triangulation

For detailed technical specifications, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

</div>

## Technology Stack

| Component         | Technology    | Purpose               |
| ----------------- | ------------- | --------------------- |
| **Browser**       | ChromiumOxide | Async Rust CDP client |
| **MCP Server**    | mcp-sdk-rs    | AI agent integration  |
| **Runtime**       | Tokio         | Async runtime         |
| **Serialization** | Serde         | JSON handling         |

## License

Apache License 2.0 - see [LICENSE](LICENSE)

---

<div align="center">

**Part of the ReasonKit Ecosystem**

[ReasonKit Core](https://github.com/ReasonKit/reasonkit-core) • [ReasonKit Mem](https://github.com/ReasonKit/reasonkit-mem) • [Website](https://reasonkit.sh)

*"See How Your AI Thinks"*

</div>

<div align="center">

# ReasonKit Web

**High-Performance Web Sensing & Browser Automation Layer**

[![Crates.io](https://img.shields.io/crates/v/reasonkit-web?style=flat-square&color=%2306b6d4)](https://crates.io/crates/reasonkit-web)
[![docs.rs](https://img.shields.io/docsrs/reasonkit-web?style=flat-square&color=%2310b981)](https://docs.rs/reasonkit-web)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square&color=%23a855f7)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.74%2B-orange?style=flat-square&logo=rust&color=%23f97316)](https://www.rust-lang.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green?style=flat-square&color=%2310b981)](https://modelcontextprotocol.io)

_The Eyes and Ears of AI Reasoning - Now Blazingly Fast_

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

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                      WEB SENSING LAYER (RUST)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   AI Agent ──▶ MCP Server ──▶ Browser Controller (CDP)         │
│                    │                │                           │
│                    ▼                ▼                           │
│              ┌──────────┐    ┌──────────────┐                   │
│              │ Capture  │    │ Extraction   │                   │
│              └────┬─────┘    └──────┬───────┘                   │
│                   │                 │                           │
│                   ▼                 ▼                           │
│            Screenshots        Content + Metadata                │
│            PDFs, HTML         Links, Structured Data            │
│                   │                 │                           │
│                   └────────┬────────┘                           │
│                            ▼                                    │
│                    reasonkit-mem (Triangulation)                │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** Content extraction happens in `reasonkit-web`. Multi-source triangulation and verification are delegated to `reasonkit-mem` via the PyO3 bridge. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

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

_"See How Your AI Thinks"_

</div>

# ReasonKit Web Architecture

## Web Sensing Layer (Rust)

The Web Sensing Layer provides high-performance browser automation and content extraction for AI reasoning systems.

### Architecture Diagram

```mermaid
flowchart LR
    subgraph WEB_SENSING_LAYER_RUST["WEB SENSING LAYER (RUST)"]
        A[AI Agent]
        B[MCP Server]
        C[Browser Controller<br/>CDP]

        A --> B --> C

        subgraph PIPELINE[""]
            D[Capture<br/>Screenshots / PDFs / HTML]
            E[Extraction<br/>Content + Metadata]
        end

        C --> D --> E
    end
    
    E -.->|delegates to| F[reasonkit-mem<br/>Triangulation]
```

**Note:** Content extraction happens in `reasonkit-web`. Multi-source triangulation and verification are handled by `reasonkit-mem` via the PyO3 bridge (see `TRIANGULATION_PROTOCOL_V2.md`).

### Component Flow

1. **AI Agent** → Sends requests via Model Context Protocol
2. **MCP Server** → Receives and routes requests to browser controller
3. **Browser Controller (CDP)** → Controls headless Chromium via Chrome DevTools Protocol
4. **Capture** → Captures screenshots, PDFs, and HTML from web pages
5. **Extraction** → Extracts main content, metadata, and structured data
6. **Triangulation** → (Delegated to `reasonkit-mem`) Verifies claims across multiple sources

### Technology Stack

| Component         | Technology    | Purpose               |
| ----------------- | ------------- | --------------------- |
| **Browser**       | ChromiumOxide | Async Rust CDP client |
| **MCP Server**    | mcp-sdk-rs    | AI agent integration  |
| **Runtime**       | Tokio         | Async runtime         |
| **Serialization** | Serde         | JSON handling         |

### Module Structure

```
reasonkit-web/
├── browser/          # Browser automation (CDP)
│   ├── controller.rs # Main browser controller
│   ├── capture.rs    # Screenshot, PDF, HTML capture
│   ├── navigation.rs # Page navigation
│   └── stealth.rs    # Anti-detection measures
├── extraction/       # Content extraction
│   ├── content.rs    # Main content extraction
│   ├── links.rs      # Link extraction
│   └── metadata.rs   # Metadata extraction
└── mcp/              # MCP server implementation
    ├── server.rs     # MCP server
    ├── tools.rs      # MCP tools (browser operations)
    └── types.rs      # MCP type definitions
```

### Data Flow

```
AI Agent Request
    ↓
MCP Server (JSON-RPC 2.0)
    ↓
Browser Controller (CDP)
    ↓
┌─────────────┬─────────────┐
│   Capture   │  Extraction │
└──────┬──────┴──────┬───────┘
       │             │
       ▼             ▼
Screenshots    Content + Metadata
PDFs, HTML    Links, Structured Data
       │             │
       └──────┬──────┘
              ↓
      reasonkit-mem (Triangulation)
              ↓
      Verified Facts
```

---

_"Designed, Not Dreamed" | https://reasonkit.sh_


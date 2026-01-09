# Architecture Diagrams

This document contains all architecture diagrams for ReasonKit Web in multiple formats.

## Mermaid Diagram (Markdown/Docs)

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

**Usage:** Copy this into any Markdown file. Most documentation platforms (GitHub, GitLab, Notion) support Mermaid rendering.

## SVG Architecture Tile

**File:** [`../assets/svg/web_sensing_layer.svg`](../assets/svg/web_sensing_layer.svg)

**Brand Compliance:**

- ✅ Colors: Cyan (#06b6d4), Purple (#a855f7), Pink (#ec4899), Green (#10b981)
- ✅ Fonts: Inter (labels), JetBrains Mono (code)
- ✅ Background: Void Black (#030508)
- ✅ Matches codebase architecture (Extraction, not Triangulate)

**Usage:**

- Inline in HTML: `<img src="assets/svg/web_sensing_layer.svg" alt="Web Sensing Layer Architecture">`
- Export to PNG for GitHub social previews
- Include in documentation sites
- Use as architecture tile in larger system diagrams

**Dimensions:** 960×320px (3:1 aspect ratio)

## ASCII Diagram (Terminal/Plain Text)

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

**Usage:** Copy into README files, terminal output, or plain text documentation.

---

## Architecture Notes

### Component Responsibilities

1. **AI Agent**: Initiates requests for web sensing operations
2. **MCP Server**: Receives JSON-RPC 2.0 requests and routes to browser controller
3. **Browser Controller**: Controls headless Chromium via Chrome DevTools Protocol
4. **Capture**: Captures visual and structural data (screenshots, PDFs, HTML)
5. **Extraction**: Extracts semantic content, metadata, and structured data
6. **ReasonKit-mem**: (External) Performs multi-source triangulation and verification

### Key Design Decisions

- **Separation of Concerns**: `reasonkit-web` handles I/O (browser automation, capture, extraction). `reasonkit-mem` handles compute (embedding, similarity, verification).
- **Rust-First**: All performance-critical paths are in Rust (CONS-005 compliance).
- **MCP Protocol**: Standard Model Context Protocol for AI agent integration.
- **Delegation Pattern**: Triangulation is delegated to `reasonkit-mem` via PyO3 bridge (see `TRIANGULATION_PROTOCOL_V2.md`).

---

_"Designed, Not Dreamed" | <https://reasonkit.sh>_

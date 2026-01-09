# ReasonKit-WEB PROJECT CONTEXT

> MCP Sidecar for Web Sensing | Pure Rust Implementation
> "Designed, Not Dreamed" | Browser Capture Without Node.js

**LICENSE:** Apache 2.0 (Open Source)
**STATUS:** Active Development
**WEBSITE:** <https://reasonkit.sh/docs/web>

---

## PROJECT IDENTITY

```
+---------------------------------------------------------------------------+
|                       REASONKIT-WEB STATUS                                |
+---------------------------------------------------------------------------+
|  Name:        reasonkit-web                                               |
|  Role:        MCP Sidecar for Web Sensing                                 |
|  Status:      Active Development                                          |
|  License:     Apache 2.0                                                  |
|  Language:    Rust 1.74+                                                  |
|  Framework:   Axum 0.7                                                    |
+---------------------------------------------------------------------------+
```

### Purpose

ReasonKit-web is a **pure Rust MCP (Model Context Protocol) sidecar** that provides browser capture and web sensing capabilities. It runs alongside AI agents to enable:

- **Browser Content Capture** - Extract and process web page content
- **DOM Analysis** - Parse and analyze document structure
- **Screenshot Capture** - Visual snapshots of web pages
- **MCP Protocol Compliance** - Standard MCP server interface

---

## TECHNOLOGY STACK

| Component         | Technology | Version | Purpose                       |
| ----------------- | ---------- | ------- | ----------------------------- |
| **Language**      | Rust       | 1.74+   | Core implementation           |
| **Framework**     | Axum       | 0.7     | HTTP server framework         |
| **Async**         | Tokio      | 1.x     | Async runtime                 |
| **Serialization** | Serde      | 1.x     | JSON serialization            |
| **HTTP Client**   | Reqwest    | 0.11+   | Outbound HTTP requests        |
| **WebDriver**     | thirtyfour | 0.31+   | Browser automation (optional) |
| **HTML Parse**    | scraper    | 0.18+   | DOM parsing                   |

---

## HARD CONSTRAINTS (NEVER VIOLATE)

| ID          | Constraint                | Enforcement | Consequence             |
| ----------- | ------------------------- | ----------- | ----------------------- |
| **WEB-001** | NO NODE.JS                | HARD        | Reject at code review   |
| **WEB-002** | Localhost Binding Default | HARD        | Security requirement    |
| **WEB-003** | No Hardcoded Secrets      | HARD        | Security incident       |
| **WEB-004** | Token Auth Required       | HARD        | All endpoints protected |
| **WEB-005** | CONS-003 Compliance       | HARD        | Environment variables   |
| **WEB-006** | Memory Safety             | HARD        | No unsafe without doc   |

### Security Requirements

```yaml
BINDING:
  default: "127.0.0.1:8765"
  remote_allowed: false # Must be explicitly enabled

AUTHENTICATION:
  method: "Bearer Token"
  token_env: "REASONKIT_WEB_TOKEN"
  token_required: true

SECRETS:
  - NEVER hardcode API keys
  - NEVER hardcode tokens
  - ALWAYS use environment variables
  - ALWAYS use .env files (gitignored)
```

---

## PROJECT STRUCTURE

```
reasonkit-web/
+-- Cargo.toml              # Crate manifest
+-- Cargo.lock              # Dependency lockfile
+-- src/
|   +-- main.rs             # Server entry point
|   +-- lib.rs              # Library exports
|   +-- mcp_protocol.rs     # MCP types and protocol
|   +-- handlers/           # Request handlers
|   |   +-- mod.rs
|   |   +-- capture.rs      # Browser capture endpoints
|   |   +-- analyze.rs      # DOM analysis endpoints
|   |   +-- screenshot.rs   # Screenshot endpoints
|   |   +-- health.rs       # Health check endpoint
|   +-- security.rs         # Security middleware
|   +-- config.rs           # Configuration management
|   +-- error.rs            # Error types
|   +-- browser/            # Browser automation
|   |   +-- mod.rs
|   |   +-- driver.rs       # WebDriver management
|   |   +-- capture.rs      # Content capture
|   +-- parser/             # HTML/DOM parsing
|       +-- mod.rs
|       +-- dom.rs          # DOM utilities
|       +-- extract.rs      # Content extraction
+-- tests/
|   +-- integration/        # Integration tests
|   +-- unit/               # Unit tests
+-- config/
|   +-- default.toml        # Default configuration
+-- scripts/
|   +-- install.sh          # Installation script
+-- systemd/
    +-- reasonkit-web.service  # Systemd unit file
```

---

## KEY FILES

| File                    | Purpose                              |
| ----------------------- | ------------------------------------ |
| `src/main.rs`           | Server entry point, Axum router      |
| `src/mcp_protocol.rs`   | MCP types and message definitions    |
| `src/handlers/`         | Request handlers for each endpoint   |
| `src/security.rs`       | Token validation middleware          |
| `src/config.rs`         | Configuration loading and validation |
| `src/browser/driver.rs` | WebDriver lifecycle management       |
| `Cargo.toml`            | Dependencies and crate metadata      |

---

## DEVELOPMENT COMMANDS

### Build and Test

```bash
# Build release binary
cargo build --release

# Run all tests
cargo test

# Run linter (MUST pass before merge)
cargo clippy -- -D warnings

# Format code
cargo fmt

# Check formatting (CI gate)
cargo fmt --check
```

### Run Server

```bash
# Set required token
export REASONKIT_WEB_TOKEN="your-secure-token-here"

# Run development server
cargo run

# Run with debug logging
RUST_LOG=debug cargo run

# Run release build
REASONKIT_WEB_TOKEN=your-token ./target/release/reasonkit-web
```

### Quality Gates (CONS-009)

All 5 gates MUST pass before merge:

```bash
# Gate 1: Build
cargo build --release

# Gate 2: Lint
cargo clippy -- -D warnings

# Gate 3: Format
cargo fmt --check

# Gate 4: Test
cargo test --all-features

# Gate 5: Bench (if applicable)
cargo bench
```

---

## DEPLOYMENT

### Systemd Service

**Service File:** `/etc/systemd/system/reasonkit-web.service`

```ini
[Unit]
Description=ReasonKit Web Sensing Sidecar
After=network.target

[Service]
Type=simple
User=reasonkit
Group=reasonkit
EnvironmentFile=/etc/reasonkit/reasonkit-web.env
ExecStart=/usr/local/bin/reasonkit-web
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Environment File:** `/etc/reasonkit/reasonkit-web.env`

```bash
REASONKIT_WEB_TOKEN=your-production-token
REASONKIT_WEB_HOST=127.0.0.1
REASONKIT_WEB_PORT=8765
RUST_LOG=info
```

### Service Management

```bash
# Enable service
sudo systemctl enable reasonkit-web

# Start service
sudo systemctl start reasonkit-web

# Check status
sudo systemctl status reasonkit-web

# View logs
journalctl -u reasonkit-web -f
```

---

## MCP PROTOCOL

### Supported Methods

| Method           | Description              | Status      |
| ---------------- | ------------------------ | ----------- |
| `tools/list`     | List available tools     | Implemented |
| `tools/call`     | Execute a tool           | Implemented |
| `resources/list` | List available resources | Planned     |
| `resources/read` | Read a resource          | Planned     |

### Available Tools

| Tool            | Description                     | Parameters                |
| --------------- | ------------------------------- | ------------------------- | ---- |
| `capture_page`  | Capture web page content        | `url: string`             |
| `analyze_dom`   | Analyze DOM structure           | `url: string`             |
| `screenshot`    | Take page screenshot            | `url: string, format: png | jpg` |
| `extract_links` | Extract all links from page     | `url: string`             |
| `extract_text`  | Extract readable text from page | `url: string`             |

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "capture_page",
    "arguments": {
      "url": "https://example.com"
    }
  },
  "id": 1
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "<page content>"
      }
    ]
  },
  "id": 1
}
```

---

## TESTING

### Unit Tests

Located in each module with `#[cfg(test)]` blocks:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_validation() {
        // Test cases
    }
}
```

### Integration Tests

Located in `tests/` directory:

```
tests/
+-- integration/
|   +-- capture_test.rs     # Page capture tests
|   +-- mcp_protocol_test.rs # Protocol compliance
|   +-- security_test.rs    # Auth tests
+-- common/
    +-- mod.rs              # Test utilities
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_capture_page

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test '*'
```

---

## TASKWARRIOR INTEGRATION

All work MUST be tracked in Taskwarrior (CONS-007):

```bash
# Create task
task add project:rk-project.web-rs.{component} "{description}" priority:{H|M|L}

# Components:
# - rk-project.web-rs.core      -> Core server implementation
# - rk-project.web-rs.handlers  -> Request handlers
# - rk-project.web-rs.security  -> Security features
# - rk-project.web-rs.browser   -> Browser automation
# - rk-project.web-rs.tests     -> Test coverage
# - rk-project.web-rs.docs      -> Documentation

# Start work
task {id} start

# Complete
task {id} done
```

---

## ERROR HANDLING

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    #[error("Authentication failed: {0}")]
    AuthError(String),

    #[error("Browser error: {0}")]
    BrowserError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}
```

### Error Responses

All errors return proper MCP error format:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32600,
    "message": "Invalid request"
  },
  "id": 1
}
```

---

## CONFIGURATION

### Environment Variables

| Variable              | Required | Default     | Description          |
| --------------------- | -------- | ----------- | -------------------- |
| `REASONKIT_WEB_TOKEN` | Yes      | -           | Authentication token |
| `REASONKIT_WEB_HOST`  | No       | `127.0.0.1` | Bind address         |
| `REASONKIT_WEB_PORT`  | No       | `8765`      | Listen port          |
| `RUST_LOG`            | No       | `info`      | Log level            |
| `WEBDRIVER_URL`       | No       | -           | WebDriver endpoint   |

### Config File (config/default.toml)

```toml
[server]
host = "127.0.0.1"
port = 8765
timeout_seconds = 30

[security]
token_required = true
max_request_size = "10MB"

[browser]
headless = true
timeout_seconds = 30

[logging]
level = "info"
format = "json"
```

---

## RELATIONSHIP TO OTHER PROJECTS

| Project            | Relationship                               |
| ------------------ | ------------------------------------------ |
| **ReasonKit-core** | Uses as Rust library for ThinkTool support |
| **ReasonKit-web**  | Python counterpart (MCP protocol bridge)   |
| **ReasonKit-mem**  | Can store captured content                 |
| **ReasonKit-pro**  | Extended browser capabilities              |

---

## CRITICAL REMINDERS

1. **NO NODE.JS** - This is a pure Rust implementation (WEB-001)
2. **LOCALHOST ONLY** - Default binding is 127.0.0.1 (WEB-002)
3. **NO HARDCODED SECRETS** - Use environment variables (WEB-003, CONS-003)
4. **TOKEN AUTH REQUIRED** - All endpoints require authentication (WEB-004)
5. **Rust-FIRST** - No Python in production paths
6. **QUALITY GATES** - All 5 gates must pass before merge
7. **TASK TRACKING** - All work tracked in Taskwarrior

---

## VERSION

```yaml
File: CLAUDE.md
Version: 1.0.0
Last Updated: 2026-01-01
Changelog:
  - v1.0.0 - Initial CLAUDE.md for reasonkit-web
    - Project identity and technology stack
    - Hard constraints and security requirements
    - Development commands and deployment
    - MCP protocol documentation
    - Testing and configuration guides
```

---

_ReasonKit-web-rs | MCP Sidecar for Web Sensing | Pure Rust_
_<https://reasonkit.sh>_

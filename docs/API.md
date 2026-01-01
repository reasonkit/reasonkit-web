# ReasonKit Web API Documentation

> **Version:** 0.1.0
> **Protocol:** MCP (Model Context Protocol) over stdio
> **License:** Apache 2.0

---

## Overview

ReasonKit Web is a high-performance MCP (Model Context Protocol) server for web sensing and browser automation. It provides AI agents with tools for headless browser control, content extraction, and page capture.

### Communication Protocol

ReasonKit Web uses **JSON-RPC 2.0 over stdio** as defined by the MCP specification:

| Property | Value |
|----------|-------|
| Transport | stdio (stdin/stdout) |
| Format | JSON-RPC 2.0 |
| Protocol Version | `2024-11-05` |

### Quick Start

```bash
# Start the MCP server
./reasonkit-web serve

# Or simply (serve is the default command)
./reasonkit-web
```

---

## JSON-RPC Protocol

All requests and responses follow JSON-RPC 2.0 specification.

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "<method_name>",
  "params": { ... },
  "id": 1
}
```

### Response Format (Success)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... }
}
```

### Response Format (Error)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Error description"
  }
}
```

---

## Lifecycle Methods

### initialize

Initialize the MCP session. Must be called before any other methods.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "your-client",
      "version": "1.0.0"
    }
  },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {
        "listChanged": false
      }
    },
    "serverInfo": {
      "name": "reasonkit-web",
      "version": "0.1.0"
    }
  }
}
```

### initialized (Notification)

Signal that initialization is complete. This is a notification (no response expected).

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "initialized"
}
```

### shutdown

Request server shutdown.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "shutdown",
  "id": 2
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": null
}
```

### ping

Health check endpoint for testing connectivity.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "ping",
  "id": 3
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "pong": true
  }
}
```

---

## Tool Methods

### tools/list

List all available MCP tools.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 4
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "tools": [
      {
        "name": "web_navigate",
        "description": "Navigate to a URL using a headless browser",
        "inputSchema": { ... }
      },
      ...
    ]
  }
}
```

### tools/call

Execute a specific tool.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "<tool_name>",
    "arguments": { ... }
  },
  "id": 5
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Result content..."
      }
    ]
  }
}
```

---

## Available Tools

ReasonKit Web provides 8 built-in tools for web automation:

| Tool | Description |
|------|-------------|
| `web_navigate` | Navigate to a URL |
| `web_screenshot` | Capture page screenshot |
| `web_pdf` | Generate PDF of page |
| `web_extract_content` | Extract main content |
| `web_extract_links` | Extract all links |
| `web_extract_metadata` | Extract page metadata |
| `web_execute_js` | Execute JavaScript |
| `web_capture_mhtml` | Capture complete MHTML archive |

---

## Tool Reference

### web_navigate

Navigate to a URL using a headless browser.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | The URL to navigate to |
| `waitFor` | string | No | CSS selector to wait for before returning |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_navigate",
    "arguments": {
      "url": "https://example.com",
      "waitFor": "#main-content"
    }
  },
  "id": 10
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Successfully navigated to: https://example.com/"
      }
    ]
  }
}
```

---

### web_screenshot

Capture a screenshot of a web page.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | The URL to capture |
| `fullPage` | boolean | No | `true` | Capture full page |
| `format` | string | No | `"png"` | Image format: `png`, `jpeg`, `webp` |
| `selector` | string | No | - | CSS selector to capture specific element |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_screenshot",
    "arguments": {
      "url": "https://example.com",
      "fullPage": true,
      "format": "png"
    }
  },
  "id": 11
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 11,
  "result": {
    "content": [
      {
        "type": "image",
        "data": "iVBORw0KGgoAAAANSUhEUgAA...",
        "mimeType": "image/png"
      }
    ]
  }
}
```

---

### web_pdf

Generate a PDF of a web page.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | The URL to convert to PDF |
| `printBackground` | boolean | No | `true` | Print background graphics |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_pdf",
    "arguments": {
      "url": "https://example.com",
      "printBackground": true
    }
  },
  "id": 12
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "PDF generated: 45678 bytes"
      },
      {
        "type": "resource",
        "uri": "pdf://https://example.com",
        "resource": {
          "mimeType": "application/pdf",
          "blob": "JVBERi0xLjQKJeLj..."
        }
      }
    ]
  }
}
```

---

### web_extract_content

Extract main content from a web page as text or markdown.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | The URL to extract content from |
| `selector` | string | No | - | CSS selector (defaults to auto-detect main content) |
| `format` | string | No | `"markdown"` | Output format: `text`, `markdown`, `html` |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_extract_content",
    "arguments": {
      "url": "https://example.com",
      "format": "markdown"
    }
  },
  "id": 13
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 13,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "# Example Domain\n\nThis domain is for use in illustrative examples..."
      }
    ]
  }
}
```

---

### web_extract_links

Extract all links from a web page with context.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | The URL to extract links from |
| `type` | string | No | `"all"` | Link type: `all`, `internal`, `external` |
| `selector` | string | No | - | CSS selector to extract links from |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_extract_links",
    "arguments": {
      "url": "https://example.com",
      "type": "external"
    }
  },
  "id": 14
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 14,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[\n  {\n    \"url\": \"https://www.iana.org/domains/example\",\n    \"text\": \"More information...\",\n    \"link_type\": \"External\"\n  }\n]"
      }
    ]
  }
}
```

---

### web_extract_metadata

Extract page metadata (title, description, Open Graph, Twitter Card, etc.).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | The URL to extract metadata from |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_extract_metadata",
    "arguments": {
      "url": "https://example.com"
    }
  },
  "id": 15
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 15,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"title\": \"Example Domain\",\n  \"description\": \"This domain is for use in illustrative examples...\",\n  \"language\": \"en\",\n  \"og_title\": null,\n  \"og_description\": null,\n  \"og_image\": null,\n  \"twitter_card\": null\n}"
      }
    ]
  }
}
```

---

### web_execute_js

Execute JavaScript on a web page and return the result.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | The URL to execute JavaScript on |
| `script` | string | Yes | The JavaScript code to execute |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_execute_js",
    "arguments": {
      "url": "https://example.com",
      "script": "document.title"
    }
  },
  "id": 16
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 16,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "\"Example Domain\""
      }
    ]
  }
}
```

**Advanced Example (DOM Manipulation):**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_execute_js",
    "arguments": {
      "url": "https://example.com",
      "script": "Array.from(document.querySelectorAll('a')).map(a => ({href: a.href, text: a.textContent}))"
    }
  },
  "id": 17
}
```

---

### web_capture_mhtml

Capture a complete web page as an MHTML archive (including all resources).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | The URL to capture |

**Request Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "web_capture_mhtml",
    "arguments": {
      "url": "https://example.com"
    }
  },
  "id": 18
}
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 18,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "MHTML captured: 123456 bytes"
      },
      {
        "type": "resource",
        "uri": "mhtml://https://example.com",
        "resource": {
          "mimeType": "multipart/related",
          "blob": "RnJvbTogPFNhdmVkIGJ5IEJsaW5rPg..."
        }
      }
    ]
  }
}
```

---

## Error Codes

Standard JSON-RPC 2.0 error codes:

| Code | Message | Description |
|------|---------|-------------|
| `-32700` | Parse error | Invalid JSON |
| `-32600` | Invalid Request | Invalid JSON-RPC request |
| `-32601` | Method not found | Unknown method |
| `-32602` | Invalid params | Invalid method parameters |
| `-32603` | Internal error | Server-side error |

### Tool-Specific Errors

Tool execution errors are returned in the result with `isError: true`:

```json
{
  "jsonrpc": "2.0",
  "id": 20,
  "result": {
    "isError": true,
    "content": [
      {
        "type": "text",
        "text": "Navigation failed: net::ERR_NAME_NOT_RESOLVED"
      }
    ]
  }
}
```

### Error Categories

| Category | Examples |
|----------|----------|
| **Browser** | Launch failed, Connection lost, Timeout |
| **Navigation** | Invalid URL, Timeout, SSL error, HTTP error |
| **Extraction** | Element not found, Invalid selector, Parse error |
| **Capture** | Screenshot failed, PDF failed, MHTML failed |

---

## CLI Commands

ReasonKit Web also provides CLI commands for direct usage:

### serve (default)

Run the MCP server:

```bash
# Default
./reasonkit-web serve

# With verbose logging
./reasonkit-web --verbose serve

# With debug log level
./reasonkit-web --log-level debug serve
```

### test

Test browser automation:

```bash
./reasonkit-web test https://example.com
```

### extract

Extract content from a URL:

```bash
# Extract as markdown (default)
./reasonkit-web extract https://example.com

# Extract as plain text
./reasonkit-web extract https://example.com --format text

# Extract as HTML
./reasonkit-web extract https://example.com --format html
```

### screenshot

Take a screenshot:

```bash
# Basic screenshot
./reasonkit-web screenshot https://example.com

# Full page capture
./reasonkit-web screenshot https://example.com --full-page

# Custom output path
./reasonkit-web screenshot https://example.com --output page.png
```

### tools

List available MCP tools:

```bash
./reasonkit-web tools
```

---

## Integration Examples

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reasonkit-web": {
      "command": "/path/to/reasonkit-web",
      "args": ["serve"]
    }
  }
}
```

### Cursor Editor Configuration

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "reasonkit-web": {
      "command": "/path/to/reasonkit-web",
      "args": ["--log-level", "warn", "serve"]
    }
  }
}
```

### Programmatic Usage (Rust)

```rust
use reasonkit_web::browser::BrowserController;
use reasonkit_web::extraction::{ContentExtractor, MetadataExtractor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize browser
    let controller = BrowserController::new().await?;

    // Navigate to page
    let page = controller.navigate("https://example.com").await?;

    // Extract metadata
    let metadata = MetadataExtractor::extract(&page).await?;
    println!("Title: {:?}", metadata.title);

    // Extract main content
    let content = ContentExtractor::extract_main_content(&page).await?;
    println!("Word count: {}", content.word_count);
    println!("Content: {}", content.text);

    // Clean up
    controller.close().await?;
    Ok(())
}
```

---

## Content Types

### ToolContent

Tool results can contain different content types:

#### Text Content

```json
{
  "type": "text",
  "text": "The extracted or generated text content"
}
```

#### Image Content

```json
{
  "type": "image",
  "data": "base64-encoded-image-data",
  "mimeType": "image/png"
}
```

#### Resource Content

```json
{
  "type": "resource",
  "uri": "resource://identifier",
  "resource": {
    "mimeType": "application/pdf",
    "text": null,
    "blob": "base64-encoded-binary-data"
  }
}
```

---

## Performance Considerations

### Browser Pool

The current implementation creates a new browser instance per tool execution. For production deployments with high throughput requirements, consider implementing browser pooling.

### Timeouts

Default timeout values:

| Operation | Timeout |
|-----------|---------|
| Browser launch | 30s |
| Navigation | 30s |
| Screenshot | 30s |
| PDF generation | 60s |

### Memory Usage

Each browser instance consumes approximately 100-200MB of RAM. Monitor memory usage when processing many concurrent requests.

---

## Security Considerations

### JavaScript Execution

The `web_execute_js` tool allows arbitrary JavaScript execution. Ensure:

- Input scripts are validated or sandboxed
- Access is restricted in multi-tenant environments
- Sensitive data is not exposed through script execution

### URL Validation

All navigation URLs are processed by the browser. Consider:

- Implementing URL allowlists for production
- Blocking local/internal network access
- Rate limiting requests per domain

### Headless Detection

The browser runs with stealth mode enabled by default to avoid headless detection. See `src/browser/stealth.rs` for configuration options.

---

## Troubleshooting

### Common Errors

**Browser failed to launch:**
```
Browser error: Failed to launch browser: no chrome found
```
- Ensure Chrome/Chromium is installed
- Set `CHROME_PATH` environment variable if needed

**Navigation timeout:**
```
Navigation error: Navigation timed out after 30000ms
```
- Check network connectivity
- Increase timeout for slow-loading pages
- Verify the URL is correct

**Element not found:**
```
Extraction error: Element not found: #nonexistent
```
- Verify the CSS selector is correct
- Ensure the page has fully loaded
- Use `waitFor` parameter in navigation

### Debug Logging

Enable debug logging for troubleshooting:

```bash
./reasonkit-web --log-level debug serve
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-01-01 | Initial release |

---

## Related Documentation

- [Architecture Guide](./ARCHITECTURE.md)
- [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md)
- [ReasonKit Core](https://github.com/ReasonKit/reasonkit-core)
- [MCP Specification](https://modelcontextprotocol.io)

---

<div align="center">

**ReasonKit Web** - _The Eyes and Ears of AI Reasoning_

[Website](https://reasonkit.sh) | [GitHub](https://github.com/ReasonKit/reasonkit-web) | [Crates.io](https://crates.io/crates/reasonkit-web)

</div>

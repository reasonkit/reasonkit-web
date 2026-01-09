# ReasonKit Web Setup Guide

> **Version:** 0.1.0
> **Prerequisites:** Rust 1.75+, Chrome/Chromium

## Installation

ReasonKit Web can be installed as a standalone binary or used as a library in Rust projects.

### Standalone Binary (MCP Server)

The standalone binary runs as a Model Context Protocol (MCP) server, allowing AI agents (like Claude Desktop, Cursor, or custom agents) to control a headless browser.

#### Option 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/ReasonKit/reasonkit-web.git
cd reasonkit-web

# Build release binary
cargo build --release

# Move to a directory in your PATH
sudo cp target/release/reasonkit-web /usr/local/bin/
```

#### Option 2: Install via Cargo

```bash
cargo install reasonkit-web
```

### Library Usage

Add `reasonkit-web` to your `Cargo.toml`:

```toml
[dependencies]
reasonkit-web = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

---

## Configuration

ReasonKit Web can be configured via environment variables or command-line arguments.

### Environment Variables

| Variable | Description | Default |
| bound | ----------- | ------- |
| `CHROME_PATH` | Path to Chrome/Chromium executable | Auto-detected |
| `RUST_LOG` | Logging level (`error`, `warn`, `info`, `debug`, `trace`) | `info` |
| `HEADLESS` | Run in headless mode | `true` |
| `USER_AGENT` | Custom User-Agent string | Random real user agent |

### Command Line Arguments

```bash
reasonkit-web [OPTIONS] <COMMAND>

Commands:
  serve       Run the MCP server (default)
  test        Test browser automation on a URL
  extract     Extract content from a URL
  screenshot  Take a screenshot of a URL
  tools       List available tools
  help        Print this message

Options:
  -v, --verbose          Enable verbose logging
  --log-level <LEVEL>    Set log level (error, warn, info, debug, trace)
  --chrome-path <PATH>   Path to Chrome executable
  -h, --help             Print help
  -V, --version          Print version
```

---

## Integration Setup

### Claude Desktop

To use ReasonKit Web with Claude Desktop:

1.  Open or create your config file:
    - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
    - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2.  Add the server configuration:

```json
{
  "mcpServers": {
    "reasonkit-web": {
      "command": "/usr/local/bin/reasonkit-web",
      "args": ["serve"]
    }
  }
}
```

3.  Restart Claude Desktop. The 🔨 icon should appear, listing tools like `web_navigate`, `web_screenshot`, etc.

### Cursor Editor

To use ReasonKit Web with Cursor:

1.  Open `.cursor/mcp.json` in your project root.

2.  Add the server configuration:

```json
{
  "mcpServers": {
    "reasonkit-web": {
      "command": "/usr/local/bin/reasonkit-web",
      "args": ["serve"]
    }
  }
}
```

### Custom Agent (Python)

If you are building a custom agent in Python using the MCP SDK:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Server parameters
server_params = StdioServerParameters(
    command="reasonkit-web",
    args=["serve"],
    env=None
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # Call a tool
            result = await session.call_tool(
                "web_navigate",
                arguments={"url": "https://example.com"}
            )
            print(result)
```

---

## Verification

To verify your installation works:

1.  Run the test command:

    ```bash
    reasonkit-web test https://example.com
    ```

2.  You should see output indicating successful navigation and content extraction.

## Troubleshooting

- **"No Chrome found":** Ensure Google Chrome or Chromium is installed. If it's in a non-standard location, set `CHROME_PATH`.
- **"Connection refused":** The tool creates a WebSocket connection to the browser. Ensure no firewall is blocking localhost ports.
- **"Zombie processes":** If the tool crashes, orphan Chrome processes might remain. Kill them with `pkill -f chrome`.

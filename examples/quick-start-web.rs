//! # ReasonKit Web - Quick Start Example
//!
//! This example demonstrates basic usage of ReasonKit Web's MCP server.
//!
//! Run with: `cargo run --example quick-start-web --package reasonkit-web`

use reasonkit_web::McpServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ReasonKit Web - Quick Start                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create MCP server instance
    let _server = McpServer::new();

    println!("✅ MCP Server created");
    println!();

    // Display server information
    println!("📋 Server Information:");
    println!("   Name: ReasonKit Web MCP Server");
    println!("   Version: 0.1.0");
    println!("   Protocol: MCP (stdio)");
    println!();

    println!("💡 Usage:");
    println!("   The MCP server runs over stdio and communicates via JSON-RPC.");
    println!("   To use it, connect your AI agent to the server's stdin/stdout.");
    println!();

    println!("🔧 Available Tools:");
    println!("   - web_capture: Capture web pages (screenshots, PDFs, HTML)");
    println!("   - web_extract: Extract content and metadata from web pages");
    println!("   - browser_navigate: Navigate to URLs");
    println!("   - browser_click: Click elements on pages");
    println!("   - browser_type: Type text into forms");
    println!();

    println!("📖 For more information:");
    println!("   - See README.md for full documentation");
    println!("   - See examples/ for usage examples");
    println!();

    println!("✅ Quick start example completed!");
    println!();
    println!("💡 Next steps:");
    println!("   - Connect to the MCP server from your AI agent");
    println!("   - Use browser automation tools");
    println!("   - Extract web content");
    println!("   - Monitor server metrics");
    println!();

    Ok(())
}

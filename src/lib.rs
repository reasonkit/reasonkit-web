//! ReasonKit Web - High-Performance Browser Connector
//!
//! A Rust-powered MCP server for browser automation, web capture, and content extraction.
//!
//! # Features
//!
//! - **CDP Integration**: Chrome DevTools Protocol client for browser automation
//! - **MCP Server**: Model Context Protocol server for AI agent integration
//! - **Content Extraction**: HTML parsing and structured data extraction
//! - **WASM Support**: Browser-native execution via WebAssembly
//!
//! # Example
//!
//! ```rust,ignore
//! use reasonkit_web::BrowserConnector;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let connector = BrowserConnector::new().await?;
//!     let page = connector.navigate("https://example.com").await?;
//!     let content = page.capture_content().await?;
//!     Ok(())
//! }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod browser;
pub mod buffer;
pub mod cors;
pub mod error;
pub mod extraction;
pub mod handlers;
pub mod mcp;
pub mod metrics;
pub mod processing;
pub mod shutdown;
pub mod stripe;

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Generate a unique request ID (UUID v4)
pub fn generate_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

// Re-export key components for convenience
pub use browser::controller::BrowserController;
pub use error::Error;
pub use extraction::content::ContentExtractor;
pub use extraction::links::LinkExtractor;
pub use extraction::metadata::MetadataExtractor;
pub use mcp::McpServer;

/// Configuration defaults
pub mod config {
    /// Default connection timeout in milliseconds
    pub const DEFAULT_TIMEOUT_MS: u64 = 30_000;

    /// Default viewport width
    pub const DEFAULT_VIEWPORT_WIDTH: u32 = 1920;

    /// Default viewport height
    pub const DEFAULT_VIEWPORT_HEIGHT: u32 = 1080;
}

/// Browser connector state for WASM environments
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmBrowserConnector {
    connected: bool,
    url: String,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmBrowserConnector {
    /// Create a new WASM browser connector
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            connected: false,
            url: String::new(),
        }
    }

    /// Check if connected
    #[wasm_bindgen(getter)]
    pub fn connected(&self) -> bool {
        self.connected
    }

    /// Get current URL
    #[wasm_bindgen(getter)]
    pub fn url(&self) -> String {
        self.url.clone()
    }

    /// Connect to a URL
    pub fn connect(&mut self, url: &str) -> bool {
        self.url = url.to_string();
        self.connected = true;
        true
    }

    /// Disconnect
    pub fn disconnect(&mut self) {
        self.connected = false;
        self.url.clear();
    }
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmBrowserConnector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        assert_eq!(config::DEFAULT_TIMEOUT_MS, 30_000);
        assert_eq!(config::DEFAULT_VIEWPORT_WIDTH, 1920);
        assert_eq!(config::DEFAULT_VIEWPORT_HEIGHT, 1080);
    }
}

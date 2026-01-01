//! ReasonKit Web - High-Performance Web Sensing & Browser Automation Layer
//!
//! This crate provides a production-ready MCP (Model Context Protocol) server
//! for web sensing, browser automation, and content extraction.
//!
//! # Features
//!
//! - **MCP Server**: Full MCP stdio server for AI agent integration
//! - **Browser Automation**: Headless browser control via ChromiumOxide (CDP)
//! - **Web Capture**: Screenshot, PDF, and HTML capture
//! - **Content Extraction**: Intelligent content parsing and metadata extraction
//! - **Content Processing**: DOM content processing with HTML cleaning and normalization
//! - **Capture Buffer**: In-memory buffer for storing captured content
//! - **CORS Security**: Strict localhost-only CORS policy for HTTP transport
//! - **Graceful Shutdown**: Signal handling, connection draining, systemd integration
//! - **Request Tracing**: Production-ready request logging with JSON output
//! - **SSE Feed**: Real-time event streaming via Server-Sent Events
//! - **Error Handling**: Comprehensive error types with HTTP status mapping
//!
//! # Architecture
//!
//! ```text
//! AI Agent ──▶ MCP Server ──▶ Browser Controller (CDP)
//!                  │                │
//!                  ▼                ▼
//!            ┌──────────┐    ┌──────────────┐
//!            │ Capture  │    │ Extraction   │
//!            └────┬─────┘    └──────┬───────┘
//!                 │                 │
//!                 ▼                 ▼
//!           Screenshots        Content + Metadata
//!           PDFs, HTML         Links, Structured Data
//!                 │                 │
//!                 └────────┬────────┘
//!                          ▼
//!                   ┌──────────────┐
//!                   │ Processing   │
//!                   │ (Cleaning)   │
//!                   └──────┬───────┘
//!                          ▼
//!                   ┌──────────────┐
//!                   │ CaptureBuffer│
//!                   │ (In-Memory)  │
//!                   └──────┬───────┘
//!                          │
//!                          ▼
//!                   ┌──────────────┐
//!                   │  SSE Feed    │──▶ Connected Clients
//!                   │  /feed       │
//!                   └──────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use reasonkit_web::browser::BrowserController;
//! use reasonkit_web::extraction::ContentExtractor;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a browser controller
//!     let controller = BrowserController::new().await?;
//!
//!     // Navigate and extract
//!     let page = controller.navigate("https://example.com").await?;
//!     let content = ContentExtractor::extract_main_content(&page).await?;
//!
//!     println!("Extracted: {}", content.text);
//!     Ok(())
//! }
//! ```
//!
//! # Error Handling
//!
//! The crate provides comprehensive error handling with HTTP-aware errors:
//!
//! ```rust,no_run
//! use reasonkit_web::error::{WebError, WebResult, RequestContext};
//!
//! fn validate_input(data: &str) -> WebResult<()> {
//!     if data.is_empty() {
//!         return Err(WebError::missing_field("data"));
//!     }
//!     if data.len() > 1024 * 1024 {
//!         return Err(WebError::content_too_large(data.len(), 1024 * 1024));
//!     }
//!     Ok(())
//! }
//!
//! fn handle_request() {
//!     let ctx = RequestContext::new();
//!
//!     match validate_input("") {
//!         Ok(_) => println!("Success"),
//!         Err(e) => {
//!             ctx.log_error(&e);
//!             let json = e.to_json_with_request_id(&ctx.request_id);
//!             println!("Status: {}, Body: {}", e.status_code(), json);
//!         }
//!     }
//! }
//! ```
//!
//! # Content Processing Example
//!
//! ```rust
//! use reasonkit_web::processing::{ContentProcessor, ContentProcessorConfig};
//!
//! // Create a processor with default settings
//! let processor = ContentProcessor::with_defaults();
//!
//! // Process raw HTML
//! let html = r#"<html><script>evil();</script><p>Hello &amp; world!</p></html>"#;
//! let result = processor.process(html);
//!
//! assert!(result.text.contains("Hello & world!"));
//! assert!(!result.text.contains("evil"));
//! println!("Processed {} words in {}us", result.word_count, result.processing_time_us);
//! ```
//!
//! # Capture Buffer Example
//!
//! ```rust,no_run
//! use reasonkit_web::buffer::{CaptureBuffer, CaptureRecord, shared_buffer};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create a shared buffer
//!     let buffer = shared_buffer();
//!
//!     // Add a capture
//!     let record = CaptureRecord::new(
//!         "https://example.com".to_string(),
//!         "<html>...</html>".to_string(),
//!         "Extracted text".to_string(),
//!         1234, // processing time in microseconds
//!     );
//!     buffer.push(record).await;
//!
//!     // Get recent captures
//!     let recent = buffer.get_recent(10).await;
//!     println!("Captured {} pages", recent.len());
//! }
//! ```
//!
//! # CORS Configuration
//!
//! The CORS module provides a strict security policy for HTTP transport:
//!
//! ```rust,no_run
//! use reasonkit_web::cors::{cors_layer, CorsConfig};
//! use axum::Router;
//!
//! // Default strict CORS (localhost only)
//! let app = Router::new()
//!     .layer(cors_layer());
//!
//! // Custom configuration
//! let config = CorsConfig::new()
//!     .with_max_age(7200)
//!     .with_allow_credentials(true);
//! ```
//!
//! # Graceful Shutdown Example
//!
//! ```rust,no_run
//! use reasonkit_web::shutdown::{ShutdownController, shutdown_signal};
//!
//! #[tokio::main]
//! async fn main() {
//!     let controller = ShutdownController::new();
//!
//!     // Check shutdown status
//!     if controller.is_shutting_down() {
//!         println!("Server is shutting down");
//!     }
//!
//!     // Track connections with RAII guard
//!     let _guard = controller.connection_guard();
//!
//!     // Use with Axum graceful shutdown
//!     // axum::serve(listener, app)
//!     //     .with_graceful_shutdown(shutdown_signal())
//!     //     .await;
//! }
//! ```
//!
//! # Request Tracing
//!
//! Production-ready request tracing with JSON logging:
//!
//! ```rust,no_run
//! use reasonkit_web::tracing_middleware::{
//!     init_tracing, RequestIdLayer, request_tracing_layer
//! };
//! use axum::Router;
//!
//! // Initialize tracing (JSON in prod, pretty in dev)
//! init_tracing();
//!
//! let app = Router::new()
//!     .layer(request_tracing_layer())
//!     .layer(RequestIdLayer::new());
//! ```
//!
//! # SSE Feed Example
//!
//! ```rust,no_run
//! use reasonkit_web::handlers::feed::{FeedState, build_feed_router};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create shared state for SSE feed
//!     let feed_state = Arc::new(FeedState::new(1024));
//!
//!     // Build router with /feed endpoint
//!     let app = build_feed_router(Arc::clone(&feed_state));
//!
//!     // Publish events from capture pipeline
//!     feed_state.publish_capture_received(
//!         "capture-123",
//!         "https://example.com",
//!         "screenshot"
//!     );
//!
//!     // Start server
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
//!     axum::serve(listener, app).await?;
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod browser;
pub mod buffer;
pub mod cors;
pub mod error;
pub mod extraction;
pub mod handlers;
pub mod mcp;
pub mod metrics;
pub mod processing;
pub mod security;
pub mod shutdown;
pub mod tracing_middleware;

// Re-exports for convenience
pub use browser::BrowserController;
pub use buffer::{CaptureBuffer, CaptureRecord, SharedCaptureBuffer};
pub use cors::{cors_layer, cors_layer_with_config, is_localhost_origin, CorsConfig};
pub use error::{
    generate_request_id, Error, ErrorResponse, RequestContext, Result, WebError, WebResult,
};
pub use extraction::{ContentExtractor, LinkExtractor, MetadataExtractor};
pub use handlers::feed::{FeedEvent, FeedState};
pub use mcp::{McpServer, McpTool};
pub use metrics::{Metrics, MetricsServer, MetricsServerConfig, RequestTimer, METRICS};
pub use processing::{ContentProcessor, ContentProcessorConfig, ProcessedContent};
pub use security::{SecurityCheck, SecurityCheckResult, SecurityConfig, SecurityLayer};
pub use shutdown::{ShutdownController, ShutdownState};
pub use tracing_middleware::{
    init_tracing, init_tracing_with_format, request_tracing_layer, LogFormat, RequestId,
    RequestIdLayer,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

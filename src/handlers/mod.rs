//! HTTP handlers for the ReasonKit Web server
//!
//! This module contains Axum handlers for various HTTP endpoints,
//! including the SSE feed for real-time event streaming and
//! the capture endpoint for DOM content processing.
//!
//! # Modules
//!
//! - [`feed`] - Server-Sent Events feed for real-time capture and processing events
//! - [`capture`] - POST endpoint for capturing and processing DOM content
//! - [`status`] - Server status and health check endpoints
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::handlers::feed::{FeedState, build_feed_router};
//! use reasonkit_web::handlers::capture::{CaptureState, CaptureConfig, capture_router, create_capture_buffer};
//! use reasonkit_web::handlers::status::{AppState, status_router};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Set up feed state
//!     let feed_state = Arc::new(FeedState::new(1024));
//!
//!     // Set up capture state with bounded buffer
//!     let (sender, receiver) = create_capture_buffer(1000);
//!     let capture_state = Arc::new(CaptureState::new(CaptureConfig::default(), sender));
//!
//!     // Set up status/health endpoints
//!     let app_state = Arc::new(AppState::new());
//!     let status_app = status_router(app_state);
//!
//!     // Build routers
//!     let feed_app = build_feed_router(feed_state);
//!     let capture_app = capture_router(capture_state);
//!
//!     // Merge and serve...
//! }
//! ```

pub mod capture;
pub mod feed;
pub mod status;

// Re-export commonly used items from feed
pub use feed::{
    build_feed_router, feed_handler, feed_handler_with_interval, CaptureReceivedData, ErrorData,
    FeedEvent, FeedState, FeedStream, HeartbeatData, ProcessingCompleteData,
};

// Re-export commonly used items from capture
pub use capture::{
    capture_handler, capture_health, capture_router, create_capture_buffer, CaptureConfig,
    CaptureError, CaptureRequest, CaptureResponse, CaptureState, ContentProcessor,
    ProcessedCapture, DEFAULT_BUFFER_CAPACITY, DEFAULT_MAX_CONTENT_LENGTH,
};

// Re-export commonly used items from status
pub use status::{
    // Handlers
    health_handler,
    readiness_handler,
    status_handler,
    // Router
    status_router,
    // Types
    AppState,
    HealthResponse,
    LatencyHistogram,
    LatencyMetrics,
    MemoryMetrics,
    StatusResponse,
    // Constants
    SERVER_NAME,
    SERVER_VERSION,
};

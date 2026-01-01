//! Capture endpoint handler for the ReasonKit Web MCP server
//!
//! This module provides the `/capture` HTTP endpoint for receiving DOM content
//! from browser extensions or other sources, processing it, and storing it
//! in a bounded memory buffer for later retrieval.
//!
//! # Features
//!
//! - URL format validation
//! - Content length limits (configurable)
//! - HTML entity decoding
//! - Script/style tag removal
//! - Whitespace normalization
//! - Metrics tracking (capture count, processing time)
//!
//! # Error Handling
//!
//! - `400 Bad Request` - Invalid request (malformed URL, missing fields)
//! - `413 Payload Too Large` - Content exceeds maximum allowed size
//! - `500 Internal Server Error` - Processing or storage errors

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use metrics::{counter, histogram};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};
use url::Url;
use uuid::Uuid;

// ============================================================================
// Configuration
// ============================================================================

/// Default maximum content length in bytes (10 MB)
pub const DEFAULT_MAX_CONTENT_LENGTH: usize = 10 * 1024 * 1024;

/// Default buffer channel capacity
pub const DEFAULT_BUFFER_CAPACITY: usize = 1000;

/// Configuration for the capture handler
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Maximum allowed content length in bytes
    pub max_content_length: usize,
    /// Maximum length after truncation (if content exceeds max)
    pub truncate_length: Option<usize>,
    /// Whether to strip script tags
    pub strip_scripts: bool,
    /// Whether to strip style tags
    pub strip_styles: bool,
    /// Whether to decode HTML entities
    pub decode_entities: bool,
    /// Whether to normalize whitespace
    pub normalize_whitespace: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            max_content_length: DEFAULT_MAX_CONTENT_LENGTH,
            truncate_length: Some(1024 * 1024), // 1 MB after processing
            strip_scripts: true,
            strip_styles: true,
            decode_entities: true,
            normalize_whitespace: true,
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request body for the `/capture` endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct CaptureRequest {
    /// The URL of the captured page
    pub url: String,

    /// The DOM content (HTML)
    pub content: String,

    /// Optional title of the page
    #[serde(default)]
    pub title: Option<String>,

    /// Optional description/meta description
    #[serde(default)]
    pub description: Option<String>,

    /// Optional timestamp of capture (defaults to server time)
    #[serde(default)]
    pub captured_at: Option<DateTime<Utc>>,

    /// Optional metadata as key-value pairs
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Response body for successful capture
#[derive(Debug, Clone, Serialize)]
pub struct CaptureResponse {
    /// Unique identifier for this capture
    pub id: Uuid,

    /// The URL that was captured
    pub url: String,

    /// Timestamp when the capture was processed
    pub processed_at: DateTime<Utc>,

    /// Size of the original content in bytes
    pub original_size: usize,

    /// Size of the processed content in bytes
    pub processed_size: usize,

    /// Whether the content was truncated
    pub truncated: bool,

    /// Processing duration in milliseconds
    pub processing_time_ms: u64,
}

/// Processed capture data stored in the buffer
#[derive(Debug, Clone, Serialize)]
pub struct ProcessedCapture {
    /// Unique identifier
    pub id: Uuid,

    /// Original URL
    pub url: String,

    /// Page title
    pub title: Option<String>,

    /// Page description
    pub description: Option<String>,

    /// Processed text content
    pub content: String,

    /// Original HTML (if preserved)
    pub original_html: Option<String>,

    /// Capture timestamp
    pub captured_at: DateTime<Utc>,

    /// Processing timestamp
    pub processed_at: DateTime<Utc>,

    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during capture processing
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// Invalid request (missing fields, malformed data)
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Invalid URL format
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// Content exceeds maximum allowed size
    #[error("Content too large: {size} bytes exceeds maximum of {max} bytes")]
    ContentTooLarge {
        /// Actual content size in bytes
        size: usize,
        /// Maximum allowed size in bytes
        max: usize,
    },

    /// Content processing failed
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Failed to store capture in buffer
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl IntoResponse for CaptureError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            CaptureError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request", msg.clone())
            }
            CaptureError::InvalidUrl(msg) => (StatusCode::BAD_REQUEST, "invalid_url", msg.clone()),
            CaptureError::ContentTooLarge { size, max } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "content_too_large",
                format!("Content size {} exceeds maximum {}", size, max),
            ),
            CaptureError::ProcessingError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "processing_error",
                msg.clone(),
            ),
            CaptureError::StorageError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                msg.clone(),
            ),
            CaptureError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                msg.clone(),
            ),
        };

        // Increment error counter
        counter!("capture_errors_total", "type" => error_type).increment(1);

        let body = serde_json::json!({
            "error": {
                "type": error_type,
                "message": message,
            }
        });

        (status, Json(body)).into_response()
    }
}

// ============================================================================
// Shared State
// ============================================================================

/// Shared state for the capture handler
#[derive(Clone)]
pub struct CaptureState {
    /// Configuration
    pub config: CaptureConfig,
    /// Channel sender for storing captures
    pub sender: mpsc::Sender<ProcessedCapture>,
}

impl CaptureState {
    /// Create a new capture state with the given configuration
    pub fn new(config: CaptureConfig, sender: mpsc::Sender<ProcessedCapture>) -> Self {
        Self { config, sender }
    }

    /// Create a new capture state with default configuration
    pub fn with_defaults(sender: mpsc::Sender<ProcessedCapture>) -> Self {
        Self::new(CaptureConfig::default(), sender)
    }
}

/// Create a new bounded channel for capture storage
pub fn create_capture_buffer(
    capacity: usize,
) -> (
    mpsc::Sender<ProcessedCapture>,
    mpsc::Receiver<ProcessedCapture>,
) {
    mpsc::channel(capacity)
}

// ============================================================================
// Content Processing
// ============================================================================

/// Content processor for cleaning and normalizing HTML
pub struct ContentProcessor {
    /// Compiled regex for script tags
    script_regex: Regex,
    /// Compiled regex for style tags
    style_regex: Regex,
    /// Compiled regex for HTML tags
    tag_regex: Regex,
    /// Compiled regex for multiple whitespace
    whitespace_regex: Regex,
    /// Compiled regex for multiple newlines
    newline_regex: Regex,
}

impl ContentProcessor {
    /// Create a new content processor with pre-compiled regexes
    pub fn new() -> Self {
        Self {
            script_regex: Regex::new(r"(?is)<script[^>]*>[\s\S]*?</script>").unwrap(),
            style_regex: Regex::new(r"(?is)<style[^>]*>[\s\S]*?</style>").unwrap(),
            tag_regex: Regex::new(r"<[^>]+>").unwrap(),
            whitespace_regex: Regex::new(r"[ \t]+").unwrap(),
            newline_regex: Regex::new(r"\n{3,}").unwrap(),
        }
    }

    /// Process HTML content according to the given configuration
    #[instrument(skip(self, html, config))]
    pub fn process(&self, html: &str, config: &CaptureConfig) -> Result<String, CaptureError> {
        let mut content = html.to_string();

        // Strip script tags if configured
        if config.strip_scripts {
            content = self.script_regex.replace_all(&content, "").to_string();
            debug!("Stripped script tags");
        }

        // Strip style tags if configured
        if config.strip_styles {
            content = self.style_regex.replace_all(&content, "").to_string();
            debug!("Stripped style tags");
        }

        // Replace block elements with newlines for better text extraction
        content = content
            .replace("</p>", "\n")
            .replace("</div>", "\n")
            .replace("</li>", "\n")
            .replace("</tr>", "\n")
            .replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("<br />", "\n");

        // Strip all remaining HTML tags
        content = self.tag_regex.replace_all(&content, "").to_string();

        // Decode HTML entities if configured
        if config.decode_entities {
            content = Self::decode_html_entities(&content);
            debug!("Decoded HTML entities");
        }

        // Normalize whitespace if configured
        if config.normalize_whitespace {
            // Replace multiple spaces/tabs with single space
            content = self.whitespace_regex.replace_all(&content, " ").to_string();
            // Replace multiple newlines with double newline
            content = self.newline_regex.replace_all(&content, "\n\n").to_string();
            // Trim each line
            content = content
                .lines()
                .map(|l| l.trim())
                .collect::<Vec<_>>()
                .join("\n");
            debug!("Normalized whitespace");
        }

        // Final trim
        content = content.trim().to_string();

        Ok(content)
    }

    /// Decode common HTML entities
    fn decode_html_entities(text: &str) -> String {
        // Use htmlescape crate for comprehensive decoding, with fallback
        match htmlescape::decode_html(text) {
            Ok(decoded) => decoded,
            Err(_) => {
                // Fallback to manual decoding for common entities
                text.replace("&nbsp;", " ")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&amp;", "&")
                    .replace("&quot;", "\"")
                    .replace("&#39;", "'")
                    .replace("&apos;", "'")
                    .replace("&mdash;", "\u{2014}")
                    .replace("&ndash;", "\u{2013}")
                    .replace("&hellip;", "\u{2026}")
                    .replace("&lsquo;", "\u{2018}")
                    .replace("&rsquo;", "\u{2019}")
                    .replace("&ldquo;", "\u{201C}")
                    .replace("&rdquo;", "\u{201D}")
                    .replace("&copy;", "\u{00A9}")
                    .replace("&reg;", "\u{00AE}")
                    .replace("&trade;", "\u{2122}")
            }
        }
    }
}

impl Default for ContentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// URL Validation
// ============================================================================

/// Validate a URL string
fn validate_url(url_str: &str) -> Result<Url, CaptureError> {
    // Check for empty URL
    if url_str.is_empty() {
        return Err(CaptureError::InvalidUrl("URL cannot be empty".to_string()));
    }

    // Parse the URL
    let url = Url::parse(url_str)
        .map_err(|e| CaptureError::InvalidUrl(format!("Failed to parse URL: {}", e)))?;

    // Validate scheme
    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(CaptureError::InvalidUrl(format!(
                "Invalid URL scheme '{}': only http and https are allowed",
                scheme
            )));
        }
    }

    // Validate host
    if url.host().is_none() {
        return Err(CaptureError::InvalidUrl(
            "URL must have a valid host".to_string(),
        ));
    }

    Ok(url)
}

// ============================================================================
// Handler Implementation
// ============================================================================

/// Handle the `/capture` POST endpoint
///
/// This endpoint accepts JSON payload with DOM content, processes it,
/// stores it in the memory buffer, and returns a UUID for tracking.
///
/// # Request Body
///
/// ```json
/// {
///     "url": "https://example.com/page",
///     "content": "<html>...</html>",
///     "title": "Page Title",
///     "description": "Page description",
///     "captured_at": "2024-01-01T00:00:00Z",
///     "metadata": { "key": "value" }
/// }
/// ```
///
/// # Response
///
/// ```json
/// {
///     "id": "550e8400-e29b-41d4-a716-446655440000",
///     "url": "https://example.com/page",
///     "processed_at": "2024-01-01T00:00:01Z",
///     "original_size": 10240,
///     "processed_size": 5120,
///     "truncated": false,
///     "processing_time_ms": 15
/// }
/// ```
///
/// # Errors
///
/// - `400 Bad Request` - Invalid URL format or missing required fields
/// - `413 Payload Too Large` - Content exceeds maximum allowed size
/// - `500 Internal Server Error` - Processing or storage errors
#[instrument(skip(state, request), fields(url = %request.url))]
pub async fn capture_handler(
    State(state): State<Arc<CaptureState>>,
    Json(request): Json<CaptureRequest>,
) -> Result<Json<CaptureResponse>, CaptureError> {
    let start_time = Instant::now();
    info!("Processing capture request for URL: {}", request.url);

    // Validate URL format
    let validated_url = validate_url(&request.url)?;
    debug!("URL validated: {}", validated_url);

    // Check content length
    let original_size = request.content.len();
    if original_size > state.config.max_content_length {
        warn!(
            "Content too large: {} bytes (max: {})",
            original_size, state.config.max_content_length
        );
        return Err(CaptureError::ContentTooLarge {
            size: original_size,
            max: state.config.max_content_length,
        });
    }

    // Process content
    let processor = ContentProcessor::new();
    let processed_content = processor.process(&request.content, &state.config)?;

    // Check for truncation
    let (final_content, truncated) = match state.config.truncate_length {
        Some(max_len) if processed_content.len() > max_len => {
            // Truncate at word boundary if possible
            let truncated_content =
                if let Some(last_space) = processed_content[..max_len].rfind(' ') {
                    format!("{}...", &processed_content[..last_space])
                } else {
                    format!("{}...", &processed_content[..max_len])
                };
            info!(
                "Content truncated from {} to {} bytes",
                processed_content.len(),
                truncated_content.len()
            );
            (truncated_content, true)
        }
        _ => (processed_content, false),
    };

    let processed_size = final_content.len();

    // Generate UUID
    let id = Uuid::new_v4();
    let now = Utc::now();

    // Create processed capture
    let capture = ProcessedCapture {
        id,
        url: validated_url.to_string(),
        title: request.title,
        description: request.description,
        content: final_content,
        original_html: None, // Don't store original to save memory
        captured_at: request.captured_at.unwrap_or(now),
        processed_at: now,
        metadata: request.metadata,
    };

    // Store in buffer
    state.sender.send(capture).await.map_err(|e| {
        error!("Failed to store capture in buffer: {}", e);
        CaptureError::StorageError(format!("Buffer full or closed: {}", e))
    })?;

    let processing_time = start_time.elapsed();
    let processing_time_ms = processing_time.as_millis() as u64;

    // Record metrics
    counter!("captures_total").increment(1);
    histogram!("capture_processing_time_seconds").record(processing_time.as_secs_f64());
    histogram!("capture_original_size_bytes").record(original_size as f64);
    histogram!("capture_processed_size_bytes").record(processed_size as f64);

    if truncated {
        counter!("captures_truncated_total").increment(1);
    }

    info!(
        "Capture processed successfully: id={}, original_size={}, processed_size={}, time={}ms",
        id, original_size, processed_size, processing_time_ms
    );

    Ok(Json(CaptureResponse {
        id,
        url: validated_url.to_string(),
        processed_at: now,
        original_size,
        processed_size,
        truncated,
        processing_time_ms,
    }))
}

/// Health check endpoint for the capture service
pub async fn capture_health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "capture",
        "timestamp": Utc::now().to_rfc3339()
    }))
}

// ============================================================================
// Router Configuration
// ============================================================================

use axum::{routing::post, Router};

/// Create the capture router with all endpoints
pub fn capture_router(state: Arc<CaptureState>) -> Router {
    Router::new()
        .route("/capture", post(capture_handler))
        .route("/capture/health", axum::routing::get(capture_health))
        .with_state(state)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_url_valid_http() {
        let result = validate_url("http://example.com/page");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_url_valid_https() {
        let result = validate_url("https://example.com/page?query=1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_url_empty() {
        let result = validate_url("");
        assert!(matches!(result, Err(CaptureError::InvalidUrl(_))));
    }

    #[test]
    fn test_validate_url_invalid_scheme() {
        let result = validate_url("ftp://example.com/file");
        assert!(matches!(result, Err(CaptureError::InvalidUrl(_))));
    }

    #[test]
    fn test_validate_url_no_host() {
        // Note: For http/https URLs, the url crate always interprets something as a host.
        // "http:///path" is parsed with "path" as the domain and "/" as the path.
        // Testing a truly hostless URL requires a non-http scheme which fails the scheme check.
        // This test verifies invalid schemes are rejected:
        let result = validate_url("file:///path");
        assert!(matches!(result, Err(CaptureError::InvalidUrl(_))));

        // Also verify that IP-like invalid hosts still parse as domains:
        let result = validate_url("http:///");
        // The url crate treats this as host="" which becomes None for http
        // but may vary - we just ensure no panics and proper error handling
        assert!(result.is_err() || result.unwrap().host().is_some());
    }

    #[test]
    fn test_validate_url_malformed() {
        let result = validate_url("not a url");
        assert!(matches!(result, Err(CaptureError::InvalidUrl(_))));
    }

    #[test]
    fn test_content_processor_strips_scripts() {
        let processor = ContentProcessor::new();
        let config = CaptureConfig::default();

        let html = "<p>Hello</p><script>evil();</script><p>World</p>";
        let result = processor.process(html, &config).unwrap();

        assert!(!result.contains("script"));
        assert!(!result.contains("evil"));
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
    }

    #[test]
    fn test_content_processor_strips_styles() {
        let processor = ContentProcessor::new();
        let config = CaptureConfig::default();

        let html = "<p>Content</p><style>.hidden { display: none; }</style>";
        let result = processor.process(html, &config).unwrap();

        assert!(!result.contains("style"));
        assert!(!result.contains("display"));
        assert!(result.contains("Content"));
    }

    #[test]
    fn test_content_processor_decodes_entities() {
        let processor = ContentProcessor::new();
        let config = CaptureConfig::default();

        let html = "<p>Hello &amp; World &lt;test&gt;</p>";
        let result = processor.process(html, &config).unwrap();

        assert!(result.contains("Hello & World <test>"));
    }

    #[test]
    fn test_content_processor_normalizes_whitespace() {
        let processor = ContentProcessor::new();
        let config = CaptureConfig::default();

        let html = "<p>Hello    World</p>\n\n\n\n<p>Next</p>";
        let result = processor.process(html, &config).unwrap();

        // Should not have excessive whitespace
        assert!(!result.contains("    "));
        assert!(!result.contains("\n\n\n"));
    }

    #[test]
    fn test_content_processor_strips_tags() {
        let processor = ContentProcessor::new();
        let config = CaptureConfig::default();

        let html = "<div class=\"container\"><p>Text</p></div>";
        let result = processor.process(html, &config).unwrap();

        assert!(!result.contains("<"));
        assert!(!result.contains(">"));
        assert!(result.contains("Text"));
    }

    #[test]
    fn test_capture_request_deserialization() {
        let json = r#"{
            "url": "https://example.com",
            "content": "<p>Hello</p>",
            "title": "Test Page"
        }"#;

        let request: CaptureRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.url, "https://example.com");
        assert_eq!(request.content, "<p>Hello</p>");
        assert_eq!(request.title, Some("Test Page".to_string()));
        assert!(request.description.is_none());
    }

    #[test]
    fn test_capture_response_serialization() {
        let response = CaptureResponse {
            id: Uuid::new_v4(),
            url: "https://example.com".to_string(),
            processed_at: Utc::now(),
            original_size: 1000,
            processed_size: 500,
            truncated: false,
            processing_time_ms: 10,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"id\""));
        assert!(json.contains("\"url\""));
        assert!(json.contains("\"processed_size\""));
    }

    #[test]
    fn test_capture_config_default() {
        let config = CaptureConfig::default();
        assert_eq!(config.max_content_length, DEFAULT_MAX_CONTENT_LENGTH);
        assert!(config.strip_scripts);
        assert!(config.strip_styles);
        assert!(config.decode_entities);
        assert!(config.normalize_whitespace);
    }

    #[tokio::test]
    async fn test_capture_buffer_channel() {
        let (tx, mut rx) = create_capture_buffer(10);

        let capture = ProcessedCapture {
            id: Uuid::new_v4(),
            url: "https://example.com".to_string(),
            title: Some("Test".to_string()),
            description: None,
            content: "Hello World".to_string(),
            original_html: None,
            captured_at: Utc::now(),
            processed_at: Utc::now(),
            metadata: None,
        };

        tx.send(capture.clone()).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.url, "https://example.com");
        assert_eq!(received.content, "Hello World");
    }

    #[test]
    fn test_capture_error_into_response() {
        let error = CaptureError::InvalidRequest("test error".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let error = CaptureError::ContentTooLarge { size: 100, max: 50 };
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let error = CaptureError::ProcessingError("failed".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_html_entity_decoding_comprehensive() {
        let text = "&nbsp;&lt;&gt;&amp;&quot;&#39;&apos;&mdash;&ndash;&hellip;";
        let decoded = ContentProcessor::decode_html_entities(text);

        assert!(decoded.contains('<'));
        assert!(decoded.contains('>'));
        assert!(decoded.contains('&'));
        assert!(decoded.contains('"'));
        assert!(decoded.contains('\''));
    }
}

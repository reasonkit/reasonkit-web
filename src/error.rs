//! Error types for ReasonKit Web
//!
//! This module provides a comprehensive error type hierarchy using `thiserror`
//! for proper error handling across all components.
//!
//! # Error Categories
//!
//! - [`WebError`] - HTTP-level errors with status codes and JSON responses
//! - [`Error`] - Internal operation errors (browser, MCP, extraction)
//! - Domain-specific errors: [`BrowserError`], [`McpError`], [`ExtractionError`], etc.
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::error::{WebError, Result};
//!
//! fn process_request(data: &str) -> Result<String> {
//!     if data.is_empty() {
//!         return Err(WebError::invalid_request("Data cannot be empty").into());
//!     }
//!     Ok(data.to_uppercase())
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use thiserror::Error;
use tracing::{error, warn};

// ============================================================================
// WebError - HTTP-level errors with JSON responses
// ============================================================================

/// HTTP-level error with status codes and structured JSON responses.
///
/// This error type is designed for web API responses, providing:
/// - Appropriate HTTP status codes
/// - Structured JSON error bodies
/// - Request ID correlation for tracing
/// - Logging integration
///
/// # JSON Response Format
///
/// ```json
/// {
///     "error": "Human-readable error message",
///     "code": "ERROR_CODE",
///     "request_id": "optional-request-id"
/// }
/// ```
#[derive(Error, Debug, Clone)]
pub enum WebError {
    /// Invalid request - malformed input, missing fields, validation failures
    /// HTTP Status: 400 Bad Request
    #[error("Invalid request: {message}")]
    InvalidRequest {
        /// Detailed error message
        message: String,
    },

    /// Unauthorized - authentication required or invalid credentials
    /// HTTP Status: 401 Unauthorized
    #[error("Unauthorized: {reason}")]
    Unauthorized {
        /// Reason for authorization failure
        reason: String,
    },

    /// Forbidden - authenticated but not permitted
    /// HTTP Status: 403 Forbidden
    #[error("Forbidden: {reason}")]
    Forbidden {
        /// Reason access was denied
        reason: String,
    },

    /// Resource not found
    /// HTTP Status: 404 Not Found
    #[error("Not found: {resource}")]
    NotFound {
        /// Description of the missing resource
        resource: String,
    },

    /// Request content too large
    /// HTTP Status: 413 Payload Too Large
    #[error("Content too large: {size} bytes exceeds maximum of {max} bytes")]
    ContentTooLarge {
        /// Actual size of the content
        size: usize,
        /// Maximum allowed size
        max: usize,
    },

    /// Rate limit exceeded
    /// HTTP Status: 429 Too Many Requests
    #[error("Rate limited: retry after {retry_after_secs} seconds")]
    RateLimited {
        /// Number of seconds to wait before retrying
        retry_after_secs: u64,
    },

    /// Processing error - operation failed during execution
    /// HTTP Status: 500 Internal Server Error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Internal server error - unexpected failures
    /// HTTP Status: 500 Internal Server Error
    #[error("Internal error: {message}")]
    InternalError {
        /// Error message (sanitized for external display)
        message: String,
    },

    /// Service unavailable - temporarily unable to handle request
    /// HTTP Status: 503 Service Unavailable
    #[error("Service unavailable: {reason}")]
    ServiceUnavailable {
        /// Reason the service is unavailable
        reason: String,
    },

    /// Gateway timeout - upstream service timed out
    /// HTTP Status: 504 Gateway Timeout
    #[error("Gateway timeout after {timeout_ms}ms")]
    GatewayTimeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },
}

impl WebError {
    // ========================================================================
    // Factory Methods
    // ========================================================================

    /// Create an invalid request error with a message
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    /// Create an invalid request error for a missing field
    pub fn missing_field(field: &str) -> Self {
        Self::InvalidRequest {
            message: format!("Missing required field: {}", field),
        }
    }

    /// Create an invalid request error for an invalid field value
    pub fn invalid_field(field: &str, reason: &str) -> Self {
        Self::InvalidRequest {
            message: format!("Invalid value for field '{}': {}", field, reason),
        }
    }

    /// Create an unauthorized error
    pub fn unauthorized(reason: impl Into<String>) -> Self {
        Self::Unauthorized {
            reason: reason.into(),
        }
    }

    /// Create a forbidden error
    pub fn forbidden(reason: impl Into<String>) -> Self {
        Self::Forbidden {
            reason: reason.into(),
        }
    }

    /// Create a not found error
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Create a content too large error
    pub fn content_too_large(size: usize, max: usize) -> Self {
        Self::ContentTooLarge { size, max }
    }

    /// Create a rate limited error
    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self::RateLimited { retry_after_secs }
    }

    /// Create a processing error from any error type
    pub fn processing<E: std::fmt::Display>(source: E) -> Self {
        Self::ProcessingError(source.to_string())
    }

    /// Create an internal error with a message
    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Create an internal error from any error, sanitizing the message
    pub fn internal_from<E: std::error::Error>(err: E) -> Self {
        // Log the full error for debugging
        error!(error = %err, "Internal error occurred");
        Self::InternalError {
            message: "An unexpected error occurred".to_string(),
        }
    }

    /// Create a service unavailable error
    pub fn service_unavailable(reason: impl Into<String>) -> Self {
        Self::ServiceUnavailable {
            reason: reason.into(),
        }
    }

    /// Create a gateway timeout error
    pub fn gateway_timeout(timeout_ms: u64) -> Self {
        Self::GatewayTimeout { timeout_ms }
    }

    // ========================================================================
    // HTTP Status Code Mapping
    // ========================================================================

    /// Get the HTTP status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            Self::InvalidRequest { .. } => 400,
            Self::Unauthorized { .. } => 401,
            Self::Forbidden { .. } => 403,
            Self::NotFound { .. } => 404,
            Self::ContentTooLarge { .. } => 413,
            Self::RateLimited { .. } => 429,
            Self::ProcessingError(_) => 500,
            Self::InternalError { .. } => 500,
            Self::ServiceUnavailable { .. } => 503,
            Self::GatewayTimeout { .. } => 504,
        }
    }

    /// Get the error code string for this error
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::InvalidRequest { .. } => "INVALID_REQUEST",
            Self::Unauthorized { .. } => "UNAUTHORIZED",
            Self::Forbidden { .. } => "FORBIDDEN",
            Self::NotFound { .. } => "NOT_FOUND",
            Self::ContentTooLarge { .. } => "CONTENT_TOO_LARGE",
            Self::RateLimited { .. } => "RATE_LIMITED",
            Self::ProcessingError(_) => "PROCESSING_ERROR",
            Self::InternalError { .. } => "INTERNAL_ERROR",
            Self::ServiceUnavailable { .. } => "SERVICE_UNAVAILABLE",
            Self::GatewayTimeout { .. } => "GATEWAY_TIMEOUT",
        }
    }

    /// Convert error to JSON response body
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "error": self.to_string(),
            "code": self.error_code()
        })
    }

    /// Convert error to JSON response body with request ID
    pub fn to_json_with_request_id(&self, request_id: &str) -> serde_json::Value {
        json!({
            "error": self.to_string(),
            "code": self.error_code(),
            "request_id": request_id
        })
    }

    /// Log the error with appropriate level and optional request ID
    pub fn log(&self, request_id: Option<&str>) {
        let request_id = request_id.unwrap_or("unknown");

        match self {
            Self::InvalidRequest { message } => {
                warn!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    message = %message,
                    "Invalid request"
                );
            }
            Self::Unauthorized { reason } => {
                warn!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    reason = %reason,
                    "Unauthorized access attempt"
                );
            }
            Self::Forbidden { reason } => {
                warn!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    reason = %reason,
                    "Forbidden access"
                );
            }
            Self::NotFound { resource } => {
                warn!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    resource = %resource,
                    "Resource not found"
                );
            }
            Self::ContentTooLarge { size, max } => {
                warn!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    size = %size,
                    max = %max,
                    "Content too large"
                );
            }
            Self::RateLimited { retry_after_secs } => {
                warn!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    retry_after_secs = %retry_after_secs,
                    "Rate limited"
                );
            }
            Self::ProcessingError(err) => {
                error!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    error = %err,
                    "Processing error"
                );
            }
            Self::InternalError { message } => {
                error!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    message = %message,
                    "Internal error"
                );
            }
            Self::ServiceUnavailable { reason } => {
                error!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    reason = %reason,
                    "Service unavailable"
                );
            }
            Self::GatewayTimeout { timeout_ms } => {
                error!(
                    request_id = %request_id,
                    error_code = %self.error_code(),
                    timeout_ms = %timeout_ms,
                    "Gateway timeout"
                );
            }
        }
    }

    /// Check if this error should be retried
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. }
                | Self::ServiceUnavailable { .. }
                | Self::GatewayTimeout { .. }
        )
    }

    /// Get retry-after header value if applicable
    pub fn retry_after(&self) -> Option<u64> {
        match self {
            Self::RateLimited { retry_after_secs } => Some(*retry_after_secs),
            _ => None,
        }
    }
}

/// Structured JSON error response for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Human-readable error message
    pub error: String,
    /// Machine-readable error code
    pub code: String,
    /// Optional request ID for correlation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            request_id: None,
            details: None,
        }
    }

    /// Set the request ID
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Set additional details
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

impl From<&WebError> for ErrorResponse {
    fn from(err: &WebError) -> Self {
        Self {
            error: err.to_string(),
            code: err.error_code().to_string(),
            request_id: None,
            details: None,
        }
    }
}

// ============================================================================
// Internal Error - Comprehensive domain errors
// ============================================================================

/// The main error type for ReasonKit Web internal operations
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP/Web-level errors
    #[error("{0}")]
    Web(#[from] WebError),

    /// Browser-related errors
    #[error("Browser error: {0}")]
    Browser(#[from] BrowserError),

    /// MCP protocol errors
    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),

    /// Content extraction errors
    #[error("Extraction error: {0}")]
    Extraction(#[from] ExtractionError),

    /// Navigation errors
    #[error("Navigation error: {0}")]
    Navigation(#[from] NavigationError),

    /// Capture errors (screenshot, PDF, etc.)
    #[error("Capture error: {0}")]
    Capture(#[from] CaptureError),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// ChromiumOxide errors
    #[error("CDP error: {0}")]
    Cdp(String),

    /// Generic error with message
    #[error("{0}")]
    Generic(String),
}

/// Browser lifecycle and control errors
#[derive(Error, Debug)]
pub enum BrowserError {
    /// Failed to launch browser
    #[error("Failed to launch browser: {0}")]
    LaunchFailed(String),

    /// Browser configuration error
    #[error("Invalid browser configuration: {0}")]
    ConfigError(String),

    /// Browser connection lost
    #[error("Browser connection lost")]
    ConnectionLost,

    /// Failed to create new page/tab
    #[error("Failed to create page: {0}")]
    PageCreationFailed(String),

    /// Browser already closed
    #[error("Browser already closed")]
    AlreadyClosed,

    /// Timeout waiting for browser
    #[error("Browser operation timed out after {0}ms")]
    Timeout(u64),
}

/// MCP protocol errors
#[derive(Error, Debug)]
pub enum McpError {
    /// Invalid JSON-RPC request
    #[error("Invalid JSON-RPC request: {0}")]
    InvalidRequest(String),

    /// Unknown method
    #[error("Unknown method: {0}")]
    UnknownMethod(String),

    /// Invalid parameters
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    ToolExecutionFailed(String),

    /// Protocol version mismatch
    #[error("Protocol version mismatch: expected {expected}, got {actual}")]
    VersionMismatch {
        /// Expected version
        expected: String,
        /// Actual version received
        actual: String,
    },

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Content extraction errors
#[derive(Error, Debug)]
pub enum ExtractionError {
    /// Element not found
    #[error("Element not found: {0}")]
    ElementNotFound(String),

    /// Invalid selector
    #[error("Invalid selector: {0}")]
    InvalidSelector(String),

    /// Extraction failed
    #[error("Extraction failed: {0}")]
    ExtractionFailed(String),

    /// Content parsing failed
    #[error("Content parsing failed: {0}")]
    ParsingFailed(String),

    /// JavaScript execution failed
    #[error("JavaScript execution failed: {0}")]
    JsExecutionFailed(String),
}

/// Navigation errors
#[derive(Error, Debug)]
pub enum NavigationError {
    /// Invalid URL
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// Navigation timeout
    #[error("Navigation timed out after {0}ms")]
    Timeout(u64),

    /// Page load failed
    #[error("Page load failed: {0}")]
    LoadFailed(String),

    /// SSL/TLS error
    #[error("SSL/TLS error: {0}")]
    SslError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// HTTP error
    #[error("HTTP error {status}: {message}")]
    HttpError {
        /// HTTP status code
        status: u16,
        /// Error message
        message: String,
    },
}

/// Capture errors (screenshots, PDFs, etc.)
#[derive(Error, Debug)]
pub enum CaptureError {
    /// Screenshot failed
    #[error("Screenshot capture failed: {0}")]
    ScreenshotFailed(String),

    /// PDF generation failed
    #[error("PDF generation failed: {0}")]
    PdfFailed(String),

    /// MHTML capture failed
    #[error("MHTML capture failed: {0}")]
    MhtmlFailed(String),

    /// HTML capture failed
    #[error("HTML capture failed: {0}")]
    HtmlFailed(String),

    /// Invalid capture format
    #[error("Invalid capture format: {0}")]
    InvalidFormat(String),

    /// Capture timeout
    #[error("Capture timed out after {0}ms")]
    Timeout(u64),
}

/// Result type alias for ReasonKit Web operations
pub type Result<T> = std::result::Result<T, Error>;

/// Result type alias for WebError operations
pub type WebResult<T> = std::result::Result<T, WebError>;

impl Error {
    /// Create a generic error from a string
    pub fn generic<S: Into<String>>(msg: S) -> Self {
        Error::Generic(msg.into())
    }

    /// Create a CDP error from a string
    pub fn cdp<S: Into<String>>(msg: S) -> Self {
        Error::Cdp(msg.into())
    }

    /// Convert internal error to WebError for HTTP responses
    pub fn into_web_error(self) -> WebError {
        match self {
            Error::Web(e) => e,
            Error::Browser(e) => match e {
                BrowserError::Timeout(ms) => WebError::GatewayTimeout { timeout_ms: ms },
                BrowserError::ConnectionLost => {
                    WebError::service_unavailable("Browser connection lost")
                }
                BrowserError::AlreadyClosed => {
                    WebError::service_unavailable("Browser session closed")
                }
                _ => WebError::internal(e.to_string()),
            },
            Error::Mcp(e) => match e {
                McpError::InvalidRequest(msg) => WebError::invalid_request(msg),
                McpError::InvalidParams(msg) => WebError::invalid_request(msg),
                McpError::ToolNotFound(tool) => {
                    WebError::not_found(format!("Tool not found: {}", tool))
                }
                _ => WebError::internal(e.to_string()),
            },
            Error::Navigation(e) => match e {
                NavigationError::InvalidUrl(url) => {
                    WebError::invalid_request(format!("Invalid URL: {}", url))
                }
                NavigationError::Timeout(ms) => WebError::GatewayTimeout { timeout_ms: ms },
                NavigationError::HttpError { status, message } => {
                    if status == 404 {
                        WebError::not_found(message)
                    } else if status == 401 {
                        WebError::unauthorized(message)
                    } else if status == 403 {
                        WebError::forbidden(message)
                    } else if status == 429 {
                        WebError::rate_limited(60) // Default retry
                    } else {
                        WebError::internal(format!("HTTP {}: {}", status, message))
                    }
                }
                _ => WebError::internal(e.to_string()),
            },
            Error::Extraction(e) => match e {
                ExtractionError::ElementNotFound(selector) => {
                    WebError::not_found(format!("Element not found: {}", selector))
                }
                ExtractionError::InvalidSelector(sel) => {
                    WebError::invalid_request(format!("Invalid selector: {}", sel))
                }
                _ => WebError::internal(e.to_string()),
            },
            Error::Capture(e) => match e {
                CaptureError::Timeout(ms) => WebError::GatewayTimeout { timeout_ms: ms },
                CaptureError::InvalidFormat(fmt) => {
                    WebError::invalid_request(format!("Invalid format: {}", fmt))
                }
                _ => WebError::internal(e.to_string()),
            },
            Error::Io(e) => WebError::internal(format!("I/O error: {}", e)),
            Error::Json(e) => WebError::invalid_request(format!("JSON error: {}", e)),
            Error::Cdp(msg) => WebError::internal(format!("CDP error: {}", msg)),
            Error::Generic(msg) => WebError::internal(msg),
        }
    }
}

/// Convert chromiumoxide errors
impl From<chromiumoxide::error::CdpError> for Error {
    fn from(err: chromiumoxide::error::CdpError) -> Self {
        Error::Cdp(err.to_string())
    }
}

// ============================================================================
// Conversion implementations for WebError
// ============================================================================

impl From<std::io::Error> for WebError {
    fn from(err: std::io::Error) -> Self {
        WebError::internal(format!("I/O error: {}", err))
    }
}

impl From<serde_json::Error> for WebError {
    fn from(err: serde_json::Error) -> Self {
        WebError::invalid_request(format!("JSON error: {}", err))
    }
}

impl From<anyhow::Error> for WebError {
    fn from(err: anyhow::Error) -> Self {
        WebError::ProcessingError(err.to_string())
    }
}

// ============================================================================
// Request ID generation
// ============================================================================

/// Generate a unique request ID for tracing
pub fn generate_request_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let id: u64 = rng.gen();
    format!("req_{:016x}", id)
}

/// Request context for error handling
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request ID
    pub request_id: String,
    /// Request start time
    pub start_time: std::time::Instant,
}

impl RequestContext {
    /// Create a new request context with a generated ID
    pub fn new() -> Self {
        Self {
            request_id: generate_request_id(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Create a new request context with a specific ID
    pub fn with_id(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Log an error with this request context
    pub fn log_error(&self, error: &WebError) {
        error.log(Some(&self.request_id));
    }
}

impl Default for RequestContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.request_id)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_error_status_codes() {
        assert_eq!(WebError::invalid_request("test").status_code(), 400);
        assert_eq!(WebError::unauthorized("test").status_code(), 401);
        assert_eq!(WebError::forbidden("test").status_code(), 403);
        assert_eq!(WebError::not_found("test").status_code(), 404);
        assert_eq!(WebError::content_too_large(100, 50).status_code(), 413);
        assert_eq!(WebError::rate_limited(60).status_code(), 429);
        assert_eq!(WebError::internal("test").status_code(), 500);
        assert_eq!(WebError::service_unavailable("test").status_code(), 503);
        assert_eq!(WebError::gateway_timeout(5000).status_code(), 504);
    }

    #[test]
    fn test_web_error_codes() {
        assert_eq!(
            WebError::invalid_request("test").error_code(),
            "INVALID_REQUEST"
        );
        assert_eq!(WebError::unauthorized("test").error_code(), "UNAUTHORIZED");
        assert_eq!(WebError::forbidden("test").error_code(), "FORBIDDEN");
        assert_eq!(WebError::not_found("test").error_code(), "NOT_FOUND");
        assert_eq!(
            WebError::content_too_large(100, 50).error_code(),
            "CONTENT_TOO_LARGE"
        );
        assert_eq!(WebError::rate_limited(60).error_code(), "RATE_LIMITED");
        assert_eq!(WebError::internal("test").error_code(), "INTERNAL_ERROR");
    }

    #[test]
    fn test_web_error_json() {
        let err = WebError::invalid_request("Missing name field");
        let json = err.to_json();

        assert_eq!(json["code"], "INVALID_REQUEST");
        assert!(json["error"]
            .as_str()
            .unwrap()
            .contains("Missing name field"));
    }

    #[test]
    fn test_web_error_json_with_request_id() {
        let err = WebError::rate_limited(120);
        let json = err.to_json_with_request_id("req_abc123");

        assert_eq!(json["code"], "RATE_LIMITED");
        assert_eq!(json["request_id"], "req_abc123");
    }

    #[test]
    fn test_web_error_factory_methods() {
        let err = WebError::missing_field("email");
        assert!(err.to_string().contains("Missing required field: email"));

        let err = WebError::invalid_field("age", "must be positive");
        assert!(err.to_string().contains("Invalid value for field 'age'"));
    }

    #[test]
    fn test_web_error_retryable() {
        assert!(!WebError::invalid_request("test").is_retryable());
        assert!(!WebError::unauthorized("test").is_retryable());
        assert!(WebError::rate_limited(60).is_retryable());
        assert!(WebError::service_unavailable("test").is_retryable());
        assert!(WebError::gateway_timeout(5000).is_retryable());
    }

    #[test]
    fn test_web_error_retry_after() {
        assert_eq!(WebError::rate_limited(120).retry_after(), Some(120));
        assert_eq!(WebError::invalid_request("test").retry_after(), None);
    }

    #[test]
    fn test_error_display() {
        let err = Error::Browser(BrowserError::LaunchFailed("no chrome".to_string()));
        assert!(err.to_string().contains("Failed to launch browser"));
        assert!(err.to_string().contains("no chrome"));
    }

    #[test]
    fn test_mcp_error() {
        let err = McpError::ToolNotFound("unknown_tool".to_string());
        assert_eq!(err.to_string(), "Tool not found: unknown_tool");
    }

    #[test]
    fn test_extraction_error() {
        let err = ExtractionError::ElementNotFound("#missing".to_string());
        assert!(err.to_string().contains("Element not found"));
    }

    #[test]
    fn test_navigation_error() {
        let err = NavigationError::HttpError {
            status: 404,
            message: "Not Found".to_string(),
        };
        assert!(err.to_string().contains("404"));
        assert!(err.to_string().contains("Not Found"));
    }

    #[test]
    fn test_generic_error() {
        let err = Error::generic("something went wrong");
        assert_eq!(err.to_string(), "something went wrong");
    }

    #[test]
    fn test_error_into_web_error() {
        let err = Error::Navigation(NavigationError::InvalidUrl("bad-url".to_string()));
        let web_err = err.into_web_error();
        assert_eq!(web_err.status_code(), 400);
        assert!(web_err.to_string().contains("Invalid URL"));

        let err = Error::Navigation(NavigationError::Timeout(5000));
        let web_err = err.into_web_error();
        assert_eq!(web_err.status_code(), 504);
    }

    #[test]
    fn test_error_response_serialization() {
        let response = ErrorResponse::new("Test error", "TEST_ERROR")
            .with_request_id("req_123")
            .with_details(json!({"field": "name"}));

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("TEST_ERROR"));
        assert!(json.contains("req_123"));
        assert!(json.contains("name"));
    }

    #[test]
    fn test_request_context() {
        let ctx = RequestContext::new();
        assert!(ctx.request_id.starts_with("req_"));
        assert_eq!(ctx.request_id.len(), 20); // "req_" + 16 hex chars

        let ctx = RequestContext::with_id("custom-id-123");
        assert_eq!(ctx.request_id, "custom-id-123");
    }

    #[test]
    fn test_generate_request_id() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("req_"));
    }

    #[test]
    fn test_web_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let web_err: WebError = io_err.into();
        assert_eq!(web_err.status_code(), 500);
    }

    #[test]
    fn test_content_too_large_display() {
        let err = WebError::content_too_large(1024 * 1024 * 10, 1024 * 1024);
        assert!(err.to_string().contains("10485760"));
        assert!(err.to_string().contains("1048576"));
    }
}

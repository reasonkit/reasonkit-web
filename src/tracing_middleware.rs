//! Request tracing and logging middleware for HTTP servers
//!
//! This module provides production-ready request tracing middleware compatible
//! with Axum 0.7, including:
//! - Request ID generation and propagation
//! - Structured JSON logging for production
//! - Pretty logging for development
//! - Client IP anonymization (GDPR compliant)
//! - Integration with tower-http tracing layer
//!
//! # Architecture
//!
//! ```text
//! Request ──> RequestIdLayer ──> TracingLayer ──> Handler
//!                  │                    │
//!                  ▼                    ▼
//!           X-Request-ID         Structured Logs
//!           Header Added         (JSON/Pretty)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use axum::{Router, routing::get};
//! use reasonkit_web::tracing_middleware::{
//!     init_tracing, RequestIdLayer, request_tracing_layer
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialize tracing (JSON in prod, pretty in dev)
//!     init_tracing();
//!
//!     let app = Router::new()
//!         .route("/", get(|| async { "Hello" }))
//!         .layer(request_tracing_layer())
//!         .layer(RequestIdLayer::new());
//!
//!     // Run server...
//! }
//! ```

use axum::{
    extract::ConnectInfo,
    http::{header::HeaderName, HeaderValue, Request, Response},
};
use std::{
    fmt,
    net::SocketAddr,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tower::{Layer, Service};
use tower_http::{
    classify::{ServerErrorsAsFailures, SharedClassifier},
    trace::{DefaultOnRequest, DefaultOnResponse, MakeSpan, OnRequest, OnResponse, TraceLayer},
};
use tracing::{info_span, Level, Span};
use tracing_subscriber::{
    fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};
use uuid::Uuid;

// ============================================================================
// Constants
// ============================================================================

/// Header name for request ID
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Environment variable for log format
pub const LOG_FORMAT_ENV: &str = "RK_LOG_FORMAT";

/// Environment variable for log level (uses RUST_LOG)
pub const LOG_LEVEL_ENV: &str = "RUST_LOG";

// ============================================================================
// Request ID Types
// ============================================================================

/// A unique identifier for a request
#[derive(Clone, Debug)]
pub struct RequestId(Arc<String>);

impl RequestId {
    /// Generate a new random request ID
    pub fn new() -> Self {
        Self(Arc::new(Uuid::new_v4().to_string()))
    }

    /// Create a request ID from an existing string
    pub fn from_string<S: Into<String>>(id: S) -> Self {
        Self(Arc::new(id.into()))
    }

    /// Get the request ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for RequestId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Request ID Layer
// ============================================================================

/// Layer that generates and attaches a unique request ID to each request
///
/// The request ID is:
/// - Generated as a UUID v4 if not present in incoming request
/// - Extracted from `X-Request-ID` header if present
/// - Added to response headers
/// - Available in tracing spans
#[derive(Clone, Debug)]
pub struct RequestIdLayer {
    header_name: HeaderName,
}

impl RequestIdLayer {
    /// Create a new request ID layer with default header name
    pub fn new() -> Self {
        Self {
            header_name: HeaderName::from_static(REQUEST_ID_HEADER),
        }
    }

    /// Create a new request ID layer with a custom header name
    pub fn with_header_name(name: HeaderName) -> Self {
        Self { header_name: name }
    }
}

impl Default for RequestIdLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdService {
            inner,
            header_name: self.header_name.clone(),
        }
    }
}

/// Service that generates and propagates request IDs
#[derive(Clone, Debug)]
pub struct RequestIdService<S> {
    inner: S,
    header_name: HeaderName,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for RequestIdService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send,
    ReqBody: Send + 'static,
    ResBody: Default + Send + 'static,
{
    type Response = Response<ResBody>;
    type Error = S::Error;
    type Future = RequestIdFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<ReqBody>) -> Self::Future {
        // Extract or generate request ID
        let request_id = request
            .headers()
            .get(&self.header_name)
            .and_then(|v| v.to_str().ok())
            .map(|s| RequestId::from_string(s.to_string()))
            .unwrap_or_else(RequestId::new);

        // Store request ID in extensions for later use
        request.extensions_mut().insert(request_id.clone());

        RequestIdFuture {
            future: self.inner.call(request),
            request_id,
            header_name: self.header_name.clone(),
        }
    }
}

/// Future for request ID service
#[pin_project::pin_project]
pub struct RequestIdFuture<F> {
    #[pin]
    future: F,
    request_id: RequestId,
    header_name: HeaderName,
}

impl<F, ResBody, E> std::future::Future for RequestIdFuture<F>
where
    F: std::future::Future<Output = Result<Response<ResBody>, E>>,
    ResBody: Default,
{
    type Output = Result<Response<ResBody>, E>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        match this.future.poll(cx) {
            Poll::Ready(Ok(mut response)) => {
                // Add request ID to response headers
                if let Ok(value) = HeaderValue::from_str(this.request_id.as_str()) {
                    response
                        .headers_mut()
                        .insert(this.header_name.clone(), value);
                }
                Poll::Ready(Ok(response))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

// ============================================================================
// Custom Span Maker
// ============================================================================

/// Custom span maker that includes request ID and anonymized client IP
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Extract request ID from extensions
        let request_id = request
            .extensions()
            .get::<RequestId>()
            .map(|id| id.as_str().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Get method and path
        let method = request.method().as_str();
        let path = request.uri().path();
        let version = format!("{:?}", request.version());

        // Anonymize client IP if present
        let client_ip = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ConnectInfo(addr)| anonymize_ip(*addr))
            .unwrap_or_else(|| "unknown".to_string());

        info_span!(
            "http_request",
            request_id = %request_id,
            method = %method,
            path = %path,
            version = %version,
            client_ip = %client_ip,
            status = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
        )
    }
}

/// Anonymize IP address for GDPR compliance
///
/// - IPv4: Zero out last octet (192.168.1.100 -> 192.168.1.0)
/// - IPv6: Zero out last 80 bits
fn anonymize_ip(addr: SocketAddr) -> String {
    match addr {
        SocketAddr::V4(v4) => {
            let ip = v4.ip();
            let octets = ip.octets();
            format!("{}.{}.{}.0", octets[0], octets[1], octets[2])
        }
        SocketAddr::V6(v6) => {
            let ip = v6.ip();
            let segments = ip.segments();
            format!(
                "{:x}:{:x}:{:x}:0:0:0:0:0",
                segments[0], segments[1], segments[2]
            )
        }
    }
}

// ============================================================================
// Custom Request/Response Handlers
// ============================================================================

/// Custom on-request handler for structured logging
#[derive(Clone, Debug)]
pub struct OnRequestLog;

impl<B> OnRequest<B> for OnRequestLog {
    fn on_request(&mut self, request: &Request<B>, _span: &Span) {
        let method = request.method();
        let uri = request.uri();
        let request_id = request
            .extensions()
            .get::<RequestId>()
            .map(|id| id.as_str())
            .unwrap_or("unknown");

        tracing::info!(
            target: "http::request",
            request_id = %request_id,
            method = %method,
            uri = %uri,
            "incoming request"
        );
    }
}

/// Custom on-response handler with timing information
#[derive(Clone, Debug)]
pub struct OnResponseLog {
    #[allow(dead_code)]
    start: Option<Instant>,
}

impl OnResponseLog {
    /// Create a new response logger
    pub fn new() -> Self {
        Self { start: None }
    }
}

impl Default for OnResponseLog {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> OnResponse<B> for OnResponseLog {
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status().as_u16();
        let duration_ms = latency.as_secs_f64() * 1000.0;

        // Record status and duration in span
        span.record("status", status);
        span.record("duration_ms", duration_ms);

        // Log based on status code
        if status >= 500 {
            tracing::error!(
                target: "http::response",
                status = status,
                duration_ms = duration_ms,
                "server error response"
            );
        } else if status >= 400 {
            tracing::warn!(
                target: "http::response",
                status = status,
                duration_ms = duration_ms,
                "client error response"
            );
        } else {
            tracing::info!(
                target: "http::response",
                status = status,
                duration_ms = duration_ms,
                "response completed"
            );
        }
    }
}

// ============================================================================
// Tracing Layer Builder
// ============================================================================

/// Create the request tracing layer with custom span maker and loggers
///
/// This layer integrates with tower-http to provide:
/// - Request/response logging
/// - Timing information
/// - Integration with tracing spans
pub fn request_tracing_layer(
) -> TraceLayer<SharedClassifier<ServerErrorsAsFailures>, RequestSpan, OnRequestLog, OnResponseLog>
{
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(OnRequestLog)
        .on_response(OnResponseLog::new())
}

/// Create a simple tracing layer with default handlers
pub fn simple_tracing_layer() -> TraceLayer<SharedClassifier<ServerErrorsAsFailures>> {
    TraceLayer::new_for_http()
        .on_request(DefaultOnRequest::new().level(Level::INFO))
        .on_response(DefaultOnResponse::new().level(Level::INFO))
}

// ============================================================================
// Tracing Initialization
// ============================================================================

/// Log format for tracing output
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogFormat {
    /// JSON format for production (structured, machine-readable)
    Json,
    /// Pretty format for development (human-readable, colored)
    #[default]
    Pretty,
    /// Compact format (single-line, minimal)
    Compact,
}

impl LogFormat {
    /// Parse from environment or string
    pub fn from_env() -> Self {
        std::env::var(LOG_FORMAT_ENV)
            .ok()
            .and_then(|s| Self::from_str(&s))
            .unwrap_or_else(|| {
                // Default to JSON in production, pretty in development
                if cfg!(debug_assertions) {
                    LogFormat::Pretty
                } else {
                    LogFormat::Json
                }
            })
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(LogFormat::Json),
            "pretty" => Some(LogFormat::Pretty),
            "compact" => Some(LogFormat::Compact),
            _ => None,
        }
    }
}

/// Initialize the tracing subscriber with appropriate format
///
/// This function:
/// - Reads `RUST_LOG` for log level filtering (default: `info`)
/// - Reads `RK_LOG_FORMAT` for output format (`json`, `pretty`, `compact`)
/// - Defaults to JSON in release builds, pretty in debug builds
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::tracing_middleware::init_tracing;
///
/// // Set environment variables before calling
/// std::env::set_var("RUST_LOG", "info,tower_http=debug");
/// std::env::set_var("RK_LOG_FORMAT", "json");
///
/// init_tracing();
/// ```
pub fn init_tracing() {
    init_tracing_with_format(LogFormat::from_env());
}

/// Initialize tracing with a specific format
pub fn init_tracing_with_format(format: LogFormat) {
    // Build the env filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug,hyper=info"));

    match format {
        LogFormat::Json => init_json_tracing(filter),
        LogFormat::Pretty => init_pretty_tracing(filter),
        LogFormat::Compact => init_compact_tracing(filter),
    }
}

/// Initialize JSON-formatted tracing (production)
fn init_json_tracing(filter: EnvFilter) {
    let subscriber = tracing_subscriber::registry().with(filter).with(
        tracing_subscriber::fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(false)
            .with_target(true)
            .with_file(false)
            .with_line_number(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .flatten_event(true)
            .with_writer(std::io::stderr),
    );

    subscriber.init();
}

/// Initialize pretty-formatted tracing (development)
fn init_pretty_tracing(filter: EnvFilter) {
    let subscriber = tracing_subscriber::registry().with(filter).with(
        tracing_subscriber::fmt::layer()
            .pretty()
            .with_ansi(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .with_thread_ids(false)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::stderr),
    );

    subscriber.init();
}

/// Initialize compact tracing format
fn init_compact_tracing(filter: EnvFilter) {
    let subscriber = tracing_subscriber::registry().with(filter).with(
        tracing_subscriber::fmt::layer()
            .compact()
            .with_ansi(true)
            .with_target(false)
            .with_file(false)
            .with_line_number(false)
            .with_writer(std::io::stderr),
    );

    subscriber.init();
}

// ============================================================================
// Middleware Timing
// ============================================================================

/// Middleware for capturing request timing
///
/// This is an alternative to the tower-http tracing layer that provides
/// more fine-grained control over timing.
#[derive(Clone, Debug)]
pub struct TimingLayer;

impl<S> Layer<S> for TimingLayer {
    type Service = TimingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TimingService { inner }
    }
}

/// Service that captures request timing
#[derive(Clone, Debug)]
pub struct TimingService<S> {
    inner: S,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for TimingService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send,
    ReqBody: Send + 'static,
    ResBody: Send + 'static,
{
    type Response = Response<ResBody>;
    type Error = S::Error;
    type Future = TimingFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<ReqBody>) -> Self::Future {
        let start = Instant::now();
        let method = request.method().clone();
        let path = request.uri().path().to_string();
        let request_id = request
            .extensions()
            .get::<RequestId>()
            .map(|id| id.to_string())
            .unwrap_or_default();

        TimingFuture {
            future: self.inner.call(request),
            start,
            method,
            path,
            request_id,
        }
    }
}

/// Future for timing service
#[pin_project::pin_project]
pub struct TimingFuture<F> {
    #[pin]
    future: F,
    start: Instant,
    method: axum::http::Method,
    path: String,
    request_id: String,
}

impl<F, ResBody, E> std::future::Future for TimingFuture<F>
where
    F: std::future::Future<Output = Result<Response<ResBody>, E>>,
{
    type Output = Result<Response<ResBody>, E>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        match this.future.poll(cx) {
            Poll::Ready(result) => {
                let duration = this.start.elapsed();
                let duration_ms = duration.as_secs_f64() * 1000.0;

                match &result {
                    Ok(response) => {
                        let status = response.status().as_u16();
                        tracing::info!(
                            target: "http::timing",
                            request_id = %this.request_id,
                            method = %this.method,
                            path = %this.path,
                            status = status,
                            duration_ms = duration_ms,
                            "request completed"
                        );
                    }
                    Err(_) => {
                        tracing::error!(
                            target: "http::timing",
                            request_id = %this.request_id,
                            method = %this.method,
                            path = %this.path,
                            duration_ms = duration_ms,
                            "request failed"
                        );
                    }
                }

                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Extract request ID from request extensions
pub fn get_request_id<B>(request: &Request<B>) -> Option<&RequestId> {
    request.extensions().get::<RequestId>()
}

/// Create a request span with custom fields
pub fn create_request_span(request_id: &str, method: &str, path: &str) -> Span {
    info_span!(
        "request",
        request_id = %request_id,
        method = %method,
        path = %path,
        status = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
    )
}

// ============================================================================
// JSON Log Types (for serialization reference)
// ============================================================================

/// Structured log entry format
///
/// This represents the JSON structure of log entries when using JSON format.
/// Provided for documentation and potential deserialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Log level (TRACE, DEBUG, INFO, WARN, ERROR)
    pub level: String,
    /// Unique request identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// HTTP method
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Request path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// HTTP status code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Request duration in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Log message
    pub message: String,
    /// Log target (module path)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Anonymized client IP
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn test_request_id_generation() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();

        // Should be unique
        assert_ne!(id1.as_str(), id2.as_str());

        // Should be valid UUIDs (36 chars)
        assert_eq!(id1.as_str().len(), 36);
        assert!(id1.as_str().contains('-'));
    }

    #[test]
    fn test_request_id_from_string() {
        let id = RequestId::from_string("custom-request-id");
        assert_eq!(id.as_str(), "custom-request-id");
    }

    #[test]
    fn test_request_id_display() {
        let id = RequestId::from_string("test-id-123");
        assert_eq!(format!("{}", id), "test-id-123");
    }

    #[test]
    fn test_anonymize_ipv4() {
        let addr = SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)), 8080);
        let anon = anonymize_ip(addr);
        assert_eq!(anon, "192.168.1.0");
    }

    #[test]
    fn test_anonymize_ipv6() {
        let addr = SocketAddr::new(
            std::net::IpAddr::V6(Ipv6Addr::new(
                0x2001, 0x0db8, 0x85a3, 0x0000, 0x0000, 0x8a2e, 0x0370, 0x7334,
            )),
            8080,
        );
        let anon = anonymize_ip(addr);
        assert_eq!(anon, "2001:db8:85a3:0:0:0:0:0");
    }

    #[test]
    fn test_log_format_from_str() {
        assert_eq!(LogFormat::from_str("json"), Some(LogFormat::Json));
        assert_eq!(LogFormat::from_str("JSON"), Some(LogFormat::Json));
        assert_eq!(LogFormat::from_str("pretty"), Some(LogFormat::Pretty));
        assert_eq!(LogFormat::from_str("compact"), Some(LogFormat::Compact));
        assert_eq!(LogFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_log_format_default() {
        // In debug mode, default should be Pretty
        #[cfg(debug_assertions)]
        assert_eq!(LogFormat::default(), LogFormat::Pretty);
    }

    #[test]
    fn test_log_entry_serialization() {
        let entry = LogEntry {
            timestamp: "2024-01-15T10:30:00Z".to_string(),
            level: "INFO".to_string(),
            request_id: Some("abc-123".to_string()),
            method: Some("POST".to_string()),
            path: Some("/capture".to_string()),
            status: Some(200),
            duration_ms: Some(42.5),
            message: "request completed".to_string(),
            target: Some("http::response".to_string()),
            client_ip: Some("192.168.1.0".to_string()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"request_id\":\"abc-123\""));
        assert!(json.contains("\"duration_ms\":42.5"));
        assert!(json.contains("\"status\":200"));
    }

    #[test]
    fn test_request_id_layer_creation() {
        let layer = RequestIdLayer::new();
        assert_eq!(layer.header_name.as_str(), REQUEST_ID_HEADER);
    }

    #[test]
    fn test_custom_header_name() {
        let layer = RequestIdLayer::with_header_name(HeaderName::from_static("x-correlation-id"));
        assert_eq!(layer.header_name.as_str(), "x-correlation-id");
    }
}

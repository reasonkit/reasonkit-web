//! Status and health check handlers for the ReasonKit Web MCP server.
//!
//! This module provides HTTP endpoints for monitoring server health and metrics:
//! - `/status` - Detailed server status with runtime metrics
//! - `/health` - Simple health check for systemd/load balancers
//!
//! # Architecture
//!
//! ```text
//! HTTP Request ──> Axum Router ──> status_handler ──> AppState
//!                                        │                │
//!                                        ▼                ▼
//!                              StatusResponse    LatencyHistogram
//!                                        │         + Counters
//!                                        ▼
//!                                   JSON Response
//! ```
//!
//! # Example Response
//!
//! ```json
//! {
//!   "version": "0.1.0",
//!   "uptime_seconds": 3600,
//!   "captures_processed": 1024,
//!   "active_sse_connections": 5,
//!   "memory": {
//!     "rss_bytes": 52428800,
//!     "virtual_bytes": 268435456,
//!     "heap_bytes": 41943040
//!   },
//!   "latency": {
//!     "p50_ms": 12.5,
//!     "p95_ms": 45.2,
//!     "p99_ms": 98.7
//!   }
//! }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use hdrhistogram::Histogram;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};
use tracing::{debug, instrument};

/// Server version from Cargo.toml
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Server name from Cargo.toml
pub const SERVER_NAME: &str = env!("CARGO_PKG_NAME");

// ============================================================================
// Response Types
// ============================================================================

/// Health check response for simple liveness probes.
///
/// Used by systemd, Kubernetes, and load balancers to verify the service is running.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Health status (always "healthy" if responding)
    pub status: String,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self {
            status: "healthy".to_string(),
        }
    }
}

/// Detailed server status response with runtime metrics.
///
/// Provides comprehensive information about the server's current state,
/// resource usage, and performance characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Server version (from Cargo.toml)
    pub version: String,

    /// Server name
    pub name: String,

    /// Server uptime in seconds
    pub uptime_seconds: u64,

    /// Total number of browser captures processed
    pub captures_processed: u64,

    /// Number of currently active SSE connections
    pub active_sse_connections: u64,

    /// Memory usage metrics
    pub memory: MemoryMetrics,

    /// Request latency statistics (percentiles)
    pub latency: LatencyMetrics,

    /// Server status (always "running" if responding)
    pub status: String,

    /// ISO8601 timestamp of when status was generated
    pub timestamp: String,
}

/// Memory usage metrics collected from sysinfo.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Resident Set Size - actual physical memory used (bytes)
    pub rss_bytes: u64,

    /// Virtual memory size (bytes)
    pub virtual_bytes: u64,

    /// CPU usage percentage (0.0 - 100.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_percent: Option<f32>,
}

/// Request latency percentile metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// 50th percentile (median) latency in milliseconds
    pub p50_ms: f64,

    /// 95th percentile latency in milliseconds
    pub p95_ms: f64,

    /// 99th percentile latency in milliseconds
    pub p99_ms: f64,

    /// Total number of requests recorded
    pub total_requests: u64,

    /// Mean latency in milliseconds
    pub mean_ms: f64,

    /// Maximum latency recorded in milliseconds
    pub max_ms: f64,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            total_requests: 0,
            mean_ms: 0.0,
            max_ms: 0.0,
        }
    }
}

// ============================================================================
// Latency Histogram
// ============================================================================

/// Thread-safe latency histogram for recording request timings.
///
/// Uses HdrHistogram for efficient percentile calculations with minimal memory.
/// The histogram tracks latencies from 1 microsecond to 60 seconds with
/// 3 significant figures of precision.
#[derive(Debug)]
pub struct LatencyHistogram {
    /// The underlying HdrHistogram wrapped in RwLock for thread safety
    inner: RwLock<Histogram<u64>>,
}

impl LatencyHistogram {
    /// Create a new latency histogram.
    ///
    /// Tracks latencies from 1us to 60 seconds with 3 significant figures.
    pub fn new() -> Self {
        // Track 1us to 60 seconds with 3 significant figures
        let histogram =
            Histogram::new_with_bounds(1, 60_000_000, 3).expect("Failed to create histogram");
        Self {
            inner: RwLock::new(histogram),
        }
    }

    /// Record a latency value in microseconds.
    ///
    /// Values outside the histogram bounds are silently ignored.
    pub fn record(&self, latency_us: u64) {
        let mut hist = self.inner.write();
        // Ignore errors from values outside bounds
        let _ = hist.record(latency_us);
    }

    /// Record a latency duration.
    ///
    /// Convenience method that converts Duration to microseconds.
    pub fn record_duration(&self, duration: std::time::Duration) {
        self.record(duration.as_micros() as u64);
    }

    /// Get a percentile value in microseconds.
    ///
    /// # Arguments
    /// * `percentile` - The percentile to retrieve (0.0 - 100.0)
    ///
    /// # Returns
    /// The latency value at the given percentile in microseconds, or 0 if empty.
    pub fn percentile(&self, percentile: f64) -> u64 {
        let hist = self.inner.read();
        hist.value_at_percentile(percentile)
    }

    /// Get the total count of recorded values.
    pub fn count(&self) -> u64 {
        let hist = self.inner.read();
        hist.len()
    }

    /// Get the mean latency in microseconds.
    pub fn mean(&self) -> f64 {
        let hist = self.inner.read();
        hist.mean()
    }

    /// Get the maximum recorded latency in microseconds.
    pub fn max(&self) -> u64 {
        let hist = self.inner.read();
        hist.max()
    }

    /// Get complete latency metrics.
    ///
    /// Returns a LatencyMetrics struct with all percentiles converted to milliseconds.
    pub fn metrics(&self) -> LatencyMetrics {
        let hist = self.inner.read();
        LatencyMetrics {
            p50_ms: hist.value_at_percentile(50.0) as f64 / 1000.0,
            p95_ms: hist.value_at_percentile(95.0) as f64 / 1000.0,
            p99_ms: hist.value_at_percentile(99.0) as f64 / 1000.0,
            total_requests: hist.len(),
            mean_ms: hist.mean() / 1000.0,
            max_ms: hist.max() as f64 / 1000.0,
        }
    }

    /// Reset the histogram, clearing all recorded values.
    pub fn reset(&self) {
        let mut hist = self.inner.write();
        hist.reset();
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Application State
// ============================================================================

/// Shared application state for metrics and status tracking.
///
/// This struct holds all the counters and state needed for the status endpoint.
/// All fields are thread-safe and can be accessed concurrently.
///
/// # Thread Safety
///
/// - `start_time`: Immutable after creation
/// - `captures_processed`: AtomicU64 for lock-free increments
/// - `active_sse_connections`: AtomicU64 for connection tracking
/// - `latency_histogram`: RwLock-wrapped for efficient reads
///
/// # Usage
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use reasonkit_web::handlers::AppState;
///
/// let state = Arc::new(AppState::new());
///
/// // Record a capture
/// state.record_capture();
///
/// // Track SSE connection
/// state.increment_sse_connections();
/// // ... later
/// state.decrement_sse_connections();
///
/// // Record request latency
/// state.record_latency_us(12500); // 12.5ms
/// ```
#[derive(Debug)]
pub struct AppState {
    /// Server start time for uptime calculation
    start_time: Instant,

    /// Total number of browser captures processed (atomic for thread safety)
    captures_processed: AtomicU64,

    /// Current number of active SSE connections (atomic for thread safety)
    active_sse_connections: AtomicU64,

    /// Request latency histogram for percentile calculations
    latency_histogram: LatencyHistogram,

    /// Total number of HTTP requests processed
    total_requests: AtomicU64,

    /// Total number of errors encountered
    error_count: AtomicU64,
}

impl AppState {
    /// Create a new AppState instance with initial values.
    ///
    /// The start time is set to the current instant.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            captures_processed: AtomicU64::new(0),
            active_sse_connections: AtomicU64::new(0),
            latency_histogram: LatencyHistogram::new(),
            total_requests: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    /// Get the server uptime in seconds.
    #[inline]
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get the server start time.
    #[inline]
    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    /// Get the total number of captures processed.
    #[inline]
    pub fn captures_processed(&self) -> u64 {
        self.captures_processed.load(Ordering::Relaxed)
    }

    /// Increment the capture counter and return the new value.
    #[inline]
    pub fn record_capture(&self) -> u64 {
        self.captures_processed.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the number of active SSE connections.
    #[inline]
    pub fn active_sse_connections(&self) -> u64 {
        self.active_sse_connections.load(Ordering::Relaxed)
    }

    /// Increment the SSE connection counter.
    #[inline]
    pub fn increment_sse_connections(&self) -> u64 {
        self.active_sse_connections.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement the SSE connection counter.
    ///
    /// Uses saturating subtraction to prevent underflow.
    #[inline]
    pub fn decrement_sse_connections(&self) -> u64 {
        // Use compare-exchange loop to prevent underflow
        loop {
            let current = self.active_sse_connections.load(Ordering::Relaxed);
            if current == 0 {
                return 0;
            }
            match self.active_sse_connections.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current - 1,
                Err(_) => continue,
            }
        }
    }

    /// Record a request latency in microseconds.
    #[inline]
    pub fn record_latency_us(&self, latency_us: u64) {
        self.latency_histogram.record(latency_us);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a request latency duration.
    #[inline]
    pub fn record_latency(&self, duration: std::time::Duration) {
        self.latency_histogram.record_duration(duration);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the latency metrics.
    #[inline]
    pub fn latency_metrics(&self) -> LatencyMetrics {
        self.latency_histogram.metrics()
    }

    /// Get the total number of requests processed.
    #[inline]
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Record an error.
    #[inline]
    pub fn record_error(&self) -> u64 {
        self.error_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the total error count.
    #[inline]
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Reset all metrics (useful for testing).
    pub fn reset_metrics(&self) {
        self.captures_processed.store(0, Ordering::Relaxed);
        self.active_sse_connections.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.latency_histogram.reset();
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// System Metrics Collection
// ============================================================================

/// Collect memory metrics for the current process using sysinfo.
///
/// This function refreshes process information and returns memory usage data.
/// If the process cannot be found, it returns default (zero) values.
fn collect_memory_metrics() -> MemoryMetrics {
    let pid = Pid::from_u32(std::process::id());
    let mut system = System::new();

    // Refresh only the current process with memory info
    // sysinfo 0.33 API: refresh_processes with ProcessesToUpdate
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);

    match system.process(pid) {
        Some(process) => MemoryMetrics {
            rss_bytes: process.memory(),
            virtual_bytes: process.virtual_memory(),
            cpu_percent: None, // CPU requires multiple samples, skip for status
        },
        None => {
            debug!("Could not find current process in sysinfo");
            MemoryMetrics::default()
        }
    }
}

// ============================================================================
// HTTP Handlers
// ============================================================================

/// Health check endpoint handler.
///
/// Returns a simple 200 OK response with `{"status": "healthy"}`.
/// Used by systemd, Kubernetes, and load balancers for liveness probes.
///
/// # Route
/// `GET /health`
///
/// # Response
/// - `200 OK` - Always, if the server is running
///
/// # Example
///
/// ```bash
/// curl http://localhost:8080/health
/// # {"status":"healthy"}
/// ```
#[instrument(skip_all)]
pub async fn health_handler() -> impl IntoResponse {
    debug!("Health check requested");
    (StatusCode::OK, Json(HealthResponse::default()))
}

/// Detailed status endpoint handler.
///
/// Returns comprehensive server status including:
/// - Server version and uptime
/// - Capture and connection counters
/// - Memory usage metrics
/// - Request latency percentiles (p50, p95, p99)
///
/// # Route
/// `GET /status`
///
/// # Arguments
/// * `state` - Shared application state extracted from Axum
///
/// # Response
/// - `200 OK` with JSON StatusResponse
///
/// # Example
///
/// ```bash
/// curl http://localhost:8080/status
/// # {
/// #   "version": "0.1.0",
/// #   "name": "reasonkit-web",
/// #   "uptime_seconds": 3600,
/// #   "captures_processed": 1024,
/// #   "active_sse_connections": 5,
/// #   "memory": {
/// #     "rss_bytes": 52428800,
/// #     "virtual_bytes": 268435456
/// #   },
/// #   "latency": {
/// #     "p50_ms": 12.5,
/// #     "p95_ms": 45.2,
/// #     "p99_ms": 98.7,
/// #     "total_requests": 5000,
/// #     "mean_ms": 18.3,
/// #     "max_ms": 250.0
/// #   },
/// #   "status": "running",
/// #   "timestamp": "2026-01-01T12:00:00Z"
/// # }
/// ```
#[instrument(skip_all)]
pub async fn status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    debug!("Status check requested");

    let memory = collect_memory_metrics();
    let latency = state.latency_metrics();

    let response = StatusResponse {
        version: SERVER_VERSION.to_string(),
        name: SERVER_NAME.to_string(),
        uptime_seconds: state.uptime_seconds(),
        captures_processed: state.captures_processed(),
        active_sse_connections: state.active_sse_connections(),
        memory,
        latency,
        status: "running".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    (StatusCode::OK, Json(response))
}

/// Readiness check endpoint handler.
///
/// Similar to health check but can include additional readiness criteria
/// such as database connections or external service availability.
/// For now, it mirrors the health check behavior.
///
/// # Route
/// `GET /ready`
///
/// # Response
/// - `200 OK` - If ready to serve traffic
/// - `503 Service Unavailable` - If not ready (future implementation)
#[instrument(skip_all)]
pub async fn readiness_handler() -> impl IntoResponse {
    debug!("Readiness check requested");
    // Future: Add checks for browser pool, external services, etc.
    (StatusCode::OK, Json(HealthResponse::default()))
}

// ============================================================================
// Router Setup
// ============================================================================

/// Create the status router with all health and status endpoints.
///
/// # Routes
/// - `GET /health` - Simple health check
/// - `GET /status` - Detailed status with metrics
/// - `GET /ready` - Readiness probe
///
/// # Arguments
/// * `state` - Shared application state
///
/// # Returns
/// An Axum Router configured with all status endpoints
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use axum::Router;
/// use reasonkit_web::handlers::{AppState, status_router};
///
/// let state = Arc::new(AppState::new());
/// let app: Router = Router::new()
///     .merge(status_router(state.clone()))
///     .with_state(state);
/// ```
pub fn status_router(state: Arc<AppState>) -> axum::Router<Arc<AppState>> {
    use axum::routing::get;

    axum::Router::new()
        .route("/health", get(health_handler))
        .route("/status", get(status_handler))
        .route("/ready", get(readiness_handler))
        .with_state(state)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_default() {
        let health = HealthResponse::default();
        assert_eq!(health.status, "healthy");
    }

    #[test]
    fn test_app_state_new() {
        let state = AppState::new();
        assert_eq!(state.captures_processed(), 0);
        assert_eq!(state.active_sse_connections(), 0);
        assert!(state.uptime_seconds() < 1);
    }

    #[test]
    fn test_app_state_capture_counter() {
        let state = AppState::new();

        assert_eq!(state.record_capture(), 1);
        assert_eq!(state.record_capture(), 2);
        assert_eq!(state.record_capture(), 3);
        assert_eq!(state.captures_processed(), 3);
    }

    #[test]
    fn test_app_state_sse_connections() {
        let state = AppState::new();

        assert_eq!(state.increment_sse_connections(), 1);
        assert_eq!(state.increment_sse_connections(), 2);
        assert_eq!(state.active_sse_connections(), 2);

        assert_eq!(state.decrement_sse_connections(), 1);
        assert_eq!(state.active_sse_connections(), 1);

        assert_eq!(state.decrement_sse_connections(), 0);
        assert_eq!(state.active_sse_connections(), 0);

        // Test underflow protection
        assert_eq!(state.decrement_sse_connections(), 0);
        assert_eq!(state.active_sse_connections(), 0);
    }

    #[test]
    fn test_latency_histogram() {
        let histogram = LatencyHistogram::new();

        histogram.record(1000); // 1ms
        histogram.record(2000); // 2ms
        histogram.record(5000); // 5ms
        histogram.record(10000); // 10ms
        histogram.record(50000); // 50ms

        assert_eq!(histogram.count(), 5);
        assert!(histogram.mean() > 0.0);
        // HDRHistogram uses bucketing with some precision loss, so max may be slightly higher
        let max = histogram.max();
        assert!(
            (50000..=51000).contains(&max),
            "max should be ~50000, got {max}"
        );

        let metrics = histogram.metrics();
        assert!(metrics.p50_ms > 0.0);
        assert!(metrics.p95_ms >= metrics.p50_ms);
        assert!(metrics.p99_ms >= metrics.p95_ms);
    }

    #[test]
    fn test_latency_histogram_reset() {
        let histogram = LatencyHistogram::new();

        histogram.record(1000);
        histogram.record(2000);
        assert_eq!(histogram.count(), 2);

        histogram.reset();
        assert_eq!(histogram.count(), 0);
    }

    #[test]
    fn test_app_state_latency_recording() {
        let state = AppState::new();

        state.record_latency_us(5000); // 5ms
        state.record_latency_us(10000); // 10ms

        assert_eq!(state.total_requests(), 2);

        let metrics = state.latency_metrics();
        assert!(metrics.total_requests == 2);
    }

    #[test]
    fn test_app_state_error_tracking() {
        let state = AppState::new();

        assert_eq!(state.error_count(), 0);
        assert_eq!(state.record_error(), 1);
        assert_eq!(state.record_error(), 2);
        assert_eq!(state.error_count(), 2);
    }

    #[test]
    fn test_app_state_reset_metrics() {
        let state = AppState::new();

        state.record_capture();
        state.increment_sse_connections();
        state.record_latency_us(1000);
        state.record_error();

        state.reset_metrics();

        assert_eq!(state.captures_processed(), 0);
        assert_eq!(state.active_sse_connections(), 0);
        assert_eq!(state.total_requests(), 0);
        assert_eq!(state.error_count(), 0);
    }

    #[test]
    fn test_memory_metrics_default() {
        let metrics = MemoryMetrics::default();
        assert_eq!(metrics.rss_bytes, 0);
        assert_eq!(metrics.virtual_bytes, 0);
        assert!(metrics.cpu_percent.is_none());
    }

    #[test]
    fn test_latency_metrics_default() {
        let metrics = LatencyMetrics::default();
        assert_eq!(metrics.p50_ms, 0.0);
        assert_eq!(metrics.p95_ms, 0.0);
        assert_eq!(metrics.p99_ms, 0.0);
        assert_eq!(metrics.total_requests, 0);
    }

    #[test]
    fn test_collect_memory_metrics() {
        // Should not panic
        let metrics = collect_memory_metrics();
        // RSS should be non-zero for a running process
        assert!(metrics.rss_bytes > 0);
    }

    #[test]
    fn test_status_response_serialization() {
        let response = StatusResponse {
            version: "0.1.0".to_string(),
            name: "test-server".to_string(),
            uptime_seconds: 3600,
            captures_processed: 100,
            active_sse_connections: 5,
            memory: MemoryMetrics::default(),
            latency: LatencyMetrics::default(),
            status: "running".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&response).expect("Failed to serialize");
        assert!(json.contains("\"version\":\"0.1.0\""));
        assert!(json.contains("\"uptime_seconds\":3600"));
        assert!(json.contains("\"status\":\"running\""));
    }

    #[test]
    fn test_server_constants() {
        assert!(!SERVER_VERSION.is_empty());
        assert!(!SERVER_NAME.is_empty());
        assert_eq!(SERVER_NAME, "reasonkit-web");
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_handler() {
        let state = Arc::new(AppState::new());

        // Record some test data
        state.record_capture();
        state.record_capture();
        state.increment_sse_connections();
        state.record_latency_us(5000);

        let response = status_handler(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_readiness_handler() {
        let response = readiness_handler().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // Thread safety tests
    #[test]
    fn test_app_state_thread_safety() {
        use std::thread;

        let state = Arc::new(AppState::new());
        let mut handles = vec![];

        // Spawn multiple threads to hammer the state
        for _ in 0..10 {
            let state_clone = Arc::clone(&state);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    state_clone.record_capture();
                    state_clone.increment_sse_connections();
                    state_clone.decrement_sse_connections();
                    state_clone.record_latency_us(1000);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // All captures should be recorded
        assert_eq!(state.captures_processed(), 10_000);
        // All latencies should be recorded
        assert_eq!(state.total_requests(), 10_000);
        // SSE connections should be balanced
        assert_eq!(state.active_sse_connections(), 0);
    }
}

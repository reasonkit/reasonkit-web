//! Metrics Collection for ReasonKit Web Observability
//!
//! This module provides production-ready metrics collection with:
//! - Atomic counters for requests, captures, and errors
//! - Memory-efficient histograms for request duration percentiles
//! - Prometheus-compatible text format export
//! - Optional HTTP /metrics endpoint
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::metrics::{Metrics, MetricsServer};
//! use std::time::Duration;
//!
//! // Global metrics instance
//! static METRICS: Metrics = Metrics::new();
//!
//! // Record a request
//! METRICS.record_request("/capture", 200, Duration::from_millis(150));
//!
//! // Get Prometheus output
//! let output = METRICS.to_prometheus_format();
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Maximum number of duration samples to keep in the histogram
/// This provides a good balance between memory usage and accuracy
const MAX_HISTOGRAM_SAMPLES: usize = 1000;

/// Default buckets for latency histograms (in milliseconds)
const DEFAULT_BUCKETS_MS: &[u64] = &[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

/// Metrics collection for ReasonKit Web observability
///
/// Thread-safe metrics collector using atomics and RwLocks for
/// high-performance concurrent access.
#[derive(Debug)]
pub struct Metrics {
    // === Counters ===
    /// Total number of requests received
    pub requests_total: AtomicU64,
    /// Total number of captures performed (screenshots, PDFs, etc.)
    pub captures_total: AtomicU64,
    /// Total number of errors encountered
    pub errors_total: AtomicU64,
    /// Total number of successful extractions
    pub extractions_total: AtomicU64,
    /// Total number of navigation operations
    pub navigations_total: AtomicU64,

    // === Gauges ===
    /// Current number of active connections
    pub active_connections: AtomicU32,
    /// Current number of active browser pages
    pub active_pages: AtomicU32,

    // === Histograms (memory-efficient ring buffers) ===
    /// Request durations for percentile calculation
    request_durations: RwLock<RingBuffer<Duration>>,

    // === Labeled counters (for detailed breakdowns) ===
    /// Requests broken down by path and status code
    requests_by_path_status: RwLock<HashMap<(String, u16), u64>>,
    /// Errors broken down by type
    errors_by_type: RwLock<HashMap<String, u64>>,
    /// Captures broken down by format (screenshot, pdf, html)
    captures_by_format: RwLock<HashMap<String, u64>>,

    // === Timing ===
    /// When metrics collection started
    start_time: RwLock<Option<Instant>>,
}

/// Memory-efficient ring buffer for histogram samples
#[derive(Debug)]
struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    /// Position of next write (wraps around)
    write_pos: usize,
    /// Total samples received (may exceed capacity)
    total_samples: u64,
}

impl<T: Clone + Ord> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            total_samples: 0,
        }
    }

    fn push(&mut self, value: T) {
        if self.data.len() < self.capacity {
            self.data.push(value);
        } else {
            self.data[self.write_pos] = value;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_samples += 1;
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Get a sorted copy of all samples (for percentile calculation)
    fn sorted_samples(&self) -> Vec<T> {
        let mut sorted = self.data.clone();
        sorted.sort();
        sorted
    }

    /// Calculate percentile (0.0 to 1.0)
    fn percentile(&self, p: f64) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let sorted = self.sorted_samples();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted.get(idx).cloned()
    }
}

impl Metrics {
    /// Create a new Metrics instance
    ///
    /// Use `const fn` for static initialization if possible
    pub const fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            captures_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            extractions_total: AtomicU64::new(0),
            navigations_total: AtomicU64::new(0),
            active_connections: AtomicU32::new(0),
            active_pages: AtomicU32::new(0),
            request_durations: RwLock::new(RingBuffer {
                data: Vec::new(),
                capacity: MAX_HISTOGRAM_SAMPLES,
                write_pos: 0,
                total_samples: 0,
            }),
            requests_by_path_status: RwLock::new(HashMap::new()),
            errors_by_type: RwLock::new(HashMap::new()),
            captures_by_format: RwLock::new(HashMap::new()),
            start_time: RwLock::new(None),
        }
    }

    /// Initialize the metrics collector (call once at startup)
    pub fn init(&self) {
        let mut start = self.start_time.write().unwrap();
        if start.is_none() {
            *start = Some(Instant::now());
        }
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time
            .read()
            .unwrap()
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    // === Counter Operations ===

    /// Record a completed request with path, status code, and duration
    pub fn record_request(&self, path: &str, status: u16, duration: Duration) {
        // Increment total counter
        self.requests_total.fetch_add(1, Ordering::Relaxed);

        // Record duration
        if let Ok(mut durations) = self.request_durations.write() {
            durations.push(duration);
        }

        // Record by path/status
        if let Ok(mut by_path) = self.requests_by_path_status.write() {
            let key = (path.to_string(), status);
            *by_path.entry(key).or_insert(0) += 1;
        }
    }

    /// Increment capture counter
    pub fn increment_captures(&self) {
        self.captures_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment capture counter with format label
    pub fn record_capture(&self, format: &str) {
        self.captures_total.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut by_format) = self.captures_by_format.write() {
            *by_format.entry(format.to_string()).or_insert(0) += 1;
        }
    }

    /// Increment error counter
    pub fn increment_errors(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error with type label
    pub fn record_error(&self, error_type: &str) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut by_type) = self.errors_by_type.write() {
            *by_type.entry(error_type.to_string()).or_insert(0) += 1;
        }
    }

    /// Increment extraction counter
    pub fn increment_extractions(&self) {
        self.extractions_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment navigation counter
    pub fn increment_navigations(&self) {
        self.navigations_total.fetch_add(1, Ordering::Relaxed);
    }

    // === Gauge Operations ===

    /// Increment active connections (call when connection opens)
    pub fn connection_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connections (call when connection closes)
    pub fn connection_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Increment active pages (call when page is created)
    pub fn page_opened(&self) {
        self.active_pages.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active pages (call when page is closed)
    pub fn page_closed(&self) {
        self.active_pages.fetch_sub(1, Ordering::Relaxed);
    }

    // === Histogram Operations ===

    /// Get percentile of request durations
    ///
    /// # Arguments
    /// * `p` - Percentile as a fraction (e.g., 0.95 for p95)
    ///
    /// # Returns
    /// Duration at the given percentile, or None if no samples
    pub fn get_percentile(&self, p: f64) -> Option<Duration> {
        self.request_durations.read().ok()?.percentile(p)
    }

    /// Get p50 (median) request duration
    pub fn p50(&self) -> Option<Duration> {
        self.get_percentile(0.50)
    }

    /// Get p95 request duration
    pub fn p95(&self) -> Option<Duration> {
        self.get_percentile(0.95)
    }

    /// Get p99 request duration
    pub fn p99(&self) -> Option<Duration> {
        self.get_percentile(0.99)
    }

    /// Get total number of duration samples recorded
    pub fn duration_samples_count(&self) -> u64 {
        self.request_durations
            .read()
            .map(|r| r.total_samples())
            .unwrap_or(0)
    }

    // === Prometheus Export ===

    /// Export metrics in Prometheus text format
    ///
    /// This produces output compatible with Prometheus scraping:
    /// ```text
    /// # HELP reasonkit_requests_total Total HTTP requests
    /// # TYPE reasonkit_requests_total counter
    /// reasonkit_requests_total 1234
    /// ```
    pub fn to_prometheus_format(&self) -> String {
        let mut output = String::with_capacity(4096);

        // Header comment
        output.push_str("# ReasonKit Web Metrics\n");
        output.push_str(&format!(
            "# Generated at: {}\n\n",
            chrono::Utc::now().to_rfc3339()
        ));

        // === Counters ===

        // Total requests
        output.push_str("# HELP reasonkit_requests_total Total HTTP requests received\n");
        output.push_str("# TYPE reasonkit_requests_total counter\n");
        output.push_str(&format!(
            "reasonkit_requests_total {}\n\n",
            self.requests_total.load(Ordering::Relaxed)
        ));

        // Requests by path and status
        if let Ok(by_path) = self.requests_by_path_status.read() {
            if !by_path.is_empty() {
                output.push_str("# HELP reasonkit_requests_by_path Requests by path and status\n");
                output.push_str("# TYPE reasonkit_requests_by_path counter\n");
                for ((path, status), count) in by_path.iter() {
                    output.push_str(&format!(
                        "reasonkit_requests_by_path{{path=\"{}\",status=\"{}\"}} {}\n",
                        escape_label_value(path),
                        status,
                        count
                    ));
                }
                output.push('\n');
            }
        }

        // Total captures
        output.push_str("# HELP reasonkit_captures_total Total captures performed\n");
        output.push_str("# TYPE reasonkit_captures_total counter\n");
        output.push_str(&format!(
            "reasonkit_captures_total {}\n\n",
            self.captures_total.load(Ordering::Relaxed)
        ));

        // Captures by format
        if let Ok(by_format) = self.captures_by_format.read() {
            if !by_format.is_empty() {
                output.push_str("# HELP reasonkit_captures_by_format Captures by format type\n");
                output.push_str("# TYPE reasonkit_captures_by_format counter\n");
                for (format, count) in by_format.iter() {
                    output.push_str(&format!(
                        "reasonkit_captures_by_format{{format=\"{}\"}} {}\n",
                        escape_label_value(format),
                        count
                    ));
                }
                output.push('\n');
            }
        }

        // Total errors
        output.push_str("# HELP reasonkit_errors_total Total errors encountered\n");
        output.push_str("# TYPE reasonkit_errors_total counter\n");
        output.push_str(&format!(
            "reasonkit_errors_total {}\n\n",
            self.errors_total.load(Ordering::Relaxed)
        ));

        // Errors by type
        if let Ok(by_type) = self.errors_by_type.read() {
            if !by_type.is_empty() {
                output.push_str("# HELP reasonkit_errors_by_type Errors by error type\n");
                output.push_str("# TYPE reasonkit_errors_by_type counter\n");
                for (error_type, count) in by_type.iter() {
                    output.push_str(&format!(
                        "reasonkit_errors_by_type{{type=\"{}\"}} {}\n",
                        escape_label_value(error_type),
                        count
                    ));
                }
                output.push('\n');
            }
        }

        // Extractions
        output.push_str("# HELP reasonkit_extractions_total Total content extractions\n");
        output.push_str("# TYPE reasonkit_extractions_total counter\n");
        output.push_str(&format!(
            "reasonkit_extractions_total {}\n\n",
            self.extractions_total.load(Ordering::Relaxed)
        ));

        // Navigations
        output.push_str("# HELP reasonkit_navigations_total Total navigation operations\n");
        output.push_str("# TYPE reasonkit_navigations_total counter\n");
        output.push_str(&format!(
            "reasonkit_navigations_total {}\n\n",
            self.navigations_total.load(Ordering::Relaxed)
        ));

        // === Gauges ===

        // Active connections
        output.push_str("# HELP reasonkit_active_connections Current active connections\n");
        output.push_str("# TYPE reasonkit_active_connections gauge\n");
        output.push_str(&format!(
            "reasonkit_active_connections {}\n\n",
            self.active_connections.load(Ordering::Relaxed)
        ));

        // Active pages
        output.push_str("# HELP reasonkit_active_pages Current active browser pages\n");
        output.push_str("# TYPE reasonkit_active_pages gauge\n");
        output.push_str(&format!(
            "reasonkit_active_pages {}\n\n",
            self.active_pages.load(Ordering::Relaxed)
        ));

        // Uptime
        output.push_str("# HELP reasonkit_uptime_seconds Process uptime in seconds\n");
        output.push_str("# TYPE reasonkit_uptime_seconds gauge\n");
        output.push_str(&format!(
            "reasonkit_uptime_seconds {:.3}\n\n",
            self.uptime_seconds()
        ));

        // === Histograms (as summary) ===

        if let Ok(durations) = self.request_durations.read() {
            if durations.len() > 0 {
                output.push_str(
                    "# HELP reasonkit_request_duration_seconds Request duration in seconds\n",
                );
                output.push_str("# TYPE reasonkit_request_duration_seconds summary\n");

                // Quantiles
                if let Some(p50) = durations.percentile(0.50) {
                    output.push_str(&format!(
                        "reasonkit_request_duration_seconds{{quantile=\"0.5\"}} {:.6}\n",
                        p50.as_secs_f64()
                    ));
                }
                if let Some(p90) = durations.percentile(0.90) {
                    output.push_str(&format!(
                        "reasonkit_request_duration_seconds{{quantile=\"0.9\"}} {:.6}\n",
                        p90.as_secs_f64()
                    ));
                }
                if let Some(p95) = durations.percentile(0.95) {
                    output.push_str(&format!(
                        "reasonkit_request_duration_seconds{{quantile=\"0.95\"}} {:.6}\n",
                        p95.as_secs_f64()
                    ));
                }
                if let Some(p99) = durations.percentile(0.99) {
                    output.push_str(&format!(
                        "reasonkit_request_duration_seconds{{quantile=\"0.99\"}} {:.6}\n",
                        p99.as_secs_f64()
                    ));
                }

                // Count and sum
                output.push_str(&format!(
                    "reasonkit_request_duration_seconds_count {}\n",
                    durations.total_samples()
                ));

                // Calculate sum
                let sum: f64 = durations.data.iter().map(|d| d.as_secs_f64()).sum();
                output.push_str(&format!(
                    "reasonkit_request_duration_seconds_sum {:.6}\n",
                    sum
                ));
                output.push('\n');

                // Also emit bucket-style histogram for compatibility
                output.push_str("# HELP reasonkit_request_duration_ms_bucket Request duration histogram buckets\n");
                output.push_str("# TYPE reasonkit_request_duration_ms_bucket histogram\n");

                let sorted = durations.sorted_samples();
                for bucket in DEFAULT_BUCKETS_MS {
                    let bucket_duration = Duration::from_millis(*bucket);
                    let count = sorted.iter().filter(|&&d| d <= bucket_duration).count();
                    output.push_str(&format!(
                        "reasonkit_request_duration_ms_bucket{{le=\"{}\"}} {}\n",
                        bucket, count
                    ));
                }
                output.push_str(&format!(
                    "reasonkit_request_duration_ms_bucket{{le=\"+Inf\"}} {}\n",
                    durations.total_samples()
                ));
                output.push('\n');
            }
        }

        output
    }

    /// Export metrics as JSON for API consumers
    pub fn to_json(&self) -> serde_json::Value {
        let mut counters = serde_json::Map::new();
        counters.insert(
            "requests_total".to_string(),
            serde_json::Value::from(self.requests_total.load(Ordering::Relaxed)),
        );
        counters.insert(
            "captures_total".to_string(),
            serde_json::Value::from(self.captures_total.load(Ordering::Relaxed)),
        );
        counters.insert(
            "errors_total".to_string(),
            serde_json::Value::from(self.errors_total.load(Ordering::Relaxed)),
        );
        counters.insert(
            "extractions_total".to_string(),
            serde_json::Value::from(self.extractions_total.load(Ordering::Relaxed)),
        );
        counters.insert(
            "navigations_total".to_string(),
            serde_json::Value::from(self.navigations_total.load(Ordering::Relaxed)),
        );

        let mut gauges = serde_json::Map::new();
        gauges.insert(
            "active_connections".to_string(),
            serde_json::Value::from(self.active_connections.load(Ordering::Relaxed)),
        );
        gauges.insert(
            "active_pages".to_string(),
            serde_json::Value::from(self.active_pages.load(Ordering::Relaxed)),
        );
        gauges.insert(
            "uptime_seconds".to_string(),
            serde_json::Value::from(self.uptime_seconds()),
        );

        let mut latency = serde_json::Map::new();
        if let Some(p50) = self.p50() {
            latency.insert(
                "p50_ms".to_string(),
                serde_json::Value::from(p50.as_millis() as u64),
            );
        }
        if let Some(p95) = self.p95() {
            latency.insert(
                "p95_ms".to_string(),
                serde_json::Value::from(p95.as_millis() as u64),
            );
        }
        if let Some(p99) = self.p99() {
            latency.insert(
                "p99_ms".to_string(),
                serde_json::Value::from(p99.as_millis() as u64),
            );
        }
        latency.insert(
            "samples".to_string(),
            serde_json::Value::from(self.duration_samples_count()),
        );

        serde_json::json!({
            "counters": counters,
            "gauges": gauges,
            "latency": latency,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }

    /// Reset all metrics (useful for testing)
    pub fn reset(&self) {
        self.requests_total.store(0, Ordering::Relaxed);
        self.captures_total.store(0, Ordering::Relaxed);
        self.errors_total.store(0, Ordering::Relaxed);
        self.extractions_total.store(0, Ordering::Relaxed);
        self.navigations_total.store(0, Ordering::Relaxed);
        self.active_connections.store(0, Ordering::Relaxed);
        self.active_pages.store(0, Ordering::Relaxed);

        if let Ok(mut durations) = self.request_durations.write() {
            *durations = RingBuffer::new(MAX_HISTOGRAM_SAMPLES);
        }
        if let Ok(mut by_path) = self.requests_by_path_status.write() {
            by_path.clear();
        }
        if let Ok(mut by_type) = self.errors_by_type.write() {
            by_type.clear();
        }
        if let Ok(mut by_format) = self.captures_by_format.write() {
            by_format.clear();
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape a label value for Prometheus format
/// Prometheus requires escaping backslash, newline, and double quote
fn escape_label_value(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

/// Global metrics instance for the application
///
/// Use this for recording metrics throughout the codebase:
/// ```rust,ignore
/// use reasonkit_web::metrics::METRICS;
/// METRICS.record_request("/capture", 200, duration);
/// ```
pub static METRICS: Metrics = Metrics::new();

/// Initialize global metrics (call once at startup)
pub fn init() {
    METRICS.init();
}

// === Metrics Server (Optional HTTP endpoint) ===

/// Configuration for the metrics HTTP server
#[derive(Debug, Clone)]
pub struct MetricsServerConfig {
    /// Address to bind to (e.g., "127.0.0.1:9090")
    pub bind_address: String,
    /// Optional authentication token for /metrics endpoint
    pub auth_token: Option<String>,
    /// Path for the metrics endpoint (default: "/metrics")
    pub metrics_path: String,
}

impl Default for MetricsServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:9090".to_string(),
            auth_token: None,
            metrics_path: "/metrics".to_string(),
        }
    }
}

/// Simple HTTP server for exposing metrics
///
/// This is a lightweight implementation using tokio TcpListener.
/// For production, consider using a proper HTTP framework.
pub struct MetricsServer {
    config: MetricsServerConfig,
    metrics: &'static Metrics,
}

impl MetricsServer {
    /// Create a new metrics server with the given configuration
    pub fn new(config: MetricsServerConfig) -> Self {
        Self {
            config,
            metrics: &METRICS,
        }
    }

    /// Create a metrics server with a custom metrics instance (for testing)
    pub fn with_metrics(config: MetricsServerConfig, metrics: &'static Metrics) -> Self {
        Self { config, metrics }
    }

    /// Run the metrics server (blocking)
    ///
    /// This will listen for HTTP requests and serve metrics at the configured path.
    pub async fn run(&self) -> std::io::Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind(&self.config.bind_address).await?;
        tracing::info!("Metrics server listening on {}", self.config.bind_address);

        loop {
            let (mut socket, addr) = listener.accept().await?;
            let config = self.config.clone();
            let metrics = self.metrics;

            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let n = match socket.read(&mut buf).await {
                    Ok(n) if n == 0 => return,
                    Ok(n) => n,
                    Err(e) => {
                        tracing::debug!("Failed to read from socket: {}", e);
                        return;
                    }
                };

                let request = String::from_utf8_lossy(&buf[..n]);
                let response = Self::handle_request(&request, &config, metrics);

                if let Err(e) = socket.write_all(response.as_bytes()).await {
                    tracing::debug!("Failed to write response to {}: {}", addr, e);
                }
            });
        }
    }

    /// Handle an HTTP request
    fn handle_request(request: &str, config: &MetricsServerConfig, metrics: &Metrics) -> String {
        let lines: Vec<&str> = request.lines().collect();
        if lines.is_empty() {
            return Self::http_response(400, "text/plain", "Bad Request");
        }

        // Parse request line
        let parts: Vec<&str> = lines[0].split_whitespace().collect();
        if parts.len() < 2 {
            return Self::http_response(400, "text/plain", "Bad Request");
        }

        let method = parts[0];
        let path = parts[1];

        // Only allow GET
        if method != "GET" {
            return Self::http_response(405, "text/plain", "Method Not Allowed");
        }

        // Check path
        if path == config.metrics_path || path == &format!("{}/", config.metrics_path) {
            // Check authentication if configured
            if let Some(ref token) = config.auth_token {
                let auth_header = lines
                    .iter()
                    .find(|l| l.to_lowercase().starts_with("authorization:"));
                let authorized = auth_header.map_or(false, |h| {
                    h.trim()
                        .strip_prefix("Authorization:")
                        .or_else(|| h.trim().strip_prefix("authorization:"))
                        .map_or(false, |v| v.trim() == format!("Bearer {}", token))
                });

                if !authorized {
                    return Self::http_response(401, "text/plain", "Unauthorized");
                }
            }

            // Return metrics
            let body = metrics.to_prometheus_format();
            Self::http_response(200, "text/plain; version=0.0.4; charset=utf-8", &body)
        } else if path == "/health" || path == "/healthz" {
            Self::http_response(200, "application/json", "{\"status\":\"ok\"}")
        } else if path == "/ready" || path == "/readyz" {
            Self::http_response(200, "application/json", "{\"status\":\"ready\"}")
        } else if path == "/metrics.json" {
            // Check auth for JSON endpoint too
            if let Some(ref token) = config.auth_token {
                let auth_header = lines
                    .iter()
                    .find(|l| l.to_lowercase().starts_with("authorization:"));
                let authorized = auth_header.map_or(false, |h| {
                    h.trim()
                        .strip_prefix("Authorization:")
                        .or_else(|| h.trim().strip_prefix("authorization:"))
                        .map_or(false, |v| v.trim() == format!("Bearer {}", token))
                });

                if !authorized {
                    return Self::http_response(401, "text/plain", "Unauthorized");
                }
            }

            let body = metrics.to_json().to_string();
            Self::http_response(200, "application/json", &body)
        } else {
            Self::http_response(404, "text/plain", "Not Found")
        }
    }

    /// Build an HTTP response
    fn http_response(status: u16, content_type: &str, body: &str) -> String {
        let status_text = match status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            405 => "Method Not Allowed",
            _ => "Unknown",
        };

        format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: {}\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            status,
            status_text,
            content_type,
            body.len(),
            body
        )
    }
}

// === Request Timer Guard ===

/// RAII guard for timing requests automatically
///
/// Records the request duration when dropped.
///
/// # Example
/// ```rust,ignore
/// let _timer = RequestTimer::new(&METRICS, "/capture");
/// // ... do work ...
/// // timer automatically records duration when it goes out of scope
/// ```
pub struct RequestTimer<'a> {
    metrics: &'a Metrics,
    path: String,
    start: Instant,
    status: u16,
}

impl<'a> RequestTimer<'a> {
    /// Start a new request timer
    pub fn new(metrics: &'a Metrics, path: &str) -> Self {
        Self {
            metrics,
            path: path.to_string(),
            start: Instant::now(),
            status: 200,
        }
    }

    /// Set the response status code
    pub fn set_status(&mut self, status: u16) {
        self.status = status;
    }

    /// Get elapsed time so far
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl<'a> Drop for RequestTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.metrics
            .record_request(&self.path, self.status, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<u32> = RingBuffer::new(5);
        for i in 0..3 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_samples(), 3);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buf: RingBuffer<u32> = RingBuffer::new(3);
        for i in 0..10 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_samples(), 10);
        // Should contain 7, 8, 9 (last 3 values)
        let sorted = buf.sorted_samples();
        assert!(sorted.contains(&7));
        assert!(sorted.contains(&8));
        assert!(sorted.contains(&9));
    }

    #[test]
    fn test_ring_buffer_percentile() {
        let mut buf: RingBuffer<u32> = RingBuffer::new(100);
        for i in 1..=100 {
            buf.push(i);
        }
        // p50 should be around 50
        assert_eq!(buf.percentile(0.50), Some(50));
        // p99 should be around 99
        assert_eq!(buf.percentile(0.99), Some(99));
    }

    #[test]
    fn test_metrics_counters() {
        let metrics = Metrics::new();

        metrics.record_request("/test", 200, Duration::from_millis(100));
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);

        metrics.increment_captures();
        assert_eq!(metrics.captures_total.load(Ordering::Relaxed), 1);

        metrics.increment_errors();
        assert_eq!(metrics.errors_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_gauges() {
        let metrics = Metrics::new();

        metrics.connection_opened();
        metrics.connection_opened();
        assert_eq!(metrics.active_connections.load(Ordering::Relaxed), 2);

        metrics.connection_closed();
        assert_eq!(metrics.active_connections.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_percentiles() {
        let metrics = Metrics::new();

        // Add 100 samples from 1ms to 100ms
        for i in 1..=100 {
            metrics.record_request("/test", 200, Duration::from_millis(i));
        }

        let p50 = metrics.p50().unwrap();
        let p95 = metrics.p95().unwrap();
        let p99 = metrics.p99().unwrap();

        // p50 should be around 50ms
        assert!(p50.as_millis() >= 45 && p50.as_millis() <= 55);
        // p95 should be around 95ms
        assert!(p95.as_millis() >= 90 && p95.as_millis() <= 100);
        // p99 should be around 99ms
        assert!(p99.as_millis() >= 95 && p99.as_millis() <= 100);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = Metrics::new();
        metrics.init();

        metrics.record_request("/capture", 200, Duration::from_millis(50));
        metrics.record_request("/capture", 200, Duration::from_millis(100));
        metrics.record_request("/extract", 500, Duration::from_millis(200));
        metrics.record_capture("screenshot");
        metrics.record_error("navigation");

        let output = metrics.to_prometheus_format();

        // Check counters are present
        assert!(output.contains("reasonkit_requests_total 3"));
        assert!(output.contains("reasonkit_captures_total 1"));
        assert!(output.contains("reasonkit_errors_total 1"));

        // Check labeled metrics
        assert!(output.contains("path=\"/capture\""));
        assert!(output.contains("status=\"200\""));
        assert!(output.contains("format=\"screenshot\""));
        assert!(output.contains("type=\"navigation\""));

        // Check histogram
        assert!(output.contains("reasonkit_request_duration_seconds"));
    }

    #[test]
    fn test_json_format() {
        let metrics = Metrics::new();
        metrics.init();

        metrics.record_request("/test", 200, Duration::from_millis(50));
        metrics.increment_captures();

        let json = metrics.to_json();

        assert_eq!(json["counters"]["requests_total"], 1);
        assert_eq!(json["counters"]["captures_total"], 1);
        assert!(json["timestamp"].is_string());
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new();

        metrics.record_request("/test", 200, Duration::from_millis(100));
        metrics.increment_captures();
        metrics.connection_opened();

        metrics.reset();

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.captures_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.active_connections.load(Ordering::Relaxed), 0);
        assert!(metrics.p50().is_none());
    }

    #[test]
    fn test_escape_label_value() {
        assert_eq!(escape_label_value("simple"), "simple");
        assert_eq!(escape_label_value("with\\backslash"), "with\\\\backslash");
        assert_eq!(escape_label_value("with\nnewline"), "with\\nnewline");
        assert_eq!(escape_label_value("with\"quote"), "with\\\"quote");
    }

    #[test]
    fn test_request_timer() {
        let metrics = Metrics::new();

        {
            let mut timer = RequestTimer::new(&metrics, "/timed");
            std::thread::sleep(Duration::from_millis(10));
            timer.set_status(201);
            // timer drops here and records
        }

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
        let p50 = metrics.p50().unwrap();
        assert!(p50.as_millis() >= 10);
    }

    #[test]
    fn test_http_response_format() {
        let response = MetricsServer::http_response(200, "text/plain", "test body");
        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains("Content-Type: text/plain"));
        assert!(response.contains("Content-Length: 9"));
        assert!(response.ends_with("test body"));
    }
}

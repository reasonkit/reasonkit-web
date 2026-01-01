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
//! use reasonkit_web::metrics::global_metrics;
//! use std::time::Duration;
//!
//! // Record a request
//! global_metrics().record_request("/capture", 200, Duration::from_millis(150));
//!
//! // Get Prometheus output
//! let output = global_metrics().to_prometheus_format();
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Maximum number of duration samples to keep in the histogram
/// This provides a good balance between memory usage and accuracy
const MAX_HISTOGRAM_SAMPLES: usize = 1000;

/// Default buckets for latency histograms (in milliseconds)
#[allow(dead_code)]
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
pub struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    /// Position of next write (wraps around)
    write_pos: usize,
    /// Total samples received (may exceed capacity)
    total_samples: u64,
}

#[allow(dead_code)]
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

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create a new Metrics instance
    ///
    /// Cannot use `const fn` due to RwLock containing non-const operations
    pub fn new() -> Self {
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

    /// Record a request with timing information
    pub fn record_request(&self, path: &str, status_code: u16, duration: Duration) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);

        // Update request histogram
        if let Ok(mut durations) = self.request_durations.write() {
            durations.push(duration);
        }

        // Update path/status breakdown
        if let Ok(mut breakdown) = self.requests_by_path_status.write() {
            *breakdown
                .entry((path.to_string(), status_code))
                .or_insert(0) += 1;
        }
    }

    /// Record a browser capture operation
    pub fn record_capture(&self, format: &str) {
        self.captures_total.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut breakdown) = self.captures_by_format.write() {
            *breakdown.entry(format.to_string()).or_insert(0) += 1;
        }
    }

    /// Record an error
    pub fn record_error(&self, error_type: &str) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut breakdown) = self.errors_by_type.write() {
            *breakdown.entry(error_type.to_string()).or_insert(0) += 1;
        }
    }

    /// Record a successful text extraction
    pub fn record_extraction(&self) {
        self.extractions_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a navigation
    pub fn record_navigation(&self) {
        self.navigations_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment active connections
    pub fn inc_active_connections(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connections
    pub fn dec_active_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Increment active browser pages
    pub fn inc_active_pages(&self) {
        self.active_pages.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active browser pages
    pub fn dec_active_pages(&self) {
        self.active_pages.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current request durations histogram
    pub fn get_request_durations(&self) -> Option<RingBuffer<Duration>> {
        self.request_durations
            .read()
            .ok()
            .map(|durations| RingBuffer {
                data: durations.data.clone(),
                capacity: durations.capacity,
                write_pos: durations.write_pos,
                total_samples: durations.total_samples,
            })
    }

    /// Convert metrics to Prometheus text format
    pub fn to_prometheus_format(&self) -> String {
        let mut output = String::new();

        // Counters
        output.push_str(&format!(
            "reasonkit_web_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "reasonkit_web_captures_total {}\n",
            self.captures_total.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "reasonkit_web_errors_total {}\n",
            self.errors_total.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "reasonkit_web_extractions_total {}\n",
            self.extractions_total.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "reasonkit_web_navigations_total {}\n",
            self.navigations_total.load(Ordering::Relaxed)
        ));

        // Gauges
        output.push_str(&format!(
            "reasonkit_web_active_connections {}\n",
            self.active_connections.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "reasonkit_web_active_pages {}\n",
            self.active_pages.load(Ordering::Relaxed)
        ));

        // Histogram metrics (simple percentile calculation)
        if let Ok(durations) = self.request_durations.read() {
            if durations.len() > 0 {
                if let Some(p50) = durations.percentile(0.5) {
                    output.push_str(&format!(
                        "reasonkit_web_request_duration_p50_ms {}\n",
                        p50.as_millis()
                    ));
                }
                if let Some(p95) = durations.percentile(0.95) {
                    output.push_str(&format!(
                        "reasonkit_web_request_duration_p95_ms {}\n",
                        p95.as_millis()
                    ));
                }
                if let Some(p99) = durations.percentile(0.99) {
                    output.push_str(&format!(
                        "reasonkit_web_request_duration_p99_ms {}\n",
                        p99.as_millis()
                    ));
                }
            }
        }

        output
    }
}

/// Global metrics instance for the web server
///
/// Use this for recording metrics throughout the codebase:
/// ```rust,ignore
/// use reasonkit_web::metrics::global_metrics;
/// global_metrics().record_request("/capture", 200, duration);
/// ```
pub static METRICS: OnceLock<Metrics> = OnceLock::new();

/// Get or initialize the global metrics instance
pub fn global_metrics() -> &'static Metrics {
    METRICS.get_or_init(Metrics::new)
}

/// Initialize global metrics (call once at startup)
pub fn init() {
    let _ = METRICS.get_or_init(Metrics::new);

    // Initialize start time
    if let Ok(mut start_time) = global_metrics().start_time.write() {
        *start_time = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = Metrics::new();

        metrics.record_request("/test", 200, Duration::from_millis(100));
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);

        metrics.record_capture("screenshot");
        assert_eq!(metrics.captures_total.load(Ordering::Relaxed), 1);

        metrics.record_error("timeout");
        assert_eq!(metrics.errors_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_global_metrics() {
        init();

        let metrics = global_metrics();
        metrics.record_request("/test", 200, Duration::from_millis(150));

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
    }
}

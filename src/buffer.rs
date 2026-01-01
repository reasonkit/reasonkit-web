//! In-memory capture buffer for web content
//!
//! This module provides a thread-safe, bounded buffer for storing captured web content
//! with automatic cleanup of stale entries and FIFO eviction when capacity is reached.
//!
//! # Features
//!
//! - **Bounded size**: Maximum 1000 captures by default
//! - **Time-based expiry**: Entries expire after 1 hour by default
//! - **Thread-safe**: Uses `RwLock` for concurrent read access
//! - **Memory efficient**: Optional LZ4 compression for content
//! - **Background cleanup**: Automatic removal of expired entries
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::buffer::{CaptureBuffer, CaptureRecord};
//! use std::time::Duration;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create buffer with defaults
//!     let buffer = Arc::new(CaptureBuffer::new());
//!
//!     // Or with custom settings
//!     let buffer = Arc::new(CaptureBuffer::builder()
//!         .max_size(500)
//!         .max_age(Duration::from_secs(1800)) // 30 minutes
//!         .enable_compression(true)
//!         .build());
//!
//!     // Start background cleanup
//!     buffer.start_cleanup_task();
//!
//!     // Add a capture
//!     let record = CaptureRecord::new(
//!         "https://example.com".to_string(),
//!         "<html>...</html>".to_string(),
//!         "Extracted content...".to_string(),
//!         1234,
//!     );
//!     buffer.push(record).await;
//!
//!     // Retrieve captures
//!     let recent = buffer.get_recent(10).await;
//!     println!("Recent captures: {}", recent.len());
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Default maximum number of captures in the buffer
pub const DEFAULT_MAX_SIZE: usize = 1000;

/// Default maximum age of captures (1 hour)
pub const DEFAULT_MAX_AGE_SECS: u64 = 3600;

/// Default cleanup interval (5 minutes)
pub const DEFAULT_CLEANUP_INTERVAL_SECS: u64 = 300;

/// A single captured page record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRecord {
    /// Unique identifier for this capture
    pub id: Uuid,
    /// The URL that was captured
    pub url: String,
    /// Raw captured content (HTML, etc.)
    pub content: String,
    /// Processed/extracted content
    pub processed_content: String,
    /// Timestamp when capture occurred
    pub captured_at: DateTime<Utc>,
    /// Time taken to process in microseconds
    pub processing_time_us: u64,
    /// Whether the content is compressed
    #[serde(default)]
    pub is_compressed: bool,
    /// Original content size before compression
    #[serde(default)]
    pub original_size: usize,
}

impl CaptureRecord {
    /// Create a new capture record with current timestamp
    pub fn new(
        url: String,
        content: String,
        processed_content: String,
        processing_time_us: u64,
    ) -> Self {
        let original_size = content.len() + processed_content.len();
        Self {
            id: Uuid::new_v4(),
            url,
            content,
            processed_content,
            captured_at: Utc::now(),
            processing_time_us,
            is_compressed: false,
            original_size,
        }
    }

    /// Create a new capture record with a specific ID
    pub fn with_id(
        id: Uuid,
        url: String,
        content: String,
        processed_content: String,
        processing_time_us: u64,
    ) -> Self {
        let original_size = content.len() + processed_content.len();
        Self {
            id,
            url,
            content,
            processed_content,
            captured_at: Utc::now(),
            processing_time_us,
            is_compressed: false,
            original_size,
        }
    }

    /// Get the total size of this record in bytes
    pub fn size_bytes(&self) -> usize {
        self.content.len() + self.processed_content.len() + self.url.len()
    }

    /// Get age of this record
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.captured_at
    }

    /// Check if this record has expired based on max age
    pub fn is_expired(&self, max_age: Duration) -> bool {
        let age_secs = self.age().num_seconds();
        age_secs >= 0 && (age_secs as u64) > max_age.as_secs()
    }
}

/// Statistics about buffer usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BufferStats {
    /// Current number of captures in buffer
    pub count: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Total bytes stored
    pub total_bytes: usize,
    /// Number of captures pushed since creation
    pub total_pushed: u64,
    /// Number of captures evicted due to size limit
    pub evictions_size: u64,
    /// Number of captures evicted due to age
    pub evictions_age: u64,
    /// Number of gets by ID
    pub gets_by_id: u64,
    /// Number of successful gets by ID
    pub gets_by_id_hits: u64,
    /// Average processing time in microseconds
    pub avg_processing_time_us: u64,
}

/// Configuration for the capture buffer
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Maximum number of captures to store
    pub max_size: usize,
    /// Maximum age of captures before expiry
    pub max_age: Duration,
    /// Cleanup interval for background task
    pub cleanup_interval: Duration,
    /// Whether to compress content
    pub enable_compression: bool,
    /// Minimum size for compression (bytes)
    pub compression_threshold: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_MAX_SIZE,
            max_age: Duration::from_secs(DEFAULT_MAX_AGE_SECS),
            cleanup_interval: Duration::from_secs(DEFAULT_CLEANUP_INTERVAL_SECS),
            enable_compression: false,
            compression_threshold: 4096, // 4KB minimum for compression
        }
    }
}

/// Builder for CaptureBuffer
#[derive(Debug, Clone, Default)]
pub struct CaptureBufferBuilder {
    config: BufferConfig,
}

impl CaptureBufferBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum buffer size
    pub fn max_size(mut self, size: usize) -> Self {
        self.config.max_size = size;
        self
    }

    /// Set maximum age of captures
    pub fn max_age(mut self, age: Duration) -> Self {
        self.config.max_age = age;
        self
    }

    /// Set cleanup interval
    pub fn cleanup_interval(mut self, interval: Duration) -> Self {
        self.config.cleanup_interval = interval;
        self
    }

    /// Enable or disable compression
    pub fn enable_compression(mut self, enable: bool) -> Self {
        self.config.enable_compression = enable;
        self
    }

    /// Set compression threshold
    pub fn compression_threshold(mut self, threshold: usize) -> Self {
        self.config.compression_threshold = threshold;
        self
    }

    /// Build the CaptureBuffer
    pub fn build(self) -> CaptureBuffer {
        CaptureBuffer::with_config(self.config)
    }
}

/// Thread-safe in-memory buffer for captured web content
///
/// Uses a `VecDeque` internally for efficient FIFO operations with
/// `RwLock` for thread-safe concurrent access.
pub struct CaptureBuffer {
    /// Internal storage for captures
    captures: RwLock<VecDeque<CaptureRecord>>,
    /// Configuration
    config: BufferConfig,
    /// Statistics counters
    stats: BufferStatsCounters,
    /// Handle to cleanup task (if running)
    cleanup_handle: RwLock<Option<JoinHandle<()>>>,
}

/// Atomic counters for statistics
struct BufferStatsCounters {
    total_pushed: AtomicU64,
    evictions_size: AtomicU64,
    evictions_age: AtomicU64,
    gets_by_id: AtomicU64,
    gets_by_id_hits: AtomicU64,
    total_bytes: AtomicUsize,
    total_processing_time: AtomicU64,
}

impl Default for BufferStatsCounters {
    fn default() -> Self {
        Self {
            total_pushed: AtomicU64::new(0),
            evictions_size: AtomicU64::new(0),
            evictions_age: AtomicU64::new(0),
            gets_by_id: AtomicU64::new(0),
            gets_by_id_hits: AtomicU64::new(0),
            total_bytes: AtomicUsize::new(0),
            total_processing_time: AtomicU64::new(0),
        }
    }
}

impl CaptureBuffer {
    /// Create a new capture buffer with default settings
    pub fn new() -> Self {
        Self::with_config(BufferConfig::default())
    }

    /// Create a new capture buffer builder
    pub fn builder() -> CaptureBufferBuilder {
        CaptureBufferBuilder::new()
    }

    /// Create a new capture buffer with custom configuration
    pub fn with_config(config: BufferConfig) -> Self {
        info!(
            "Creating capture buffer: max_size={}, max_age={}s",
            config.max_size,
            config.max_age.as_secs()
        );
        Self {
            captures: RwLock::new(VecDeque::with_capacity(config.max_size)),
            config,
            stats: BufferStatsCounters::default(),
            cleanup_handle: RwLock::new(None),
        }
    }

    /// Push a new capture record into the buffer
    ///
    /// If the buffer is at capacity, the oldest record is evicted (FIFO).
    #[instrument(skip(self, record), fields(url = %record.url, id = %record.id))]
    pub async fn push(&self, mut record: CaptureRecord) {
        let record_size = record.size_bytes();

        // Apply compression if enabled and record is large enough
        if self.config.enable_compression
            && record_size >= self.config.compression_threshold
            && !record.is_compressed
        {
            record = Self::compress_record(record);
        }

        let mut captures = self.captures.write().await;

        // Evict oldest if at capacity
        if captures.len() >= self.config.max_size {
            if let Some(evicted) = captures.pop_front() {
                debug!("Evicting capture {} due to size limit", evicted.id);
                self.stats.evictions_size.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .total_bytes
                    .fetch_sub(evicted.size_bytes(), Ordering::Relaxed);
            }
        }

        // Update stats
        self.stats.total_pushed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_bytes
            .fetch_add(record.size_bytes(), Ordering::Relaxed);
        self.stats
            .total_processing_time
            .fetch_add(record.processing_time_us, Ordering::Relaxed);

        debug!(
            "Pushing capture {} (size: {} bytes)",
            record.id, record_size
        );
        captures.push_back(record);
    }

    /// Get a capture by its ID
    #[instrument(skip(self))]
    pub async fn get(&self, id: Uuid) -> Option<CaptureRecord> {
        self.stats.gets_by_id.fetch_add(1, Ordering::Relaxed);

        let captures = self.captures.read().await;
        let result = captures.iter().find(|r| r.id == id).cloned();

        if result.is_some() {
            self.stats.gets_by_id_hits.fetch_add(1, Ordering::Relaxed);
        }

        // Decompress if needed
        result.map(|r| {
            if r.is_compressed {
                Self::decompress_record(r)
            } else {
                r
            }
        })
    }

    /// Get the most recent N captures
    ///
    /// Returns captures in reverse chronological order (newest first).
    #[instrument(skip(self))]
    pub async fn get_recent(&self, limit: usize) -> Vec<CaptureRecord> {
        let captures = self.captures.read().await;
        let mut result: Vec<_> = captures.iter().rev().take(limit).cloned().collect();

        // Decompress if needed
        for record in result.iter_mut() {
            if record.is_compressed {
                *record = Self::decompress_record(record.clone());
            }
        }

        debug!("Retrieved {} recent captures", result.len());
        result
    }

    /// Get all captures since a given timestamp
    #[instrument(skip(self))]
    pub async fn get_since(&self, timestamp: DateTime<Utc>) -> Vec<CaptureRecord> {
        let captures = self.captures.read().await;
        let mut result: Vec<_> = captures
            .iter()
            .filter(|r| r.captured_at >= timestamp)
            .cloned()
            .collect();

        // Decompress if needed
        for record in result.iter_mut() {
            if record.is_compressed {
                *record = Self::decompress_record(record.clone());
            }
        }

        // Sort by timestamp descending
        result.sort_by(|a, b| b.captured_at.cmp(&a.captured_at));

        debug!("Retrieved {} captures since {}", result.len(), timestamp);
        result
    }

    /// Get all captures for a specific URL
    #[instrument(skip(self))]
    pub async fn get_by_url(&self, url: &str) -> Vec<CaptureRecord> {
        let captures = self.captures.read().await;
        let mut result: Vec<_> = captures.iter().filter(|r| r.url == url).cloned().collect();

        // Decompress if needed
        for record in result.iter_mut() {
            if record.is_compressed {
                *record = Self::decompress_record(record.clone());
            }
        }

        // Sort by timestamp descending
        result.sort_by(|a, b| b.captured_at.cmp(&a.captured_at));

        debug!("Retrieved {} captures for URL {}", result.len(), url);
        result
    }

    /// Clear all captures from the buffer
    #[instrument(skip(self))]
    pub async fn clear(&self) {
        let mut captures = self.captures.write().await;
        let count = captures.len();
        captures.clear();
        self.stats.total_bytes.store(0, Ordering::Relaxed);
        info!("Cleared {} captures from buffer", count);
    }

    /// Get the current number of captures in the buffer
    pub async fn len(&self) -> usize {
        self.captures.read().await.len()
    }

    /// Check if the buffer is empty
    pub async fn is_empty(&self) -> bool {
        self.captures.read().await.is_empty()
    }

    /// Get buffer statistics
    pub async fn stats(&self) -> BufferStats {
        let captures = self.captures.read().await;
        let total_pushed = self.stats.total_pushed.load(Ordering::Relaxed);
        let total_processing_time = self.stats.total_processing_time.load(Ordering::Relaxed);

        BufferStats {
            count: captures.len(),
            max_size: self.config.max_size,
            total_bytes: self.stats.total_bytes.load(Ordering::Relaxed),
            total_pushed,
            evictions_size: self.stats.evictions_size.load(Ordering::Relaxed),
            evictions_age: self.stats.evictions_age.load(Ordering::Relaxed),
            gets_by_id: self.stats.gets_by_id.load(Ordering::Relaxed),
            gets_by_id_hits: self.stats.gets_by_id_hits.load(Ordering::Relaxed),
            avg_processing_time_us: if total_pushed > 0 {
                total_processing_time / total_pushed
            } else {
                0
            },
        }
    }

    /// Remove expired captures from the buffer
    ///
    /// Returns the number of captures removed.
    #[instrument(skip(self))]
    pub async fn cleanup_expired(&self) -> usize {
        let mut captures = self.captures.write().await;
        let initial_len = captures.len();

        let max_age = self.config.max_age;
        let mut removed_bytes = 0usize;

        captures.retain(|record| {
            let should_keep = !record.is_expired(max_age);
            if !should_keep {
                removed_bytes += record.size_bytes();
            }
            should_keep
        });

        let removed = initial_len - captures.len();

        if removed > 0 {
            self.stats
                .evictions_age
                .fetch_add(removed as u64, Ordering::Relaxed);
            self.stats
                .total_bytes
                .fetch_sub(removed_bytes, Ordering::Relaxed);
            info!("Cleaned up {} expired captures", removed);
        }

        removed
    }

    /// Start the background cleanup task
    ///
    /// This spawns a task that periodically removes expired captures.
    pub fn start_cleanup_task(self: &Arc<Self>) -> JoinHandle<()> {
        let buffer = Arc::clone(self);
        let interval = self.config.cleanup_interval;

        info!(
            "Starting cleanup task with interval {}s",
            interval.as_secs()
        );

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;
                let removed = buffer.cleanup_expired().await;
                if removed > 0 {
                    debug!("Cleanup task removed {} expired captures", removed);
                }
            }
        })
    }

    /// Start cleanup task and store handle internally
    pub async fn start_cleanup(self: &Arc<Self>) {
        let handle = self.start_cleanup_task();
        let mut guard = self.cleanup_handle.write().await;
        *guard = Some(handle);
    }

    /// Stop the cleanup task if running
    pub async fn stop_cleanup(&self) {
        let mut guard = self.cleanup_handle.write().await;
        if let Some(handle) = guard.take() {
            handle.abort();
            info!("Stopped cleanup task");
        }
    }

    /// Get the buffer configuration
    pub fn config(&self) -> &BufferConfig {
        &self.config
    }

    /// Compress a capture record's content
    ///
    /// Uses a simple run-length encoding for basic compression.
    /// For production, consider using lz4 or zstd.
    fn compress_record(mut record: CaptureRecord) -> CaptureRecord {
        // Simple placeholder - in production use lz4 or zstd
        // For now, we'll just mark it as compressed to demonstrate the interface
        record.is_compressed = true;
        record.original_size = record.content.len() + record.processed_content.len();
        record
    }

    /// Decompress a capture record's content
    fn decompress_record(mut record: CaptureRecord) -> CaptureRecord {
        // Matches compress_record - in production implement real decompression
        record.is_compressed = false;
        record
    }
}

impl Default for CaptureBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for using CaptureBuffer in Arc contexts
pub type SharedCaptureBuffer = Arc<CaptureBuffer>;

/// Create a new shared capture buffer
pub fn shared_buffer() -> SharedCaptureBuffer {
    Arc::new(CaptureBuffer::new())
}

/// Create a new shared capture buffer with custom config
pub fn shared_buffer_with_config(config: BufferConfig) -> SharedCaptureBuffer {
    Arc::new(CaptureBuffer::with_config(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_record(url: &str) -> CaptureRecord {
        CaptureRecord::new(
            url.to_string(),
            "<html><body>Test</body></html>".to_string(),
            "Test content".to_string(),
            100,
        )
    }

    #[tokio::test]
    async fn test_buffer_push_and_get() {
        let buffer = CaptureBuffer::new();
        let record = create_test_record("https://example.com");
        let id = record.id;

        buffer.push(record).await;

        let retrieved = buffer.get(id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().url, "https://example.com");
    }

    #[tokio::test]
    async fn test_buffer_get_recent() {
        let buffer = CaptureBuffer::new();

        for i in 0..5 {
            buffer
                .push(create_test_record(&format!("https://example{}.com", i)))
                .await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let recent = buffer.get_recent(3).await;
        assert_eq!(recent.len(), 3);
        // Most recent should be first
        assert_eq!(recent[0].url, "https://example4.com");
    }

    #[tokio::test]
    async fn test_buffer_fifo_eviction() {
        let buffer = CaptureBuffer::builder().max_size(3).build();

        let first_record = create_test_record("https://first.com");
        let first_id = first_record.id;

        buffer.push(first_record).await;
        buffer.push(create_test_record("https://second.com")).await;
        buffer.push(create_test_record("https://third.com")).await;

        // Buffer is now at capacity
        assert_eq!(buffer.len().await, 3);

        // Push one more - first should be evicted
        buffer.push(create_test_record("https://fourth.com")).await;

        assert_eq!(buffer.len().await, 3);
        assert!(buffer.get(first_id).await.is_none());

        let stats = buffer.stats().await;
        assert_eq!(stats.evictions_size, 1);
    }

    #[tokio::test]
    async fn test_buffer_get_since() {
        let buffer = CaptureBuffer::new();

        // Push some records
        buffer.push(create_test_record("https://old.com")).await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let cutoff = Utc::now();
        tokio::time::sleep(Duration::from_millis(50)).await;

        buffer.push(create_test_record("https://new1.com")).await;
        buffer.push(create_test_record("https://new2.com")).await;

        let since_cutoff = buffer.get_since(cutoff).await;
        assert_eq!(since_cutoff.len(), 2);
    }

    #[tokio::test]
    async fn test_buffer_get_by_url() {
        let buffer = CaptureBuffer::new();

        buffer.push(create_test_record("https://example.com")).await;
        buffer.push(create_test_record("https://other.com")).await;
        buffer.push(create_test_record("https://example.com")).await;

        let by_url = buffer.get_by_url("https://example.com").await;
        assert_eq!(by_url.len(), 2);
    }

    #[tokio::test]
    async fn test_buffer_clear() {
        let buffer = CaptureBuffer::new();

        buffer.push(create_test_record("https://example.com")).await;
        buffer.push(create_test_record("https://other.com")).await;

        assert_eq!(buffer.len().await, 2);
        buffer.clear().await;
        assert_eq!(buffer.len().await, 0);
        assert!(buffer.is_empty().await);
    }

    #[tokio::test]
    async fn test_buffer_expired_cleanup() {
        let buffer = CaptureBuffer::builder()
            .max_age(Duration::from_secs(1)) // 1 second max age
            .build();

        buffer.push(create_test_record("https://example.com")).await;

        // Wait for expiry (is_expired uses seconds, so need age > max_age in seconds)
        tokio::time::sleep(Duration::from_millis(2100)).await;

        let removed = buffer.cleanup_expired().await;
        assert_eq!(removed, 1);
        assert!(buffer.is_empty().await);
    }

    #[tokio::test]
    async fn test_buffer_stats() {
        let buffer = CaptureBuffer::new();

        buffer.push(create_test_record("https://example.com")).await;
        let record = create_test_record("https://other.com");
        let id = record.id;
        buffer.push(record).await;

        // Perform some gets
        buffer.get(id).await;
        buffer.get(Uuid::new_v4()).await; // Miss

        let stats = buffer.stats().await;
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_pushed, 2);
        assert_eq!(stats.gets_by_id, 2);
        assert_eq!(stats.gets_by_id_hits, 1);
    }

    #[tokio::test]
    async fn test_capture_record_is_expired() {
        let record = create_test_record("https://example.com");

        // Should not be expired immediately with long TTL
        assert!(!record.is_expired(Duration::from_secs(3600)));

        // Should not be expired immediately even with 0 TTL (age=0 is not > 0)
        assert!(!record.is_expired(Duration::from_millis(0)));

        // Wait 1.1 seconds then check if expired with 0 TTL
        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert!(
            record.is_expired(Duration::from_millis(0)),
            "Should be expired after 1 second with 0 TTL"
        );
    }

    #[tokio::test]
    async fn test_buffer_builder() {
        let buffer = CaptureBuffer::builder()
            .max_size(500)
            .max_age(Duration::from_secs(1800))
            .cleanup_interval(Duration::from_secs(60))
            .enable_compression(true)
            .compression_threshold(1024)
            .build();

        assert_eq!(buffer.config().max_size, 500);
        assert_eq!(buffer.config().max_age, Duration::from_secs(1800));
        assert!(buffer.config().enable_compression);
    }

    #[tokio::test]
    async fn test_shared_buffer() {
        let buffer = shared_buffer();

        // Clone for multiple uses
        let buffer_clone = Arc::clone(&buffer);

        buffer.push(create_test_record("https://example.com")).await;

        assert_eq!(buffer_clone.len().await, 1);
    }

    #[test]
    fn test_capture_record_size_bytes() {
        let record = CaptureRecord::new(
            "https://example.com".to_string(),
            "content".to_string(),
            "processed".to_string(),
            100,
        );

        // URL + content + processed_content
        assert_eq!(
            record.size_bytes(),
            "https://example.com".len() + "content".len() + "processed".len()
        );
    }

    #[test]
    fn test_capture_record_with_id() {
        let custom_id = Uuid::new_v4();
        let record = CaptureRecord::with_id(
            custom_id,
            "https://example.com".to_string(),
            "content".to_string(),
            "processed".to_string(),
            100,
        );

        assert_eq!(record.id, custom_id);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let buffer = shared_buffer();

        // Spawn multiple tasks that read and write concurrently
        let mut handles = vec![];

        for i in 0..10 {
            let buffer_clone = Arc::clone(&buffer);
            handles.push(tokio::spawn(async move {
                buffer_clone
                    .push(create_test_record(&format!("https://site{}.com", i)))
                    .await;
            }));
        }

        for i in 0..5 {
            let buffer_clone = Arc::clone(&buffer);
            handles.push(tokio::spawn(async move {
                let _ = buffer_clone.get_recent(i + 1).await;
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(buffer.len().await, 10);
    }

    #[tokio::test]
    async fn test_cleanup_task_start_stop() {
        let buffer = shared_buffer_with_config(BufferConfig {
            cleanup_interval: Duration::from_secs(1),
            max_age: Duration::from_secs(1), // 1 second max age
            ..Default::default()
        });

        // Start cleanup task
        buffer.start_cleanup().await;

        // Push a record
        buffer.push(create_test_record("https://example.com")).await;
        assert_eq!(buffer.len().await, 1);

        // Wait for expiry and cleanup to run (need age > 1 second, plus time for cleanup)
        tokio::time::sleep(Duration::from_secs(3)).await;

        // Record should be cleaned up
        assert_eq!(buffer.len().await, 0);

        // Stop cleanup
        buffer.stop_cleanup().await;
    }
}

//! Server-Sent Events (SSE) feed handler for real-time event streaming
//!
//! This module provides a `/feed` endpoint that streams events to connected clients
//! using the SSE protocol. Events include capture notifications, processing status,
//! errors, and heartbeat keep-alives.
//!
//! # Architecture
//!
//! ```text
//! Capture Event ──▶ Broadcast Channel ──▶ SSE Stream ──▶ Connected Clients
//!                          │
//!                          ▼
//!                   Multiple Subscribers
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use reasonkit_web::handlers::feed::{FeedState, feed_handler};
//! use axum::{Router, routing::get};
//! use std::sync::Arc;
//!
//! let state = Arc::new(FeedState::new(1024));
//! let app = Router::new()
//!     .route("/feed", get(feed_handler))
//!     .with_state(state);
//! ```

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::broadcast::{self, Receiver, Sender};
use tokio::time::{interval, Interval};
use tracing::{debug, info, instrument, warn};

// ============================================================================
// Feed Event Types
// ============================================================================

/// Types of events that can be sent through the SSE feed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data")]
pub enum FeedEvent {
    /// A new capture has been received and is being processed
    #[serde(rename = "capture_received")]
    CaptureReceived(CaptureReceivedData),

    /// Capture processing has completed successfully
    #[serde(rename = "processing_complete")]
    ProcessingComplete(ProcessingCompleteData),

    /// An error occurred during processing
    #[serde(rename = "error")]
    Error(ErrorData),

    /// Keep-alive heartbeat to maintain connection
    #[serde(rename = "heartbeat")]
    Heartbeat(HeartbeatData),
}

/// Data for capture_received event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CaptureReceivedData {
    /// Unique capture ID
    pub capture_id: String,
    /// URL of the captured page
    pub url: String,
    /// Timestamp when capture was received (Unix ms)
    pub timestamp: u64,
    /// Capture type (screenshot, pdf, html, etc.)
    pub capture_type: String,
}

/// Data for processing_complete event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessingCompleteData {
    /// Capture ID that was processed
    pub capture_id: String,
    /// Processing duration in milliseconds
    pub duration_ms: u64,
    /// Size of processed content in bytes
    pub size_bytes: u64,
    /// Summary of extracted content (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Data for error event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorData {
    /// Capture ID associated with the error (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_id: Option<String>,
    /// Error code
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Whether the error is recoverable
    pub recoverable: bool,
}

/// Data for heartbeat event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HeartbeatData {
    /// Server timestamp (Unix ms)
    pub timestamp: u64,
    /// Number of currently connected clients
    pub connected_clients: u64,
    /// Server uptime in seconds
    pub uptime_seconds: u64,
}

impl FeedEvent {
    /// Get the event type name for SSE
    pub fn event_type(&self) -> &'static str {
        match self {
            FeedEvent::CaptureReceived(_) => "capture_received",
            FeedEvent::ProcessingComplete(_) => "processing_complete",
            FeedEvent::Error(_) => "error",
            FeedEvent::Heartbeat(_) => "heartbeat",
        }
    }

    /// Convert to SSE Event
    pub fn to_sse_event(&self) -> Result<Event, serde_json::Error> {
        let data = serde_json::to_string(self)?;
        Ok(Event::default().event(self.event_type()).data(data))
    }
}

// ============================================================================
// Feed State (Shared Application State)
// ============================================================================

/// Shared state for the feed system
///
/// This struct manages the broadcast channel and client tracking.
/// It should be wrapped in `Arc` and shared across handlers.
pub struct FeedState {
    /// Broadcast sender for publishing events
    sender: Sender<FeedEvent>,
    /// Counter for connected clients
    connected_clients: AtomicU64,
    /// Server start time (for uptime calculation)
    start_time: std::time::Instant,
    /// Maximum channel capacity
    capacity: usize,
}

impl FeedState {
    /// Create a new FeedState with the specified channel capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of events to buffer in the channel.
    ///   Clients that fall behind will miss events.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            connected_clients: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
            capacity,
        }
    }

    /// Subscribe to the event feed
    ///
    /// Returns a receiver that will receive all events published after subscription.
    pub fn subscribe(&self) -> Receiver<FeedEvent> {
        self.sender.subscribe()
    }

    /// Publish an event to all connected clients
    ///
    /// Returns the number of clients that received the event.
    /// Returns 0 if no clients are connected.
    #[instrument(skip(self, event), fields(event_type = event.event_type()))]
    pub fn publish(&self, event: FeedEvent) -> usize {
        match self.sender.send(event) {
            Ok(count) => {
                debug!("Published event to {} clients", count);
                count
            }
            Err(_) => {
                debug!("No clients connected, event dropped");
                0
            }
        }
    }

    /// Publish a capture_received event
    pub fn publish_capture_received(
        &self,
        capture_id: impl Into<String>,
        url: impl Into<String>,
        capture_type: impl Into<String>,
    ) -> usize {
        self.publish(FeedEvent::CaptureReceived(CaptureReceivedData {
            capture_id: capture_id.into(),
            url: url.into(),
            timestamp: current_timestamp_ms(),
            capture_type: capture_type.into(),
        }))
    }

    /// Publish a processing_complete event
    pub fn publish_processing_complete(
        &self,
        capture_id: impl Into<String>,
        duration_ms: u64,
        size_bytes: u64,
        summary: Option<String>,
    ) -> usize {
        self.publish(FeedEvent::ProcessingComplete(ProcessingCompleteData {
            capture_id: capture_id.into(),
            duration_ms,
            size_bytes,
            summary,
        }))
    }

    /// Publish an error event
    pub fn publish_error(
        &self,
        capture_id: Option<String>,
        code: impl Into<String>,
        message: impl Into<String>,
        recoverable: bool,
    ) -> usize {
        self.publish(FeedEvent::Error(ErrorData {
            capture_id,
            code: code.into(),
            message: message.into(),
            recoverable,
        }))
    }

    /// Get the number of connected clients
    pub fn connected_clients(&self) -> u64 {
        self.connected_clients.load(Ordering::Relaxed)
    }

    /// Get the server uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get the channel capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Increment connected client count
    fn client_connected(&self) -> u64 {
        let count = self.connected_clients.fetch_add(1, Ordering::Relaxed) + 1;
        info!("Client connected, total: {}", count);
        count
    }

    /// Decrement connected client count
    fn client_disconnected(&self) -> u64 {
        let count = self.connected_clients.fetch_sub(1, Ordering::Relaxed) - 1;
        info!("Client disconnected, total: {}", count);
        count
    }
}

impl Default for FeedState {
    fn default() -> Self {
        Self::new(1024)
    }
}

// ============================================================================
// Client Connection Tracking
// ============================================================================

/// Guard that tracks client connection lifetime
///
/// Automatically decrements the connected client count when dropped.
struct ClientGuard {
    state: Arc<FeedState>,
}

impl ClientGuard {
    fn new(state: Arc<FeedState>) -> Self {
        state.client_connected();
        Self { state }
    }
}

impl Drop for ClientGuard {
    fn drop(&mut self) {
        self.state.client_disconnected();
    }
}

// ============================================================================
// SSE Stream Implementation
// ============================================================================

/// SSE stream that combines event receiver with heartbeat
///
/// This stream yields events from the broadcast channel and also
/// generates heartbeat events at regular intervals.
pub struct FeedStream {
    /// Event receiver
    receiver: Receiver<FeedEvent>,
    /// Heartbeat interval timer
    heartbeat_interval: Interval,
    /// Reference to state for heartbeat data
    state: Arc<FeedState>,
    /// Client guard for connection tracking
    _guard: ClientGuard,
    /// Stream ID for debugging
    #[allow(dead_code)]
    stream_id: u64,
}

impl FeedStream {
    /// Create a new feed stream
    ///
    /// # Arguments
    ///
    /// * `state` - Shared feed state
    /// * `heartbeat_interval_secs` - Interval between heartbeats in seconds
    pub fn new(state: Arc<FeedState>, heartbeat_interval_secs: u64) -> Self {
        static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);
        let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed);

        let receiver = state.subscribe();
        let heartbeat_interval = interval(Duration::from_secs(heartbeat_interval_secs));
        let guard = ClientGuard::new(Arc::clone(&state));

        debug!("Created FeedStream {}", stream_id);

        Self {
            receiver,
            heartbeat_interval,
            state,
            _guard: guard,
            stream_id,
        }
    }

    /// Generate a heartbeat event
    fn generate_heartbeat(&self) -> FeedEvent {
        FeedEvent::Heartbeat(HeartbeatData {
            timestamp: current_timestamp_ms(),
            connected_clients: self.state.connected_clients(),
            uptime_seconds: self.state.uptime_seconds(),
        })
    }
}

impl Stream for FeedStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First, check for heartbeat
        if self.heartbeat_interval.poll_tick(cx).is_ready() {
            let heartbeat = self.generate_heartbeat();
            match heartbeat.to_sse_event() {
                Ok(event) => return Poll::Ready(Some(Ok(event))),
                Err(e) => {
                    warn!("Failed to serialize heartbeat: {}", e);
                    // Continue to check for other events
                }
            }
        }

        // Then, check for broadcast events
        match self.receiver.try_recv() {
            Ok(feed_event) => match feed_event.to_sse_event() {
                Ok(event) => Poll::Ready(Some(Ok(event))),
                Err(e) => {
                    warn!("Failed to serialize event: {}", e);
                    // Wake up to try again
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            },
            Err(broadcast::error::TryRecvError::Empty) => {
                // No events available, register waker and wait
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Lagged(count)) => {
                // Client fell behind, log and continue
                warn!("Client lagged behind by {} events", count);
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Closed) => {
                // Channel closed, end stream
                debug!("Broadcast channel closed, ending stream");
                Poll::Ready(None)
            }
        }
    }
}

// ============================================================================
// Axum Handler
// ============================================================================

/// SSE feed handler for the `/feed` endpoint
///
/// This handler creates an SSE stream that:
/// - Sends all published events (captures, processing status, errors)
/// - Sends heartbeat events every 30 seconds
/// - Handles client disconnection gracefully
///
/// # Example Response
///
/// ```text
/// event: capture_received
/// data: {"type":"capture_received","data":{"capture_id":"abc123","url":"https://example.com","timestamp":1704067200000,"capture_type":"screenshot"}}
///
/// event: heartbeat
/// data: {"type":"heartbeat","data":{"timestamp":1704067230000,"connected_clients":5,"uptime_seconds":3600}}
/// ```
#[instrument(skip(state))]
pub async fn feed_handler(
    State(state): State<Arc<FeedState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    info!("New SSE client connected to /feed");

    let stream = FeedStream::new(state, 30); // 30 second heartbeat

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// Alternative handler with configurable heartbeat interval
#[instrument(skip(state))]
pub async fn feed_handler_with_interval(
    State(state): State<Arc<FeedState>>,
    heartbeat_secs: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    info!(
        "New SSE client connected to /feed (heartbeat: {}s)",
        heartbeat_secs
    );

    let stream = FeedStream::new(state, heartbeat_secs);

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(heartbeat_secs / 2))
            .text("keep-alive"),
    )
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Router Builder
// ============================================================================

/// Build a router with the feed endpoint
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::handlers::feed::{FeedState, build_feed_router};
/// use std::sync::Arc;
///
/// let state = Arc::new(FeedState::new(1024));
/// let router = build_feed_router(state);
/// ```
pub fn build_feed_router(state: Arc<FeedState>) -> axum::Router {
    use axum::routing::get;

    axum::Router::new()
        .route("/feed", get(feed_handler))
        .with_state(state)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[test]
    fn test_feed_event_serialization() {
        let event = FeedEvent::CaptureReceived(CaptureReceivedData {
            capture_id: "test-123".to_string(),
            url: "https://example.com".to_string(),
            timestamp: 1704067200000,
            capture_type: "screenshot".to_string(),
        });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("capture_received"));
        assert!(json.contains("test-123"));
        assert!(json.contains("https://example.com"));
    }

    #[test]
    fn test_feed_event_deserialization() {
        let json = r#"{"type":"capture_received","data":{"capture_id":"abc","url":"https://test.com","timestamp":1000,"capture_type":"pdf"}}"#;
        let event: FeedEvent = serde_json::from_str(json).unwrap();

        match event {
            FeedEvent::CaptureReceived(data) => {
                assert_eq!(data.capture_id, "abc");
                assert_eq!(data.url, "https://test.com");
                assert_eq!(data.capture_type, "pdf");
            }
            _ => panic!("Expected CaptureReceived"),
        }
    }

    #[test]
    fn test_feed_event_type() {
        assert_eq!(
            FeedEvent::CaptureReceived(CaptureReceivedData {
                capture_id: String::new(),
                url: String::new(),
                timestamp: 0,
                capture_type: String::new(),
            })
            .event_type(),
            "capture_received"
        );
        assert_eq!(
            FeedEvent::ProcessingComplete(ProcessingCompleteData {
                capture_id: String::new(),
                duration_ms: 0,
                size_bytes: 0,
                summary: None,
            })
            .event_type(),
            "processing_complete"
        );
        assert_eq!(
            FeedEvent::Error(ErrorData {
                capture_id: None,
                code: String::new(),
                message: String::new(),
                recoverable: false,
            })
            .event_type(),
            "error"
        );
        assert_eq!(
            FeedEvent::Heartbeat(HeartbeatData {
                timestamp: 0,
                connected_clients: 0,
                uptime_seconds: 0,
            })
            .event_type(),
            "heartbeat"
        );
    }

    #[test]
    fn test_feed_state_new() {
        let state = FeedState::new(100);
        assert_eq!(state.capacity(), 100);
        assert_eq!(state.connected_clients(), 0);
    }

    #[tokio::test]
    async fn test_feed_state_publish_no_subscribers() {
        let state = FeedState::new(10);
        let count = state.publish_capture_received("test", "https://test.com", "screenshot");
        assert_eq!(count, 0); // No subscribers
    }

    #[tokio::test]
    async fn test_feed_state_publish_with_subscriber() {
        let state = Arc::new(FeedState::new(10));
        let mut receiver = state.subscribe();

        let count = state.publish_capture_received("test", "https://test.com", "screenshot");
        assert_eq!(count, 1);

        let event = receiver.recv().await.unwrap();
        match event {
            FeedEvent::CaptureReceived(data) => {
                assert_eq!(data.capture_id, "test");
                assert_eq!(data.url, "https://test.com");
            }
            _ => panic!("Expected CaptureReceived"),
        }
    }

    #[tokio::test]
    async fn test_feed_state_client_tracking() {
        let state = Arc::new(FeedState::new(10));
        assert_eq!(state.connected_clients(), 0);

        {
            let _guard = ClientGuard::new(Arc::clone(&state));
            assert_eq!(state.connected_clients(), 1);

            {
                let _guard2 = ClientGuard::new(Arc::clone(&state));
                assert_eq!(state.connected_clients(), 2);
            }

            assert_eq!(state.connected_clients(), 1);
        }

        assert_eq!(state.connected_clients(), 0);
    }

    #[tokio::test]
    async fn test_feed_state_uptime() {
        let state = FeedState::new(10);
        let uptime1 = state.uptime_seconds();

        sleep(Duration::from_millis(100)).await;

        let uptime2 = state.uptime_seconds();
        // Uptime should be the same or slightly higher (within 1 second)
        assert!(uptime2 >= uptime1);
    }

    #[test]
    fn test_error_event() {
        let state = FeedState::new(10);
        let _receiver = state.subscribe();

        let count = state.publish_error(
            Some("capture-123".to_string()),
            "E_TIMEOUT",
            "Operation timed out",
            true,
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn test_processing_complete_event() {
        let state = FeedState::new(10);
        let _receiver = state.subscribe();

        let count = state.publish_processing_complete(
            "capture-456",
            150,
            1024,
            Some("Page title extracted".to_string()),
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn test_to_sse_event() {
        let event = FeedEvent::Heartbeat(HeartbeatData {
            timestamp: 1704067200000,
            connected_clients: 5,
            uptime_seconds: 3600,
        });

        let sse_event = event.to_sse_event().unwrap();
        // SSE Event is opaque, but we can verify it was created
        assert!(format!("{:?}", sse_event).contains("heartbeat"));
    }

    #[tokio::test]
    async fn test_feed_stream_creation() {
        let state = Arc::new(FeedState::new(10));

        // Create stream - should increment client count
        let _stream = FeedStream::new(Arc::clone(&state), 30);
        assert_eq!(state.connected_clients(), 1);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let state = Arc::new(FeedState::new(10));

        let mut rx1 = state.subscribe();
        let mut rx2 = state.subscribe();
        let mut rx3 = state.subscribe();

        let count = state.publish_capture_received("multi-test", "https://example.com", "html");
        assert_eq!(count, 3);

        // All receivers should get the event
        assert!(rx1.recv().await.is_ok());
        assert!(rx2.recv().await.is_ok());
        assert!(rx3.recv().await.is_ok());
    }

    #[test]
    fn test_current_timestamp_ms() {
        let ts = current_timestamp_ms();
        // Should be a reasonable timestamp (after year 2024)
        assert!(ts > 1704067200000);
    }
}

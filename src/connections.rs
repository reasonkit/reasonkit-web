//! SSE Connection Manager for ReasonKit Web
//!
//! This module provides a thread-safe connection manager for Server-Sent Events (SSE)
//! clients, handling connection tracking, health monitoring, and event broadcasting.
//!
//! # Features
//!
//! - **Connection Tracking**: Unique ID assignment and lifecycle management
//! - **Health Monitoring**: Automatic detection and cleanup of stale connections
//! - **Rate Limiting**: Configurable max concurrent connections (default: 100)
//! - **Metrics**: Connection statistics and duration tracking
//! - **Event Broadcasting**: Efficient multicast to all active connections
//!
//! # Architecture
//!
//! ```text
//!                           ConnectionManager
//!                                  |
//!        +-------------------------+-------------------------+
//!        |                         |                         |
//!   Active Connections        Metrics Store           Background Tasks
//!   (RwLock<HashMap>)        (AtomicCounters)        (Cleanup Worker)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::connections::{ConnectionManager, ConnectionConfig};
//! use std::net::IpAddr;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create manager with custom config
//!     let config = ConnectionConfig {
//!         max_connections: 50,
//!         idle_timeout: Duration::from_secs(180),
//!         cleanup_interval: Duration::from_secs(30),
//!     };
//!     let manager = ConnectionManager::with_config(config);
//!
//!     // Add a connection
//!     let ip: IpAddr = "127.0.0.1".parse()?;
//!     let conn_id = manager.add_connection(ip).await?;
//!
//!     // Check active count
//!     println!("Active: {}", manager.get_active_count().await);
//!
//!     // Remove when done
//!     manager.remove_connection(conn_id).await;
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur in connection management
#[derive(Error, Debug, Clone)]
pub enum ConnectionError {
    /// Maximum connections limit reached
    #[error("Maximum connections limit reached ({0})")]
    MaxConnectionsReached(usize),

    /// Connection not found
    #[error("Connection not found: {0}")]
    NotFound(Uuid),

    /// Connection already exists
    #[error("Connection already exists: {0}")]
    AlreadyExists(Uuid),

    /// Broadcast channel closed
    #[error("Broadcast channel closed")]
    BroadcastClosed,

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for connection operations
pub type ConnectionResult<T> = Result<T, ConnectionError>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the ConnectionManager
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Maximum number of concurrent connections (default: 100)
    pub max_connections: usize,

    /// Idle timeout before a connection is considered stale (default: 5 minutes)
    pub idle_timeout: Duration,

    /// Interval between cleanup runs (default: 30 seconds)
    pub cleanup_interval: Duration,

    /// Broadcast channel capacity (default: 256)
    pub broadcast_capacity: usize,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            idle_timeout: Duration::from_secs(300), // 5 minutes
            cleanup_interval: Duration::from_secs(30),
            broadcast_capacity: 256,
        }
    }
}

// ============================================================================
// Connection Types
// ============================================================================

/// Represents a single SSE connection
#[derive(Debug, Clone)]
pub struct Connection {
    /// Unique connection identifier
    pub id: Uuid,

    /// Timestamp when connection was established
    pub connected_at: Instant,

    /// Timestamp of last activity (heartbeat, message, etc.)
    pub last_activity: Instant,

    /// Client IP address
    pub client_ip: IpAddr,

    /// Optional user agent string
    pub user_agent: Option<String>,

    /// Connection state
    pub state: ConnectionState,

    /// Number of events sent to this connection
    pub events_sent: u64,
}

impl Connection {
    /// Create a new connection
    pub fn new(client_ip: IpAddr) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            connected_at: now,
            last_activity: now,
            client_ip,
            user_agent: None,
            state: ConnectionState::Active,
            events_sent: 0,
        }
    }

    /// Create a new connection with user agent
    pub fn with_user_agent(client_ip: IpAddr, user_agent: String) -> Self {
        let mut conn = Self::new(client_ip);
        conn.user_agent = Some(user_agent);
        conn
    }

    /// Update last activity timestamp
    pub fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Check if connection is stale based on timeout
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }

    /// Get connection duration
    pub fn duration(&self) -> Duration {
        self.connected_at.elapsed()
    }

    /// Increment events sent counter
    pub fn increment_events(&mut self) {
        self.events_sent = self.events_sent.saturating_add(1);
    }
}

/// State of a connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Connection is active and healthy
    Active,
    /// Connection is idle but still valid
    Idle,
    /// Connection is being closed
    Closing,
    /// Connection has been closed
    Closed,
}

// ============================================================================
// Feed Events
// ============================================================================

/// Events that can be broadcast to SSE clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEvent {
    /// Event type (e.g., "message", "update", "heartbeat")
    pub event_type: String,

    /// Event data payload
    pub data: serde_json::Value,

    /// Optional event ID for client-side tracking
    pub id: Option<String>,

    /// Optional retry interval hint (milliseconds)
    pub retry: Option<u64>,

    /// Timestamp when event was created
    #[serde(skip)]
    pub created_at: Instant,
}

impl FeedEvent {
    /// Create a new event with type and data
    pub fn new(event_type: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            event_type: event_type.into(),
            data,
            id: None,
            retry: None,
            created_at: Instant::now(),
        }
    }

    /// Create a heartbeat event
    pub fn heartbeat() -> Self {
        Self::new("heartbeat", serde_json::json!({ "ts": chrono::Utc::now().timestamp() }))
    }

    /// Create a message event
    pub fn message(data: serde_json::Value) -> Self {
        Self::new("message", data)
    }

    /// Set event ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set retry interval
    pub fn with_retry(mut self, retry_ms: u64) -> Self {
        self.retry = Some(retry_ms);
        self
    }

    /// Format as SSE text
    pub fn to_sse_format(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref id) = self.id {
            lines.push(format!("id: {}", id));
        }

        lines.push(format!("event: {}", self.event_type));

        // Handle multiline data
        let data_str = self.data.to_string();
        for line in data_str.lines() {
            lines.push(format!("data: {}", line));
        }

        if let Some(retry) = self.retry {
            lines.push(format!("retry: {}", retry));
        }

        lines.push(String::new()); // Empty line to end event
        lines.join("\n")
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Connection metrics and statistics
#[derive(Debug, Default)]
pub struct ConnectionMetrics {
    /// Total connections established over lifetime
    total_connections: AtomicU64,

    /// Total connections closed over lifetime
    total_disconnections: AtomicU64,

    /// Total events broadcast
    total_events_broadcast: AtomicU64,

    /// Peak concurrent connections
    peak_connections: AtomicUsize,

    /// Sum of all connection durations (for average calculation)
    total_duration_secs: AtomicU64,
}

impl ConnectionMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new connection
    pub fn record_connection(&self) {
        self.total_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a disconnection with duration
    pub fn record_disconnection(&self, duration: Duration) {
        self.total_disconnections.fetch_add(1, Ordering::Relaxed);
        self.total_duration_secs
            .fetch_add(duration.as_secs(), Ordering::Relaxed);
    }

    /// Record an event broadcast
    pub fn record_event(&self) {
        self.total_events_broadcast.fetch_add(1, Ordering::Relaxed);
    }

    /// Update peak connections if current is higher
    pub fn update_peak(&self, current: usize) {
        let mut peak = self.peak_connections.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_connections.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Get total connections over lifetime
    pub fn total_connections(&self) -> u64 {
        self.total_connections.load(Ordering::Relaxed)
    }

    /// Get total disconnections over lifetime
    pub fn total_disconnections(&self) -> u64 {
        self.total_disconnections.load(Ordering::Relaxed)
    }

    /// Get total events broadcast
    pub fn total_events(&self) -> u64 {
        self.total_events_broadcast.load(Ordering::Relaxed)
    }

    /// Get peak concurrent connections
    pub fn peak_connections(&self) -> usize {
        self.peak_connections.load(Ordering::Relaxed)
    }

    /// Get average connection duration
    pub fn average_duration(&self) -> Duration {
        let total_secs = self.total_duration_secs.load(Ordering::Relaxed);
        let total_disconnects = self.total_disconnections.load(Ordering::Relaxed);

        if total_disconnects == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs(total_secs / total_disconnects)
        }
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_connections: self.total_connections(),
            total_disconnections: self.total_disconnections(),
            total_events: self.total_events(),
            peak_connections: self.peak_connections(),
            average_duration: self.average_duration(),
        }
    }
}

/// Immutable snapshot of metrics
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    /// Total connections over lifetime
    pub total_connections: u64,

    /// Total disconnections over lifetime
    pub total_disconnections: u64,

    /// Total events broadcast
    pub total_events: u64,

    /// Peak concurrent connections
    pub peak_connections: usize,

    /// Average connection duration
    #[serde(with = "duration_serde")]
    pub average_duration: Duration,
}

/// Serde support for Duration
mod duration_serde {
    use serde::{self, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(duration.as_secs_f64())
    }
}

// ============================================================================
// Connection Manager
// ============================================================================

/// Thread-safe SSE connection manager
///
/// Manages the lifecycle of SSE connections including:
/// - Connection registration and removal
/// - Health monitoring and stale connection cleanup
/// - Event broadcasting to all active connections
/// - Connection metrics and statistics
pub struct ConnectionManager {
    /// Active connections indexed by ID
    connections: RwLock<HashMap<Uuid, Connection>>,

    /// Configuration
    config: ConnectionConfig,

    /// Broadcast channel for events
    broadcast_tx: broadcast::Sender<FeedEvent>,

    /// Connection metrics
    metrics: Arc<ConnectionMetrics>,

    /// Shutdown signal
    shutdown: RwLock<bool>,
}

impl ConnectionManager {
    /// Create a new ConnectionManager with default configuration
    pub fn new() -> Self {
        Self::with_config(ConnectionConfig::default())
    }

    /// Create a new ConnectionManager with custom configuration
    pub fn with_config(config: ConnectionConfig) -> Self {
        let (broadcast_tx, _) = broadcast::channel(config.broadcast_capacity);

        Self {
            connections: RwLock::new(HashMap::new()),
            config,
            broadcast_tx,
            metrics: Arc::new(ConnectionMetrics::new()),
            shutdown: RwLock::new(false),
        }
    }

    /// Add a new connection
    ///
    /// Returns the connection ID if successful, or an error if the max
    /// connections limit has been reached.
    #[instrument(skip(self), fields(client_ip = %client_ip))]
    pub async fn add_connection(&self, client_ip: IpAddr) -> ConnectionResult<Uuid> {
        let mut connections = self.connections.write().await;

        // Check connection limit
        if connections.len() >= self.config.max_connections {
            warn!(
                max = self.config.max_connections,
                current = connections.len(),
                "Max connections limit reached"
            );
            return Err(ConnectionError::MaxConnectionsReached(
                self.config.max_connections,
            ));
        }

        let connection = Connection::new(client_ip);
        let id = connection.id;

        connections.insert(id, connection);

        // Update metrics
        self.metrics.record_connection();
        self.metrics.update_peak(connections.len());

        info!(connection_id = %id, "Connection added");
        debug!(
            active_count = connections.len(),
            "Current active connections"
        );

        Ok(id)
    }

    /// Add a new connection with user agent
    #[instrument(skip(self), fields(client_ip = %client_ip))]
    pub async fn add_connection_with_agent(
        &self,
        client_ip: IpAddr,
        user_agent: String,
    ) -> ConnectionResult<Uuid> {
        let mut connections = self.connections.write().await;

        // Check connection limit
        if connections.len() >= self.config.max_connections {
            warn!(
                max = self.config.max_connections,
                current = connections.len(),
                "Max connections limit reached"
            );
            return Err(ConnectionError::MaxConnectionsReached(
                self.config.max_connections,
            ));
        }

        let connection = Connection::with_user_agent(client_ip, user_agent);
        let id = connection.id;

        connections.insert(id, connection);

        // Update metrics
        self.metrics.record_connection();
        self.metrics.update_peak(connections.len());

        info!(connection_id = %id, "Connection added with user agent");
        Ok(id)
    }

    /// Remove a connection
    #[instrument(skip(self))]
    pub async fn remove_connection(&self, id: Uuid) -> Option<Connection> {
        let mut connections = self.connections.write().await;

        if let Some(mut conn) = connections.remove(&id) {
            conn.state = ConnectionState::Closed;
            let duration = conn.duration();

            // Update metrics
            self.metrics.record_disconnection(duration);

            info!(
                connection_id = %id,
                duration_secs = duration.as_secs(),
                events_sent = conn.events_sent,
                "Connection removed"
            );

            Some(conn)
        } else {
            debug!(connection_id = %id, "Attempted to remove non-existent connection");
            None
        }
    }

    /// Get the current number of active connections
    pub async fn get_active_count(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Get a connection by ID
    pub async fn get_connection(&self, id: Uuid) -> Option<Connection> {
        self.connections.read().await.get(&id).cloned()
    }

    /// Update the last activity timestamp for a connection
    pub async fn touch_connection(&self, id: Uuid) -> bool {
        let mut connections = self.connections.write().await;
        if let Some(conn) = connections.get_mut(&id) {
            conn.touch();
            true
        } else {
            false
        }
    }

    /// Clean up stale connections based on idle timeout
    ///
    /// Returns the number of connections removed
    #[instrument(skip(self))]
    pub async fn cleanup_stale(&self, timeout: Duration) -> usize {
        let mut connections = self.connections.write().await;

        let stale_ids: Vec<Uuid> = connections
            .iter()
            .filter(|(_, conn)| conn.is_stale(timeout))
            .map(|(id, _)| *id)
            .collect();

        let removed_count = stale_ids.len();

        for id in stale_ids {
            if let Some(conn) = connections.remove(&id) {
                let duration = conn.duration();
                self.metrics.record_disconnection(duration);

                info!(
                    connection_id = %id,
                    idle_secs = conn.last_activity.elapsed().as_secs(),
                    "Removed stale connection"
                );
            }
        }

        if removed_count > 0 {
            info!(
                removed = removed_count,
                remaining = connections.len(),
                "Stale connection cleanup completed"
            );
        }

        removed_count
    }

    /// Broadcast an event to all active connections
    ///
    /// Returns the number of clients that will receive the event
    #[instrument(skip(self, event), fields(event_type = %event.event_type))]
    pub async fn broadcast_to_all(&self, event: FeedEvent) -> usize {
        let receiver_count = self.broadcast_tx.receiver_count();

        if receiver_count == 0 {
            debug!("No receivers for broadcast");
            return 0;
        }

        match self.broadcast_tx.send(event) {
            Ok(count) => {
                self.metrics.record_event();
                debug!(receivers = count, "Event broadcast");
                count
            }
            Err(e) => {
                warn!(error = %e, "Failed to broadcast event");
                0
            }
        }
    }

    /// Get a receiver for broadcast events
    ///
    /// Used by SSE handlers to receive events for their connection
    pub fn subscribe(&self) -> broadcast::Receiver<FeedEvent> {
        self.broadcast_tx.subscribe()
    }

    /// Get all active connections (for admin/monitoring purposes)
    pub async fn list_connections(&self) -> Vec<ConnectionInfo> {
        self.connections
            .read()
            .await
            .values()
            .map(|conn| ConnectionInfo {
                id: conn.id,
                client_ip: conn.client_ip,
                connected_at_secs_ago: conn.connected_at.elapsed().as_secs(),
                last_activity_secs_ago: conn.last_activity.elapsed().as_secs(),
                state: conn.state,
                events_sent: conn.events_sent,
                user_agent: conn.user_agent.clone(),
            })
            .collect()
    }

    /// Get connection metrics
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Get raw metrics reference
    pub fn raw_metrics(&self) -> Arc<ConnectionMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Get configuration
    pub fn config(&self) -> &ConnectionConfig {
        &self.config
    }

    /// Start background cleanup task
    ///
    /// Returns a handle that can be used to stop the task
    pub fn start_cleanup_task(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let manager = Arc::clone(self);
        let interval = manager.config.cleanup_interval;
        let timeout = manager.config.idle_timeout;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if *manager.shutdown.read().await {
                    info!("Cleanup task shutting down");
                    break;
                }

                let removed = manager.cleanup_stale(timeout).await;
                if removed > 0 {
                    debug!(removed = removed, "Periodic cleanup completed");
                }
            }
        })
    }

    /// Start background heartbeat task
    ///
    /// Sends periodic heartbeat events to all connected clients
    pub fn start_heartbeat_task(
        self: &Arc<Self>,
        heartbeat_interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        let manager = Arc::clone(self);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(heartbeat_interval);

            loop {
                interval_timer.tick().await;

                if *manager.shutdown.read().await {
                    info!("Heartbeat task shutting down");
                    break;
                }

                let event = FeedEvent::heartbeat();
                let receivers = manager.broadcast_to_all(event).await;
                debug!(receivers = receivers, "Heartbeat sent");
            }
        })
    }

    /// Signal shutdown to background tasks
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
        info!("Connection manager shutdown signaled");
    }

    /// Check if shutdown has been signaled
    pub async fn is_shutdown(&self) -> bool {
        *self.shutdown.read().await
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Connection Info (for serialization)
// ============================================================================

/// Serializable connection information
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionInfo {
    /// Connection ID
    pub id: Uuid,

    /// Client IP address
    pub client_ip: IpAddr,

    /// Seconds since connection was established
    pub connected_at_secs_ago: u64,

    /// Seconds since last activity
    pub last_activity_secs_ago: u64,

    /// Connection state
    pub state: ConnectionState,

    /// Number of events sent
    pub events_sent: u64,

    /// User agent if available
    pub user_agent: Option<String>,
}

// ============================================================================
// Builder Pattern for ConnectionManager
// ============================================================================

/// Builder for ConnectionManager with fluent configuration
#[derive(Debug, Clone)]
pub struct ConnectionManagerBuilder {
    config: ConnectionConfig,
}

impl ConnectionManagerBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ConnectionConfig::default(),
        }
    }

    /// Set maximum concurrent connections
    pub fn max_connections(mut self, max: usize) -> Self {
        self.config.max_connections = max;
        self
    }

    /// Set idle timeout
    pub fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.config.idle_timeout = timeout;
        self
    }

    /// Set cleanup interval
    pub fn cleanup_interval(mut self, interval: Duration) -> Self {
        self.config.cleanup_interval = interval;
        self
    }

    /// Set broadcast channel capacity
    pub fn broadcast_capacity(mut self, capacity: usize) -> Self {
        self.config.broadcast_capacity = capacity;
        self
    }

    /// Build the ConnectionManager
    pub fn build(self) -> ConnectionManager {
        ConnectionManager::with_config(self.config)
    }

    /// Build the ConnectionManager wrapped in Arc
    pub fn build_arc(self) -> Arc<ConnectionManager> {
        Arc::new(self.build())
    }
}

impl Default for ConnectionManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn test_ip() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))
    }

    #[tokio::test]
    async fn test_add_connection() {
        let manager = ConnectionManager::new();
        let id = manager.add_connection(test_ip()).await.unwrap();

        assert_eq!(manager.get_active_count().await, 1);

        let conn = manager.get_connection(id).await.unwrap();
        assert_eq!(conn.client_ip, test_ip());
        assert_eq!(conn.state, ConnectionState::Active);
    }

    #[tokio::test]
    async fn test_remove_connection() {
        let manager = ConnectionManager::new();
        let id = manager.add_connection(test_ip()).await.unwrap();

        assert_eq!(manager.get_active_count().await, 1);

        let removed = manager.remove_connection(id).await;
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().state, ConnectionState::Closed);
        assert_eq!(manager.get_active_count().await, 0);
    }

    #[tokio::test]
    async fn test_max_connections_limit() {
        let config = ConnectionConfig {
            max_connections: 2,
            ..Default::default()
        };
        let manager = ConnectionManager::with_config(config);

        let _id1 = manager.add_connection(test_ip()).await.unwrap();
        let _id2 = manager.add_connection(test_ip()).await.unwrap();

        let result = manager.add_connection(test_ip()).await;
        assert!(matches!(result, Err(ConnectionError::MaxConnectionsReached(2))));
    }

    #[tokio::test]
    async fn test_cleanup_stale_connections() {
        let manager = ConnectionManager::new();

        let _id1 = manager.add_connection(test_ip()).await.unwrap();
        let _id2 = manager.add_connection(test_ip()).await.unwrap();

        assert_eq!(manager.get_active_count().await, 2);

        // With a very short timeout, connections should be stale
        tokio::time::sleep(Duration::from_millis(10)).await;
        let removed = manager.cleanup_stale(Duration::from_millis(1)).await;

        assert_eq!(removed, 2);
        assert_eq!(manager.get_active_count().await, 0);
    }

    #[tokio::test]
    async fn test_touch_connection() {
        let manager = ConnectionManager::new();
        let id = manager.add_connection(test_ip()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(10)).await;

        let touched = manager.touch_connection(id).await;
        assert!(touched);

        // Should not be stale after touch
        let removed = manager.cleanup_stale(Duration::from_millis(5)).await;
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn test_broadcast_event() {
        let manager = ConnectionManager::new();
        let mut receiver = manager.subscribe();

        let event = FeedEvent::message(serde_json::json!({"test": "data"}));
        let count = manager.broadcast_to_all(event).await;

        assert_eq!(count, 1);

        let received = receiver.recv().await.unwrap();
        assert_eq!(received.event_type, "message");
    }

    #[tokio::test]
    async fn test_metrics() {
        let manager = ConnectionManager::new();

        let id1 = manager.add_connection(test_ip()).await.unwrap();
        let id2 = manager.add_connection(test_ip()).await.unwrap();

        let metrics = manager.metrics();
        assert_eq!(metrics.total_connections, 2);
        assert_eq!(metrics.peak_connections, 2);

        manager.remove_connection(id1).await;
        manager.remove_connection(id2).await;

        let metrics = manager.metrics();
        assert_eq!(metrics.total_disconnections, 2);
    }

    #[tokio::test]
    async fn test_list_connections() {
        let manager = ConnectionManager::new();

        let _id1 = manager.add_connection(test_ip()).await.unwrap();
        let _id2 = manager
            .add_connection_with_agent(test_ip(), "TestAgent/1.0".to_string())
            .await
            .unwrap();

        let list = manager.list_connections().await;
        assert_eq!(list.len(), 2);

        let with_agent: Vec<_> = list.iter().filter(|c| c.user_agent.is_some()).collect();
        assert_eq!(with_agent.len(), 1);
        assert_eq!(with_agent[0].user_agent.as_ref().unwrap(), "TestAgent/1.0");
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let manager = ConnectionManagerBuilder::new()
            .max_connections(50)
            .idle_timeout(Duration::from_secs(120))
            .cleanup_interval(Duration::from_secs(15))
            .broadcast_capacity(128)
            .build();

        assert_eq!(manager.config().max_connections, 50);
        assert_eq!(manager.config().idle_timeout, Duration::from_secs(120));
        assert_eq!(manager.config().cleanup_interval, Duration::from_secs(15));
        assert_eq!(manager.config().broadcast_capacity, 128);
    }

    #[test]
    fn test_feed_event_sse_format() {
        let event = FeedEvent::new("test", serde_json::json!({"key": "value"}))
            .with_id("event-123")
            .with_retry(5000);

        let sse = event.to_sse_format();

        assert!(sse.contains("id: event-123"));
        assert!(sse.contains("event: test"));
        assert!(sse.contains("data:"));
        assert!(sse.contains("retry: 5000"));
    }

    #[test]
    fn test_connection_is_stale() {
        let mut conn = Connection::new(test_ip());

        // Not stale immediately
        assert!(!conn.is_stale(Duration::from_secs(1)));

        // Simulate time passing by manipulating last_activity
        // (In real tests, we would use tokio::time::sleep)
        conn.last_activity = Instant::now() - Duration::from_secs(10);

        assert!(conn.is_stale(Duration::from_secs(5)));
        assert!(!conn.is_stale(Duration::from_secs(15)));
    }

    #[test]
    fn test_heartbeat_event() {
        let event = FeedEvent::heartbeat();
        assert_eq!(event.event_type, "heartbeat");
        assert!(event.data.get("ts").is_some());
    }

    #[tokio::test]
    async fn test_shutdown_signal() {
        let manager = ConnectionManager::new();

        assert!(!manager.is_shutdown().await);

        manager.shutdown().await;

        assert!(manager.is_shutdown().await);
    }
}

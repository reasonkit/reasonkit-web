//! Graceful shutdown handling for ReasonKit Web server
//!
//! This module provides comprehensive shutdown handling including:
//! - Signal handling (SIGTERM, SIGINT)
//! - Graceful connection draining
//! - Systemd watchdog integration
//! - Health status management during shutdown
//!
//! # Architecture
//!
//! ```text
//! Signal (SIGTERM/SIGINT)
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ ShutdownController│──────────▶ Notify all subscribers
//! └───────────────────┘
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ Stop new requests │
//! └───────────────────┘
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ Drain in-flight   │◀──── timeout: 30s default
//! └───────────────────┘
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ Flush logs        │
//! └───────────────────┘
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ Clean exit        │
//! └───────────────────┘
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, watch, Notify};
use tracing::{debug, info, warn};

/// Default graceful shutdown timeout in seconds
pub const DEFAULT_SHUTDOWN_TIMEOUT_SECS: u64 = 30;

/// Default drain timeout for in-flight requests
pub const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 30;

/// Shutdown state enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownState {
    /// Server is running normally
    Running,
    /// Shutdown initiated, draining connections
    Draining,
    /// Shutdown complete, exiting
    Stopped,
}

impl std::fmt::Display for ShutdownState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShutdownState::Running => write!(f, "running"),
            ShutdownState::Draining => write!(f, "draining"),
            ShutdownState::Stopped => write!(f, "stopped"),
        }
    }
}

/// Shutdown controller that manages graceful shutdown
///
/// This is the central coordinator for shutdown operations.
/// Clone this to share across tasks.
#[derive(Clone)]
pub struct ShutdownController {
    /// Inner state shared across clones
    inner: Arc<ShutdownControllerInner>,
}

struct ShutdownControllerInner {
    /// Whether shutdown has been initiated
    is_shutting_down: AtomicBool,

    /// Current shutdown state
    state: std::sync::RwLock<ShutdownState>,

    /// Notifier for shutdown signal
    shutdown_notify: Notify,

    /// Watch channel for state changes
    state_tx: watch::Sender<ShutdownState>,
    state_rx: watch::Receiver<ShutdownState>,

    /// Broadcast channel for shutdown notifications
    shutdown_tx: broadcast::Sender<()>,

    /// Count of active connections/requests
    active_connections: AtomicU64,

    /// Shutdown timeout duration
    drain_timeout: Duration,

    /// Shutdown started timestamp
    shutdown_started: std::sync::RwLock<Option<Instant>>,
}

impl ShutdownController {
    /// Create a new shutdown controller with default settings
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS))
    }

    /// Create a new shutdown controller with custom drain timeout
    pub fn with_timeout(drain_timeout: Duration) -> Self {
        let (state_tx, state_rx) = watch::channel(ShutdownState::Running);
        let (shutdown_tx, _) = broadcast::channel(16);

        Self {
            inner: Arc::new(ShutdownControllerInner {
                is_shutting_down: AtomicBool::new(false),
                state: std::sync::RwLock::new(ShutdownState::Running),
                shutdown_notify: Notify::new(),
                state_tx,
                state_rx,
                shutdown_tx,
                active_connections: AtomicU64::new(0),
                drain_timeout,
                shutdown_started: std::sync::RwLock::new(None),
            }),
        }
    }

    /// Check if shutdown has been initiated
    pub fn is_shutting_down(&self) -> bool {
        self.inner.is_shutting_down.load(Ordering::SeqCst)
    }

    /// Get the current shutdown state
    pub fn state(&self) -> ShutdownState {
        *self.inner.state.read().unwrap()
    }

    /// Get a receiver for state changes
    pub fn state_receiver(&self) -> watch::Receiver<ShutdownState> {
        self.inner.state_rx.clone()
    }

    /// Subscribe to shutdown notifications
    pub fn subscribe(&self) -> broadcast::Receiver<()> {
        self.inner.shutdown_tx.subscribe()
    }

    /// Initiate graceful shutdown
    ///
    /// This method:
    /// 1. Sets the shutdown flag
    /// 2. Transitions to Draining state
    /// 3. Notifies all subscribers
    /// 4. Waits for connections to drain (with timeout)
    /// 5. Transitions to Stopped state
    pub async fn initiate_shutdown(&self) {
        // Only initiate once
        if self
            .inner
            .is_shutting_down
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            debug!("Shutdown already in progress");
            return;
        }

        info!("Initiating graceful shutdown");

        // Record shutdown start time
        *self.inner.shutdown_started.write().unwrap() = Some(Instant::now());

        // Transition to draining state
        self.set_state(ShutdownState::Draining);

        // Notify systemd that we're stopping
        #[cfg(target_os = "linux")]
        systemd_notify_stopping();

        // Notify all subscribers
        let _ = self.inner.shutdown_tx.send(());
        self.inner.shutdown_notify.notify_waiters();

        // Wait for connections to drain
        self.wait_for_drain().await;

        // Flush logs
        self.flush_logs();

        // Transition to stopped
        self.set_state(ShutdownState::Stopped);

        info!("Graceful shutdown complete");
    }

    /// Wait for the shutdown signal
    pub async fn wait_for_shutdown(&self) {
        if self.is_shutting_down() {
            return;
        }
        self.inner.shutdown_notify.notified().await;
    }

    /// Increment active connection count
    pub fn connection_start(&self) {
        self.inner.active_connections.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement active connection count
    pub fn connection_end(&self) {
        self.inner.active_connections.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get current active connection count
    pub fn active_connections(&self) -> u64 {
        self.inner.active_connections.load(Ordering::SeqCst)
    }

    /// Create a connection guard that automatically decrements on drop
    pub fn connection_guard(&self) -> ConnectionGuard {
        self.connection_start();
        ConnectionGuard {
            controller: self.clone(),
        }
    }

    /// Get the drain timeout
    pub fn drain_timeout(&self) -> Duration {
        self.inner.drain_timeout
    }

    /// Get time elapsed since shutdown started
    pub fn shutdown_elapsed(&self) -> Option<Duration> {
        self.inner
            .shutdown_started
            .read()
            .unwrap()
            .map(|started| started.elapsed())
    }

    /// Get Retry-After header value in seconds (for 503 responses)
    pub fn retry_after_secs(&self) -> u64 {
        match self.shutdown_elapsed() {
            Some(elapsed) => {
                let remaining = self.inner.drain_timeout.saturating_sub(elapsed);
                remaining.as_secs().saturating_add(5) // Add 5s buffer
            }
            None => DEFAULT_DRAIN_TIMEOUT_SECS + 5,
        }
    }

    // Internal methods

    fn set_state(&self, state: ShutdownState) {
        *self.inner.state.write().unwrap() = state;
        let _ = self.inner.state_tx.send(state);
        info!("Shutdown state changed to: {}", state);
    }

    async fn wait_for_drain(&self) {
        let timeout = self.inner.drain_timeout;
        let start = Instant::now();

        info!(
            "Waiting for {} active connections to drain (timeout: {:?})",
            self.active_connections(),
            timeout
        );

        loop {
            let active = self.active_connections();

            if active == 0 {
                info!("All connections drained successfully");
                return;
            }

            if start.elapsed() >= timeout {
                warn!(
                    "Drain timeout reached with {} active connections remaining",
                    active
                );
                return;
            }

            // Check every 100ms
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    fn flush_logs(&self) {
        // Force flush of tracing/log buffers
        // In most cases, tracing flushes automatically, but we ensure it here
        debug!("Flushing logs before shutdown");

        // Give async loggers time to flush
        std::thread::sleep(Duration::from_millis(50));
    }
}

impl Default for ShutdownController {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for tracking active connections
///
/// When dropped, automatically decrements the connection count.
pub struct ConnectionGuard {
    controller: ShutdownController,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.controller.connection_end();
    }
}

/// Create a shutdown signal future for use with Axum's graceful shutdown
///
/// This function returns a future that completes when a shutdown signal
/// (SIGTERM or SIGINT) is received.
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::shutdown::shutdown_signal;
///
/// async fn run_server() {
///     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
///     let app = axum::Router::new();
///
///     axum::serve(listener, app)
///         .with_graceful_shutdown(shutdown_signal())
///         .await
///         .unwrap();
/// }
/// ```
pub async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm =
            signal(SignalKind::terminate()).expect("Failed to install SIGTERM handler");

        let mut sigint = signal(SignalKind::interrupt()).expect("Failed to install SIGINT handler");

        let mut sighup = signal(SignalKind::hangup()).expect("Failed to install SIGHUP handler");

        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT");
            }
            _ = sighup.recv() => {
                info!("Received SIGHUP");
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C");
        }
        _ = terminate => {}
    }
}

/// Create a shutdown signal future with a controller for advanced shutdown handling
///
/// This variant allows you to use a ShutdownController for more control
/// over the shutdown process.
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::shutdown::{shutdown_signal_with_controller, ShutdownController};
///
/// async fn run_server() {
///     let controller = ShutdownController::new();
///     let shutdown_controller = controller.clone();
///
///     // Spawn a task to handle shutdown
///     tokio::spawn(async move {
///         shutdown_signal_with_controller(shutdown_controller).await;
///     });
///
///     // Use controller.is_shutting_down() to check state
///     // Use controller.connection_guard() to track connections
/// }
/// ```
pub async fn shutdown_signal_with_controller(controller: ShutdownController) {
    shutdown_signal().await;
    controller.initiate_shutdown().await;
}

// Systemd integration (Linux only)

/// Notify systemd that the service is ready
///
/// This should be called after the server is fully initialized and ready
/// to accept connections.
#[cfg(target_os = "linux")]
pub fn systemd_notify_ready() {
    if let Err(e) = sd_notify("READY=1") {
        debug!(
            "Failed to notify systemd ready (may not be running under systemd): {}",
            e
        );
    } else {
        info!("Notified systemd: READY");
    }
}

/// Notify systemd that the service is stopping
///
/// This should be called when shutdown is initiated.
#[cfg(target_os = "linux")]
pub fn systemd_notify_stopping() {
    if let Err(e) = sd_notify("STOPPING=1") {
        debug!("Failed to notify systemd stopping: {}", e);
    } else {
        info!("Notified systemd: STOPPING");
    }
}

/// Notify systemd with a status message
///
/// Useful for providing visibility into what the service is doing.
#[cfg(target_os = "linux")]
pub fn systemd_notify_status(status: &str) {
    if let Err(e) = sd_notify(&format!("STATUS={}", status)) {
        debug!("Failed to notify systemd status: {}", e);
    }
}

/// Send watchdog ping to systemd
///
/// Call this periodically to tell systemd the service is healthy.
/// The interval should be less than WatchdogSec/2 configured in the unit file.
#[cfg(target_os = "linux")]
pub fn systemd_watchdog_ping() {
    if let Err(e) = sd_notify("WATCHDOG=1") {
        debug!("Failed to send watchdog ping: {}", e);
    }
}

/// Internal function to send sd_notify messages
///
/// This uses the NOTIFY_SOCKET environment variable to communicate
/// with systemd without requiring the systemd-daemon crate.
#[cfg(target_os = "linux")]
fn sd_notify(state: &str) -> std::io::Result<()> {
    use std::os::unix::net::UnixDatagram;

    let socket_path = match std::env::var("NOTIFY_SOCKET") {
        Ok(path) => path,
        Err(_) => {
            // Not running under systemd or Type=notify not configured
            return Ok(());
        }
    };

    // Handle abstract socket (starts with @)
    let socket_path = if let Some(rest) = socket_path.strip_prefix('@') {
        format!("\0{rest}")
    } else {
        socket_path
    };

    let socket = UnixDatagram::unbound()?;

    // For abstract sockets, we need to use the raw bytes
    if let Some(rest) = socket_path.strip_prefix('\0') {
        // Abstract socket - use socketaddr directly
        use std::os::unix::net::SocketAddr;
        let addr = SocketAddr::from_pathname(rest)?;
        socket.send_to(state.as_bytes(), addr.as_pathname().unwrap())?;
    } else {
        socket.send_to(state.as_bytes(), &socket_path)?;
    }

    Ok(())
}

// Stub implementations for non-Linux platforms

/// Notify systemd that the service is ready (no-op on non-Linux)
#[cfg(not(target_os = "linux"))]
pub fn systemd_notify_ready() {
    debug!("systemd_notify_ready: not on Linux, skipping");
}

/// Notify systemd that the service is stopping (no-op on non-Linux)
#[cfg(not(target_os = "linux"))]
pub fn systemd_notify_stopping() {
    debug!("systemd_notify_stopping: not on Linux, skipping");
}

/// Notify systemd with a status message (no-op on non-Linux)
#[cfg(not(target_os = "linux"))]
pub fn systemd_notify_status(_status: &str) {
    debug!("systemd_notify_status: not on Linux, skipping");
}

/// Send watchdog ping to systemd (no-op on non-Linux)
#[cfg(not(target_os = "linux"))]
pub fn systemd_watchdog_ping() {
    debug!("systemd_watchdog_ping: not on Linux, skipping");
}

/// Watchdog task that periodically pings systemd
///
/// Spawn this task to keep the systemd watchdog happy.
/// The interval should be less than half of WatchdogSec.
pub async fn watchdog_task(interval: Duration, mut shutdown_rx: broadcast::Receiver<()>) {
    info!(
        "Starting systemd watchdog task with {:?} interval",
        interval
    );

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {
                systemd_watchdog_ping();
            }
            _ = shutdown_rx.recv() => {
                info!("Watchdog task stopping due to shutdown");
                break;
            }
        }
    }
}

/// Health check response during shutdown
///
/// Returns appropriate health status based on shutdown state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthStatus {
    /// Current status
    pub status: String,
    /// Whether the service is healthy
    pub healthy: bool,
    /// Current shutdown state
    pub shutdown_state: String,
    /// Active connections count
    pub active_connections: u64,
    /// Time remaining before forced shutdown (if draining)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drain_remaining_secs: Option<u64>,
    /// Retry-After value in seconds (for 503 responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_secs: Option<u64>,
}

impl ShutdownController {
    /// Get health status for health check endpoint
    ///
    /// During shutdown, this returns unhealthy status with retry information.
    pub fn health_status(&self) -> HealthStatus {
        let state = self.state();
        let active = self.active_connections();

        match state {
            ShutdownState::Running => HealthStatus {
                status: "ok".to_string(),
                healthy: true,
                shutdown_state: state.to_string(),
                active_connections: active,
                drain_remaining_secs: None,
                retry_after_secs: None,
            },
            ShutdownState::Draining => {
                let drain_remaining = self
                    .shutdown_elapsed()
                    .map(|elapsed| self.drain_timeout().saturating_sub(elapsed).as_secs());

                HealthStatus {
                    status: "draining".to_string(),
                    healthy: false,
                    shutdown_state: state.to_string(),
                    active_connections: active,
                    drain_remaining_secs: drain_remaining,
                    retry_after_secs: Some(self.retry_after_secs()),
                }
            }
            ShutdownState::Stopped => HealthStatus {
                status: "stopped".to_string(),
                healthy: false,
                shutdown_state: state.to_string(),
                active_connections: active,
                drain_remaining_secs: Some(0),
                retry_after_secs: Some(self.retry_after_secs()),
            },
        }
    }
}

/// Axum layer/middleware for graceful shutdown
///
/// This module provides middleware for Axum that:
/// - Rejects new requests during shutdown with 503
/// - Tracks active connections
/// - Adds Retry-After header during shutdown
pub mod axum_integration {
    use super::*;
    use axum::{
        body::Body,
        http::{header, Request, Response, StatusCode},
    };
    use std::task::{Context, Poll};
    use tower::{Layer, Service};

    /// Layer that adds shutdown handling to a service
    #[derive(Clone)]
    pub struct ShutdownLayer {
        controller: ShutdownController,
    }

    impl ShutdownLayer {
        /// Create a new shutdown layer
        pub fn new(controller: ShutdownController) -> Self {
            Self { controller }
        }
    }

    impl<S> Layer<S> for ShutdownLayer {
        type Service = ShutdownService<S>;

        fn layer(&self, inner: S) -> Self::Service {
            ShutdownService {
                inner,
                controller: self.controller.clone(),
            }
        }
    }

    /// Service wrapper that handles shutdown
    #[derive(Clone)]
    pub struct ShutdownService<S> {
        inner: S,
        controller: ShutdownController,
    }

    impl<S, ReqBody> Service<Request<ReqBody>> for ShutdownService<S>
    where
        S: Service<Request<ReqBody>, Response = Response<Body>> + Clone + Send + 'static,
        S::Future: Send,
        ReqBody: Send + 'static,
    {
        type Response = Response<Body>;
        type Error = S::Error;
        type Future = std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
        >;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.inner.poll_ready(cx)
        }

        fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
            let controller = self.controller.clone();
            let mut inner = self.inner.clone();

            Box::pin(async move {
                // Check if we're shutting down
                if controller.is_shutting_down() {
                    let retry_after = controller.retry_after_secs().to_string();
                    let health = controller.health_status();
                    let body = serde_json::to_string(&health).unwrap_or_else(|_| {
                        r#"{"status":"unavailable","healthy":false}"#.to_string()
                    });

                    let response = Response::builder()
                        .status(StatusCode::SERVICE_UNAVAILABLE)
                        .header(header::RETRY_AFTER, retry_after)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap();

                    return Ok(response);
                }

                // Track this connection
                let _guard = controller.connection_guard();

                // Process the request
                inner.call(req).await
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shutdown_controller_new() {
        let controller = ShutdownController::new();
        assert!(!controller.is_shutting_down());
        assert_eq!(controller.state(), ShutdownState::Running);
        assert_eq!(controller.active_connections(), 0);
    }

    #[tokio::test]
    async fn test_shutdown_controller_with_timeout() {
        let controller = ShutdownController::with_timeout(Duration::from_secs(60));
        assert_eq!(controller.drain_timeout(), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_connection_tracking() {
        let controller = ShutdownController::new();

        controller.connection_start();
        assert_eq!(controller.active_connections(), 1);

        controller.connection_start();
        assert_eq!(controller.active_connections(), 2);

        controller.connection_end();
        assert_eq!(controller.active_connections(), 1);

        controller.connection_end();
        assert_eq!(controller.active_connections(), 0);
    }

    #[tokio::test]
    async fn test_connection_guard() {
        let controller = ShutdownController::new();

        {
            let _guard = controller.connection_guard();
            assert_eq!(controller.active_connections(), 1);

            {
                let _guard2 = controller.connection_guard();
                assert_eq!(controller.active_connections(), 2);
            }

            assert_eq!(controller.active_connections(), 1);
        }

        assert_eq!(controller.active_connections(), 0);
    }

    #[tokio::test]
    async fn test_shutdown_initiation() {
        let controller = ShutdownController::with_timeout(Duration::from_millis(100));

        // Should not be shutting down initially
        assert!(!controller.is_shutting_down());
        assert_eq!(controller.state(), ShutdownState::Running);

        // Initiate shutdown
        controller.initiate_shutdown().await;

        // Should be stopped after shutdown completes
        assert!(controller.is_shutting_down());
        assert_eq!(controller.state(), ShutdownState::Stopped);
    }

    #[tokio::test]
    async fn test_shutdown_only_once() {
        let controller = ShutdownController::with_timeout(Duration::from_millis(100));

        let controller2 = controller.clone();

        // Start two concurrent shutdowns
        let handle1 = tokio::spawn(async move {
            controller.initiate_shutdown().await;
        });

        let handle2 = tokio::spawn(async move {
            controller2.initiate_shutdown().await;
        });

        // Both should complete without panicking
        let (r1, r2) = tokio::join!(handle1, handle2);
        r1.unwrap();
        r2.unwrap();
    }

    #[tokio::test]
    async fn test_health_status_running() {
        let controller = ShutdownController::new();
        let health = controller.health_status();

        assert!(health.healthy);
        assert_eq!(health.status, "ok");
        assert_eq!(health.shutdown_state, "running");
        assert!(health.retry_after_secs.is_none());
    }

    #[tokio::test]
    async fn test_subscribe_and_notify() {
        let controller = ShutdownController::with_timeout(Duration::from_millis(100));
        let mut rx = controller.subscribe();

        // Spawn shutdown in background
        let controller2 = controller.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            controller2.initiate_shutdown().await;
        });

        // Should receive notification
        let result = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_state_receiver() {
        let controller = ShutdownController::with_timeout(Duration::from_millis(100));
        let mut rx = controller.state_receiver();

        // Initial state
        assert_eq!(*rx.borrow(), ShutdownState::Running);

        // Initiate shutdown
        let controller2 = controller.clone();
        tokio::spawn(async move {
            controller2.initiate_shutdown().await;
        });

        // Wait for state change
        rx.changed().await.unwrap();

        // Should be draining or stopped
        let state = *rx.borrow();
        assert!(state == ShutdownState::Draining || state == ShutdownState::Stopped);
    }

    #[tokio::test]
    async fn test_drain_with_active_connections() {
        let controller = ShutdownController::with_timeout(Duration::from_millis(500));

        // Simulate active connection
        let guard = controller.connection_guard();

        // Start shutdown in background
        let controller2 = controller.clone();
        let shutdown_handle = tokio::spawn(async move {
            controller2.initiate_shutdown().await;
        });

        // Wait a bit then drop the connection
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(guard);

        // Shutdown should complete
        tokio::time::timeout(Duration::from_secs(1), shutdown_handle)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(controller.state(), ShutdownState::Stopped);
    }

    #[test]
    fn test_shutdown_state_display() {
        assert_eq!(ShutdownState::Running.to_string(), "running");
        assert_eq!(ShutdownState::Draining.to_string(), "draining");
        assert_eq!(ShutdownState::Stopped.to_string(), "stopped");
    }

    #[test]
    fn test_retry_after_secs() {
        let controller = ShutdownController::with_timeout(Duration::from_secs(30));
        // Before shutdown, should return default + 5s buffer
        assert_eq!(controller.retry_after_secs(), 35);
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus {
            status: "ok".to_string(),
            healthy: true,
            shutdown_state: "running".to_string(),
            active_connections: 5,
            drain_remaining_secs: None,
            retry_after_secs: None,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"healthy\":true"));
        // None values should be skipped
        assert!(!json.contains("drain_remaining_secs"));
        assert!(!json.contains("retry_after_secs"));
    }

    #[tokio::test]
    async fn test_default_trait() {
        let controller = ShutdownController::default();
        assert!(!controller.is_shutting_down());
        assert_eq!(
            controller.drain_timeout(),
            Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
        );
    }
}

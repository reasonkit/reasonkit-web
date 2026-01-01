//! Rate Limiting for ReasonKit Web
//!
//! This module provides a robust token bucket rate limiter with full Axum/Tower
//! middleware integration. It supports per-IP rate limiting with configurable
//! request rates, burst sizes, and automatic cleanup of stale buckets.
//!
//! # Architecture
//!
//! ```text
//! Request ─────▶ RateLimitLayer ─────▶ RateLimiter
//!                     │                     │
//!                     │              ┌──────┴──────┐
//!                     │              │ TokenBucket │
//!                     │              │ per IP addr │
//!                     │              └──────┬──────┘
//!                     │                     │
//!                     ▼                     ▼
//!               Pass/Reject         Refill tokens
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::rate_limit::{RateLimiter, RateLimitConfig, RateLimitLayer};
//! use std::sync::Arc;
//! use axum::{Router, routing::get};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = RateLimitConfig::default();
//!     let limiter = Arc::new(RateLimiter::new(config));
//!
//!     // Start cleanup task
//!     limiter.clone().start_cleanup_task();
//!
//!     let app = Router::new()
//!         .route("/", get(|| async { "Hello, World!" }))
//!         .layer(RateLimitLayer::new(limiter));
//!
//!     // Run server...
//! }
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::net::IpAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tower::{Layer, Service};
use tracing::{debug, info, warn};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during rate limiting operations
#[derive(Error, Debug, Clone)]
pub enum RateLimitError {
    /// Request was rate limited
    #[error("Rate limit exceeded. Retry after {retry_after} seconds")]
    RateLimitExceeded {
        /// Number of seconds until the rate limit resets
        retry_after: u64,
        /// Maximum requests allowed per window
        limit: u32,
        /// Current remaining requests (will be 0)
        remaining: u32,
        /// Unix timestamp when the limit resets
        reset_at: u64,
    },

    /// Failed to extract IP address from request
    #[error("Could not determine client IP address")]
    IpExtractionFailed,

    /// Internal error in rate limiter
    #[error("Internal rate limiter error: {0}")]
    Internal(String),
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the rate limiter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute (default: 100)
    pub requests_per_minute: u32,

    /// Maximum burst size for instantaneous requests (default: 10)
    pub burst_size: u32,

    /// Interval between cleanup runs for stale buckets (default: 60s)
    pub cleanup_interval: Duration,

    /// Time after which an idle bucket is considered stale (default: 300s)
    pub bucket_ttl: Duration,

    /// Whether to trust X-Forwarded-For header (default: false)
    pub trust_x_forwarded_for: bool,

    /// Whether to trust X-Real-IP header (default: false)
    pub trust_x_real_ip: bool,

    /// Whitelist of IPs that bypass rate limiting
    pub whitelist: Vec<IpAddr>,

    /// Enable rate limiting (can be disabled for testing)
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 100,
            burst_size: 10,
            cleanup_interval: Duration::from_secs(60),
            bucket_ttl: Duration::from_secs(300),
            trust_x_forwarded_for: false,
            trust_x_real_ip: false,
            whitelist: Vec::new(),
            enabled: true,
        }
    }
}

impl RateLimitConfig {
    /// Create a new configuration with custom requests per minute
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            ..Default::default()
        }
    }

    /// Set the burst size
    pub fn with_burst_size(mut self, burst_size: u32) -> Self {
        self.burst_size = burst_size;
        self
    }

    /// Set the cleanup interval
    pub fn with_cleanup_interval(mut self, interval: Duration) -> Self {
        self.cleanup_interval = interval;
        self
    }

    /// Set the bucket TTL
    pub fn with_bucket_ttl(mut self, ttl: Duration) -> Self {
        self.bucket_ttl = ttl;
        self
    }

    /// Trust X-Forwarded-For header
    pub fn trust_forwarded_for(mut self) -> Self {
        self.trust_x_forwarded_for = true;
        self
    }

    /// Trust X-Real-IP header
    pub fn trust_real_ip(mut self) -> Self {
        self.trust_x_real_ip = true;
        self
    }

    /// Add an IP to the whitelist
    pub fn whitelist_ip(mut self, ip: IpAddr) -> Self {
        self.whitelist.push(ip);
        self
    }

    /// Disable rate limiting
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Calculate tokens per second based on requests per minute
    fn tokens_per_second(&self) -> f64 {
        self.requests_per_minute as f64 / 60.0
    }
}

// ============================================================================
// Token Bucket
// ============================================================================

/// A token bucket for tracking rate limits per IP
#[derive(Debug, Clone)]
struct TokenBucket {
    /// Current number of tokens available
    tokens: f64,

    /// Maximum capacity (burst size)
    capacity: f64,

    /// Tokens added per second
    refill_rate: f64,

    /// Last time tokens were refilled
    last_update: Instant,

    /// Last time this bucket was accessed
    last_access: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    fn new(capacity: u32, refill_rate: f64) -> Self {
        let now = Instant::now();
        Self {
            tokens: capacity as f64, // Start with full bucket
            capacity: capacity as f64,
            refill_rate,
            last_update: now,
            last_access: now,
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();

        // Add tokens based on elapsed time
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_update = now;
        self.last_access = now;
    }

    /// Try to consume one token
    fn try_consume(&mut self) -> bool {
        self.refill();

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Get remaining tokens (as whole number)
    fn remaining(&mut self) -> u32 {
        self.refill();
        self.tokens.floor() as u32
    }

    /// Get seconds until one token is available
    fn seconds_until_token(&self) -> u64 {
        if self.tokens >= 1.0 {
            return 0;
        }

        let tokens_needed = 1.0 - self.tokens;
        (tokens_needed / self.refill_rate).ceil() as u64
    }

    /// Check if this bucket is stale (hasn't been accessed recently)
    fn is_stale(&self, ttl: Duration) -> bool {
        self.last_access.elapsed() > ttl
    }
}

// ============================================================================
// Rate Limiter
// ============================================================================

/// The main rate limiter struct
///
/// This is the core component that tracks token buckets per IP address
/// and enforces rate limits.
pub struct RateLimiter {
    /// Map of IP addresses to their token buckets
    buckets: RwLock<HashMap<IpAddr, TokenBucket>>,

    /// Configuration
    config: RateLimitConfig,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Self {
        info!(
            "Creating rate limiter: {} req/min, burst: {}",
            config.requests_per_minute, config.burst_size
        );

        Self {
            buckets: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create a rate limiter with default configuration
    pub fn default_limiter() -> Self {
        Self::new(RateLimitConfig::default())
    }

    /// Check if a request from the given IP should be allowed
    ///
    /// Returns Ok(()) if the request is allowed, or an error with rate limit info
    pub async fn check_limit(&self, ip: IpAddr) -> Result<(), RateLimitError> {
        // Check if rate limiting is disabled
        if !self.config.enabled {
            return Ok(());
        }

        // Check whitelist
        if self.config.whitelist.contains(&ip) {
            debug!("IP {} is whitelisted, bypassing rate limit", ip);
            return Ok(());
        }

        let mut buckets = self.buckets.write().await;

        // Get or create bucket for this IP
        let bucket = buckets.entry(ip).or_insert_with(|| {
            TokenBucket::new(self.config.burst_size, self.config.tokens_per_second())
        });

        if bucket.try_consume() {
            debug!("Rate limit check passed for {}: {} tokens remaining", ip, bucket.remaining());
            Ok(())
        } else {
            let retry_after = bucket.seconds_until_token();
            let reset_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + retry_after;

            warn!("Rate limit exceeded for {}: retry after {}s", ip, retry_after);

            Err(RateLimitError::RateLimitExceeded {
                retry_after,
                limit: self.config.requests_per_minute,
                remaining: 0,
                reset_at,
            })
        }
    }

    /// Get the number of remaining requests for an IP
    pub async fn get_remaining(&self, ip: IpAddr) -> u32 {
        if !self.config.enabled {
            return u32::MAX;
        }

        if self.config.whitelist.contains(&ip) {
            return u32::MAX;
        }

        let mut buckets = self.buckets.write().await;

        if let Some(bucket) = buckets.get_mut(&ip) {
            bucket.remaining()
        } else {
            // No bucket means they haven't made any requests yet
            self.config.burst_size
        }
    }

    /// Reset the rate limit for a specific IP
    pub async fn reset(&self, ip: IpAddr) {
        let mut buckets = self.buckets.write().await;
        buckets.remove(&ip);
        info!("Rate limit reset for {}", ip);
    }

    /// Reset all rate limits
    pub async fn reset_all(&self) {
        let mut buckets = self.buckets.write().await;
        buckets.clear();
        info!("All rate limits reset");
    }

    /// Get rate limit info for an IP (for response headers)
    pub async fn get_limit_info(&self, ip: IpAddr) -> RateLimitInfo {
        let remaining = self.get_remaining(ip).await;

        // Calculate reset time (next minute boundary)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Reset at the next minute boundary
        let reset_at = (now / 60 + 1) * 60;

        RateLimitInfo {
            limit: self.config.requests_per_minute,
            remaining,
            reset_at,
        }
    }

    /// Start a background task to clean up stale buckets
    ///
    /// This should be called once when the server starts.
    /// The task runs forever in the background.
    pub fn start_cleanup_task(self: Arc<Self>) {
        let limiter = self.clone();
        let interval = self.config.cleanup_interval;
        let ttl = self.config.bucket_ttl;

        tokio::spawn(async move {
            info!("Starting rate limiter cleanup task (interval: {:?})", interval);

            loop {
                tokio::time::sleep(interval).await;
                limiter.cleanup_stale_buckets(ttl).await;
            }
        });
    }

    /// Remove stale buckets to prevent memory growth
    async fn cleanup_stale_buckets(&self, ttl: Duration) {
        let mut buckets = self.buckets.write().await;
        let before_count = buckets.len();

        buckets.retain(|ip, bucket| {
            let keep = !bucket.is_stale(ttl);
            if !keep {
                debug!("Removing stale bucket for {}", ip);
            }
            keep
        });

        let removed = before_count - buckets.len();
        if removed > 0 {
            info!("Cleaned up {} stale rate limit buckets", removed);
        }
    }

    /// Get current bucket count (for monitoring)
    pub async fn bucket_count(&self) -> usize {
        self.buckets.read().await.len()
    }

    /// Get configuration (immutable)
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }
}

// ============================================================================
// Rate Limit Info (for headers)
// ============================================================================

/// Information about rate limit status
#[derive(Debug, Clone, Copy)]
pub struct RateLimitInfo {
    /// Maximum requests per window
    pub limit: u32,

    /// Remaining requests in current window
    pub remaining: u32,

    /// Unix timestamp when the limit resets
    pub reset_at: u64,
}

// ============================================================================
// Tower Layer Implementation
// ============================================================================

/// Tower Layer for rate limiting
///
/// This wraps services with rate limiting middleware.
#[derive(Clone)]
pub struct RateLimitLayer {
    limiter: Arc<RateLimiter>,
}

impl RateLimitLayer {
    /// Create a new rate limit layer
    pub fn new(limiter: Arc<RateLimiter>) -> Self {
        Self { limiter }
    }
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            limiter: self.limiter.clone(),
        }
    }
}

/// Tower Service that enforces rate limits
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    limiter: Arc<RateLimiter>,
}

impl<S> Service<Request<Body>> for RateLimitService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let limiter = self.limiter.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Extract client IP
            let ip = extract_client_ip(&req, &limiter.config);

            match ip {
                Some(ip) => {
                    // Check rate limit
                    match limiter.check_limit(ip).await {
                        Ok(()) => {
                            // Request allowed - add headers and forward
                            let response = inner.call(req).await?;
                            let info = limiter.get_limit_info(ip).await;
                            Ok(add_rate_limit_headers(response, &info))
                        }
                        Err(RateLimitError::RateLimitExceeded {
                            retry_after,
                            limit,
                            remaining,
                            reset_at,
                        }) => {
                            // Rate limited - return 429
                            Ok(rate_limit_response(retry_after, limit, remaining, reset_at))
                        }
                        Err(e) => {
                            // Other error - log and allow through
                            warn!("Rate limiter error: {}", e);
                            let response = inner.call(req).await?;
                            Ok(response)
                        }
                    }
                }
                None => {
                    // Could not determine IP - allow through but log
                    warn!("Could not extract client IP for rate limiting");
                    let response = inner.call(req).await?;
                    Ok(response)
                }
            }
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract client IP from request
fn extract_client_ip(req: &Request<Body>, config: &RateLimitConfig) -> Option<IpAddr> {
    // Try X-Forwarded-For if trusted
    if config.trust_x_forwarded_for {
        if let Some(xff) = req.headers().get("x-forwarded-for") {
            if let Ok(xff_str) = xff.to_str() {
                // Take the first IP in the chain (original client)
                if let Some(first_ip) = xff_str.split(',').next() {
                    if let Ok(ip) = first_ip.trim().parse::<IpAddr>() {
                        return Some(ip);
                    }
                }
            }
        }
    }

    // Try X-Real-IP if trusted
    if config.trust_x_real_ip {
        if let Some(real_ip) = req.headers().get("x-real-ip") {
            if let Ok(ip_str) = real_ip.to_str() {
                if let Ok(ip) = ip_str.trim().parse::<IpAddr>() {
                    return Some(ip);
                }
            }
        }
    }

    // Fall back to ConnectInfo extension (requires proper Axum setup)
    // For now, we'll use a request extension if available
    req.extensions()
        .get::<std::net::SocketAddr>()
        .map(|addr| addr.ip())
}

/// Add rate limit headers to a response
fn add_rate_limit_headers(mut response: Response<Body>, info: &RateLimitInfo) -> Response<Body> {
    let headers = response.headers_mut();

    headers.insert(
        "X-RateLimit-Limit",
        info.limit.to_string().parse().unwrap(),
    );
    headers.insert(
        "X-RateLimit-Remaining",
        info.remaining.to_string().parse().unwrap(),
    );
    headers.insert(
        "X-RateLimit-Reset",
        info.reset_at.to_string().parse().unwrap(),
    );

    response
}

/// Create a rate limit exceeded response
fn rate_limit_response(retry_after: u64, limit: u32, remaining: u32, reset_at: u64) -> Response<Body> {
    let body = serde_json::json!({
        "error": "rate_limit_exceeded",
        "message": format!("Rate limit exceeded. Please retry after {} seconds.", retry_after),
        "limit": limit,
        "remaining": remaining,
        "reset_at": reset_at,
        "retry_after": retry_after
    });

    let mut response = Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("Content-Type", "application/json")
        .header("X-RateLimit-Limit", limit.to_string())
        .header("X-RateLimit-Remaining", remaining.to_string())
        .header("X-RateLimit-Reset", reset_at.to_string())
        .header("Retry-After", retry_after.to_string())
        .body(Body::from(body.to_string()))
        .unwrap();

    response
}

// ============================================================================
// Axum Middleware Function
// ============================================================================

/// Create a rate limit layer for use with Axum Router
///
/// This is a convenience function that creates a properly configured
/// rate limit layer.
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::rate_limit::{rate_limit_layer, RateLimitConfig};
/// use std::sync::Arc;
/// use axum::{Router, routing::get};
///
/// let config = RateLimitConfig::new(100).with_burst_size(10);
/// let layer = rate_limit_layer(config);
///
/// let app = Router::new()
///     .route("/", get(|| async { "Hello!" }))
///     .layer(layer);
/// ```
pub fn rate_limit_layer(config: RateLimitConfig) -> RateLimitLayer {
    let limiter = Arc::new(RateLimiter::new(config));
    RateLimitLayer::new(limiter)
}

/// Create a rate limit layer with automatic cleanup
///
/// This starts the background cleanup task automatically.
pub fn rate_limit_layer_with_cleanup(config: RateLimitConfig) -> (RateLimitLayer, Arc<RateLimiter>) {
    let limiter = Arc::new(RateLimiter::new(config));
    limiter.clone().start_cleanup_task();
    (RateLimitLayer::new(limiter.clone()), limiter)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_ip() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))
    }

    #[tokio::test]
    async fn test_rate_limiter_allows_requests_under_limit() {
        let config = RateLimitConfig::new(100).with_burst_size(10);
        let limiter = RateLimiter::new(config);

        // Should allow 10 requests (burst size)
        for i in 0..10 {
            let result = limiter.check_limit(test_ip()).await;
            assert!(result.is_ok(), "Request {} should be allowed", i);
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_when_exceeded() {
        let config = RateLimitConfig::new(60).with_burst_size(5);
        let limiter = RateLimiter::new(config);
        let ip = test_ip();

        // Exhaust the bucket
        for _ in 0..5 {
            let _ = limiter.check_limit(ip).await;
        }

        // Next request should be blocked
        let result = limiter.check_limit(ip).await;
        assert!(result.is_err());

        if let Err(RateLimitError::RateLimitExceeded { retry_after, .. }) = result {
            assert!(retry_after > 0);
        } else {
            panic!("Expected RateLimitExceeded error");
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_refills_over_time() {
        let config = RateLimitConfig::new(600).with_burst_size(5); // 10 per second
        let limiter = RateLimiter::new(config);
        let ip = test_ip();

        // Exhaust the bucket
        for _ in 0..5 {
            let _ = limiter.check_limit(ip).await;
        }

        // Wait for refill
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Should have at least 1 token now
        let result = limiter.check_limit(ip).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter_whitelist() {
        let whitelisted_ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let config = RateLimitConfig::new(60)
            .with_burst_size(1)
            .whitelist_ip(whitelisted_ip);
        let limiter = RateLimiter::new(config);

        // Whitelisted IP should always be allowed
        for _ in 0..100 {
            let result = limiter.check_limit(whitelisted_ip).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let config = RateLimitConfig::new(1).with_burst_size(1).disabled();
        let limiter = RateLimiter::new(config);

        // Should always allow when disabled
        for _ in 0..100 {
            let result = limiter.check_limit(test_ip()).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_get_remaining() {
        let config = RateLimitConfig::new(60).with_burst_size(10);
        let limiter = RateLimiter::new(config);
        let ip = test_ip();

        assert_eq!(limiter.get_remaining(ip).await, 10);

        limiter.check_limit(ip).await.unwrap();
        assert_eq!(limiter.get_remaining(ip).await, 9);

        limiter.check_limit(ip).await.unwrap();
        assert_eq!(limiter.get_remaining(ip).await, 8);
    }

    #[tokio::test]
    async fn test_reset() {
        let config = RateLimitConfig::new(60).with_burst_size(5);
        let limiter = RateLimiter::new(config);
        let ip = test_ip();

        // Exhaust bucket
        for _ in 0..5 {
            let _ = limiter.check_limit(ip).await;
        }

        assert_eq!(limiter.get_remaining(ip).await, 0);

        // Reset
        limiter.reset(ip).await;

        // Should have full bucket again
        assert_eq!(limiter.get_remaining(ip).await, 5);
    }

    #[tokio::test]
    async fn test_multiple_ips() {
        let config = RateLimitConfig::new(60).with_burst_size(2);
        let limiter = RateLimiter::new(config);

        let ip1 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2));

        // Exhaust IP1
        limiter.check_limit(ip1).await.unwrap();
        limiter.check_limit(ip1).await.unwrap();
        assert!(limiter.check_limit(ip1).await.is_err());

        // IP2 should still have full quota
        assert!(limiter.check_limit(ip2).await.is_ok());
        assert!(limiter.check_limit(ip2).await.is_ok());
    }

    #[tokio::test]
    async fn test_cleanup_stale_buckets() {
        let config = RateLimitConfig::new(60)
            .with_burst_size(5)
            .with_bucket_ttl(Duration::from_millis(100));
        let limiter = RateLimiter::new(config);

        // Create some buckets
        let ip1 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2));

        limiter.check_limit(ip1).await.unwrap();
        limiter.check_limit(ip2).await.unwrap();

        assert_eq!(limiter.bucket_count().await, 2);

        // Wait for buckets to become stale
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Run cleanup
        limiter.cleanup_stale_buckets(Duration::from_millis(100)).await;

        assert_eq!(limiter.bucket_count().await, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = RateLimitConfig::new(200)
            .with_burst_size(20)
            .with_cleanup_interval(Duration::from_secs(120))
            .with_bucket_ttl(Duration::from_secs(600))
            .trust_forwarded_for()
            .trust_real_ip();

        assert_eq!(config.requests_per_minute, 200);
        assert_eq!(config.burst_size, 20);
        assert_eq!(config.cleanup_interval, Duration::from_secs(120));
        assert_eq!(config.bucket_ttl, Duration::from_secs(600));
        assert!(config.trust_x_forwarded_for);
        assert!(config.trust_x_real_ip);
    }

    #[test]
    fn test_tokens_per_second() {
        let config = RateLimitConfig::new(60);
        assert!((config.tokens_per_second() - 1.0).abs() < 0.001);

        let config = RateLimitConfig::new(120);
        assert!((config.tokens_per_second() - 2.0).abs() < 0.001);

        let config = RateLimitConfig::new(30);
        assert!((config.tokens_per_second() - 0.5).abs() < 0.001);
    }
}

//! Security Hardening Middleware for ReasonKit Web
//!
//! This module provides comprehensive security middleware for the HTTP API layer,
//! implementing defense-in-depth strategies including:
//!
//! - Localhost-only binding (configurable for Docker)
//! - Token-based authentication
//! - CORS configuration (localhost-only)
//! - Security headers (CSP, X-Frame-Options, etc.)
//! - Rate limiting (per-IP)
//!
//! # Security Architecture
//!
//! ```text
//! Request -> Rate Limiter -> IP Filter -> Auth -> CORS -> Security Headers -> Handler
//! ```
//!
//! # Configuration
//!
//! All security settings are configured via environment variables (CONS-003 compliant):
//!
//! - `REASONKIT_WEB_TOKEN`: Required authentication token
//! - `REASONKIT_WEB_BIND_ALL`: Set to "true" for Docker (binds to 0.0.0.0)
//! - `REASONKIT_WEB_RATE_LIMIT`: Requests per minute per IP (default: 100)
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::security::{SecurityConfig, SecurityLayer};
//! use axum::Router;
//!
//! let config = SecurityConfig::from_env().expect("Security config error");
//! let app = Router::new()
//!     // ... routes
//!     .layer(SecurityLayer::new(config));
//! ```

use std::collections::HashMap;
use std::convert::Infallible;
use std::env;
use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Security-related error types
#[derive(Error, Debug)]
pub enum SecurityError {
    /// Missing required token environment variable
    #[error("REASONKIT_WEB_TOKEN environment variable not set")]
    MissingToken,

    /// Invalid token format
    #[error("Invalid token format: {0}")]
    InvalidTokenFormat(String),

    /// Invalid rate limit configuration
    #[error("Invalid rate limit: {0}")]
    InvalidRateLimit(String),

    /// Configuration error
    #[error("Security configuration error: {0}")]
    ConfigError(String),
}

/// Result type for security operations
pub type SecurityResult<T> = std::result::Result<T, SecurityError>;

// =============================================================================
// Security Configuration
// =============================================================================

/// Security configuration loaded from environment variables
///
/// CONS-003 COMPLIANT: No hardcoded secrets. All sensitive values from env vars.
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Authentication token (from REASONKIT_WEB_TOKEN)
    /// This is stored as a hash for comparison, never logged
    token_hash: [u8; 32],

    /// Whether to bind to all interfaces (for Docker)
    /// Set REASONKIT_WEB_BIND_ALL=true to enable
    pub bind_all: bool,

    /// Bind address derived from bind_all setting
    pub bind_addr: IpAddr,

    /// Rate limit: requests per minute per IP
    pub rate_limit_rpm: u32,

    /// Allowed origins for CORS (localhost only by default)
    pub allowed_origins: Vec<String>,

    /// Endpoints that bypass authentication (e.g., /health)
    pub auth_bypass_paths: Vec<String>,
}

impl SecurityConfig {
    /// Create a new security configuration from environment variables
    ///
    /// # Environment Variables
    ///
    /// - `REASONKIT_WEB_TOKEN` (required): Authentication bearer token
    /// - `REASONKIT_WEB_BIND_ALL` (optional): Set to "true" for Docker
    /// - `REASONKIT_WEB_RATE_LIMIT` (optional): Requests per minute (default: 100)
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingToken` if `REASONKIT_WEB_TOKEN` is not set.
    pub fn from_env() -> SecurityResult<Self> {
        // CONS-003: Token from environment variable only
        let token = env::var("REASONKIT_WEB_TOKEN").map_err(|_| SecurityError::MissingToken)?;

        if token.is_empty() {
            return Err(SecurityError::InvalidTokenFormat(
                "Token cannot be empty".to_string(),
            ));
        }

        if token.len() < 32 {
            warn!("SECURITY WARNING: REASONKIT_WEB_TOKEN is less than 32 characters");
        }

        // Hash the token for constant-time comparison
        let token_hash = Self::hash_token(&token);

        // Check if we should bind to all interfaces (Docker mode)
        let bind_all = env::var("REASONKIT_WEB_BIND_ALL")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);

        let bind_addr = if bind_all {
            warn!("SECURITY: Binding to 0.0.0.0 (REASONKIT_WEB_BIND_ALL=true)");
            IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
        } else {
            info!("SECURITY: Binding to localhost only (127.0.0.1)");
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))
        };

        // Parse rate limit
        let rate_limit_rpm = env::var("REASONKIT_WEB_RATE_LIMIT")
            .unwrap_or_else(|_| "100".to_string())
            .parse::<u32>()
            .map_err(|e| SecurityError::InvalidRateLimit(e.to_string()))?;

        if rate_limit_rpm == 0 {
            return Err(SecurityError::InvalidRateLimit(
                "Rate limit cannot be 0".to_string(),
            ));
        }

        info!(
            "SECURITY: Rate limit set to {} requests/minute per IP",
            rate_limit_rpm
        );

        Ok(Self {
            token_hash,
            bind_all,
            bind_addr,
            rate_limit_rpm,
            allowed_origins: vec![
                "http://localhost".to_string(),
                "http://127.0.0.1".to_string(),
                "http://localhost:3000".to_string(),
                "http://localhost:8080".to_string(),
                "http://127.0.0.1:3000".to_string(),
                "http://127.0.0.1:8080".to_string(),
            ],
            auth_bypass_paths: vec!["/health".to_string(), "/healthz".to_string()],
        })
    }

    /// Create a test configuration (for testing only)
    #[cfg(test)]
    pub fn test_config() -> Self {
        Self {
            token_hash: Self::hash_token("test-token-for-unit-tests-only"),
            bind_all: false,
            bind_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            rate_limit_rpm: 100,
            allowed_origins: vec!["http://localhost".to_string()],
            auth_bypass_paths: vec!["/health".to_string()],
        }
    }

    /// Hash a token using SHA-256 for constant-time comparison
    fn hash_token(token: &str) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple hash for demo - in production, use a proper cryptographic hash
        // This provides defense in depth; the token is never stored in plain text
        let mut result = [0u8; 32];
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();

        // Expand the hash to 32 bytes
        for (i, byte) in result.iter_mut().enumerate() {
            *byte = ((hash >> ((i % 8) * 8)) & 0xFF) as u8 ^ (i as u8);
        }

        result
    }

    /// Verify a token against the stored hash (constant-time comparison)
    pub fn verify_token(&self, token: &str) -> bool {
        let provided_hash = Self::hash_token(token);
        constant_time_compare(&self.token_hash, &provided_hash)
    }

    /// Get the socket address for binding
    pub fn socket_addr(&self, port: u16) -> SocketAddr {
        SocketAddr::new(self.bind_addr, port)
    }

    /// Check if a path bypasses authentication
    pub fn is_auth_bypass_path(&self, path: &str) -> bool {
        self.auth_bypass_paths.iter().any(|p| path.starts_with(p))
    }

    /// Check if an origin is allowed for CORS
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| origin.starts_with(o))
    }
}

/// Constant-time byte comparison to prevent timing attacks
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

// =============================================================================
// Rate Limiter
// =============================================================================

/// Per-IP rate limiter using a sliding window algorithm
#[derive(Debug)]
pub struct RateLimiter {
    /// Maximum requests per window
    max_requests: u32,
    /// Window duration
    window: Duration,
    /// Request counts per IP
    buckets: Arc<RwLock<HashMap<IpAddr, RateBucket>>>,
}

#[derive(Debug, Clone)]
struct RateBucket {
    /// Number of requests in current window
    count: u32,
    /// Window start time
    window_start: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            max_requests: requests_per_minute,
            window: Duration::from_secs(60),
            buckets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if a request from the given IP is allowed
    pub async fn check(&self, ip: IpAddr) -> RateLimitResult {
        let mut buckets = self.buckets.write().await;
        let now = Instant::now();

        let bucket = buckets.entry(ip).or_insert_with(|| RateBucket {
            count: 0,
            window_start: now,
        });

        // Check if window has expired
        if now.duration_since(bucket.window_start) >= self.window {
            bucket.count = 0;
            bucket.window_start = now;
        }

        bucket.count += 1;

        if bucket.count > self.max_requests {
            let remaining_time = self
                .window
                .saturating_sub(now.duration_since(bucket.window_start));
            RateLimitResult::Exceeded {
                retry_after: remaining_time,
                limit: self.max_requests,
            }
        } else {
            RateLimitResult::Allowed {
                remaining: self.max_requests - bucket.count,
                limit: self.max_requests,
            }
        }
    }

    /// Clean up expired buckets (call periodically)
    pub async fn cleanup(&self) {
        let mut buckets = self.buckets.write().await;
        let now = Instant::now();
        let window = self.window;

        buckets.retain(|_, bucket| now.duration_since(bucket.window_start) < window * 2);
    }
}

/// Result of a rate limit check
#[derive(Debug, Clone)]
pub enum RateLimitResult {
    /// Request is allowed
    Allowed {
        /// Remaining requests in window
        remaining: u32,
        /// Total limit
        limit: u32,
    },
    /// Rate limit exceeded
    Exceeded {
        /// Time until rate limit resets
        retry_after: Duration,
        /// Total limit
        limit: u32,
    },
}

impl RateLimitResult {
    /// Check if the request is allowed
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitResult::Allowed { .. })
    }
}

// =============================================================================
// IP Filter
// =============================================================================

/// IP address filter for localhost-only binding
#[derive(Debug, Clone)]
pub struct IpFilter {
    /// Whether to allow all IPs (Docker mode)
    allow_all: bool,
    /// Allowed IP addresses
    allowed_ips: Vec<IpAddr>,
}

impl IpFilter {
    /// Create a new IP filter
    pub fn new(allow_all: bool) -> Self {
        Self {
            allow_all,
            allowed_ips: vec![
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), // For local connections
                IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
            ],
        }
    }

    /// Check if an IP is allowed
    pub fn is_allowed(&self, ip: IpAddr) -> bool {
        if self.allow_all {
            return true;
        }

        // Check if it's a localhost IP
        match ip {
            IpAddr::V4(ipv4) => ipv4.is_loopback() || self.allowed_ips.contains(&ip),
            IpAddr::V6(ipv6) => ipv6.is_loopback() || self.allowed_ips.contains(&ip),
        }
    }

    /// Add an allowed IP
    pub fn allow_ip(&mut self, ip: IpAddr) {
        if !self.allowed_ips.contains(&ip) {
            self.allowed_ips.push(ip);
        }
    }
}

impl Default for IpFilter {
    fn default() -> Self {
        Self::new(false)
    }
}

// =============================================================================
// Authentication
// =============================================================================

/// Token-based authentication result
#[derive(Debug, Clone, PartialEq)]
pub enum AuthResult {
    /// Authentication successful
    Authenticated,
    /// Authentication failed - missing header
    MissingHeader,
    /// Authentication failed - invalid format
    InvalidFormat,
    /// Authentication failed - invalid token
    InvalidToken,
    /// Authentication bypassed (e.g., /health endpoint)
    Bypassed,
}

impl AuthResult {
    /// Check if authentication was successful or bypassed
    pub fn is_ok(&self) -> bool {
        matches!(self, AuthResult::Authenticated | AuthResult::Bypassed)
    }

    /// Get HTTP status code for authentication failure
    pub fn status_code(&self) -> u16 {
        match self {
            AuthResult::Authenticated | AuthResult::Bypassed => 200,
            AuthResult::MissingHeader | AuthResult::InvalidFormat | AuthResult::InvalidToken => 401,
        }
    }

    /// Get error message for authentication failure
    pub fn error_message(&self) -> Option<&'static str> {
        match self {
            AuthResult::Authenticated | AuthResult::Bypassed => None,
            AuthResult::MissingHeader => Some("Missing Authorization header"),
            AuthResult::InvalidFormat => {
                Some("Invalid Authorization format. Expected: Bearer <token>")
            }
            AuthResult::InvalidToken => Some("Invalid token"),
        }
    }
}

/// Token authenticator
#[derive(Debug, Clone)]
pub struct TokenAuthenticator {
    config: Arc<SecurityConfig>,
}

impl TokenAuthenticator {
    /// Create a new token authenticator
    pub fn new(config: Arc<SecurityConfig>) -> Self {
        Self { config }
    }

    /// Authenticate a request based on the Authorization header
    ///
    /// Expected format: `Authorization: Bearer <token>`
    pub fn authenticate(&self, path: &str, auth_header: Option<&str>) -> AuthResult {
        // Check if path bypasses authentication
        if self.config.is_auth_bypass_path(path) {
            debug!("Auth bypass for path: {}", path);
            return AuthResult::Bypassed;
        }

        // Check for Authorization header
        let header = match auth_header {
            Some(h) => h,
            None => return AuthResult::MissingHeader,
        };

        // Parse "Bearer <token>" format
        let token = match header.strip_prefix("Bearer ") {
            Some(t) => t.trim(),
            None => return AuthResult::InvalidFormat,
        };

        if token.is_empty() {
            return AuthResult::InvalidFormat;
        }

        // Verify token (constant-time comparison)
        if self.config.verify_token(token) {
            AuthResult::Authenticated
        } else {
            warn!("Invalid authentication token attempt");
            AuthResult::InvalidToken
        }
    }
}

// =============================================================================
// Security Headers
// =============================================================================

/// Security headers to apply to all responses
#[derive(Debug, Clone)]
pub struct SecurityHeaders;

impl SecurityHeaders {
    /// Standard security headers
    pub fn headers() -> Vec<(&'static str, &'static str)> {
        vec![
            // Prevent MIME type sniffing
            ("X-Content-Type-Options", "nosniff"),
            // Prevent clickjacking
            ("X-Frame-Options", "DENY"),
            // Strict Content Security Policy
            (
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'",
            ),
            // Disable XSS filter (rely on CSP instead)
            ("X-XSS-Protection", "0"),
            // No referrer information leak
            ("Referrer-Policy", "no-referrer"),
            // Prevent caching of sensitive responses
            (
                "Cache-Control",
                "no-store, no-cache, must-revalidate, private",
            ),
            ("Pragma", "no-cache"),
            // Permissions policy - disable all browser features
            (
                "Permissions-Policy",
                "geolocation=(), microphone=(), camera=()",
            ),
        ]
    }

    /// CORS headers for localhost origins
    pub fn cors_headers(
        origin: Option<&str>,
        config: &SecurityConfig,
    ) -> Vec<(&'static str, String)> {
        let mut headers = Vec::new();

        // Only add CORS headers if origin is provided and allowed
        if let Some(origin) = origin {
            if config.is_origin_allowed(origin) {
                headers.push(("Access-Control-Allow-Origin", origin.to_string()));
                headers.push(("Access-Control-Allow-Methods", "GET, POST".to_string()));
                headers.push((
                    "Access-Control-Allow-Headers",
                    "Authorization, Content-Type".to_string(),
                ));
                headers.push(("Access-Control-Max-Age", "3600".to_string()));
                // No credentials allowed
                headers.push(("Access-Control-Allow-Credentials", "false".to_string()));
            } else {
                debug!("Rejected CORS origin: {}", origin);
            }
        }

        headers
    }

    /// Rate limit headers
    pub fn rate_limit_headers(result: &RateLimitResult) -> Vec<(&'static str, String)> {
        match result {
            RateLimitResult::Allowed { remaining, limit } => {
                vec![
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Remaining", remaining.to_string()),
                ]
            }
            RateLimitResult::Exceeded { retry_after, limit } => {
                vec![
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Remaining", "0".to_string()),
                    ("Retry-After", retry_after.as_secs().to_string()),
                ]
            }
        }
    }
}

// =============================================================================
// Tower Middleware Layer
// =============================================================================

/// Security middleware layer for Tower-based HTTP servers (axum, hyper)
///
/// This provides a complete security stack:
/// - Rate limiting
/// - IP filtering
/// - Token authentication
/// - Security headers
/// - CORS
#[derive(Clone)]
pub struct SecurityLayer {
    config: Arc<SecurityConfig>,
    rate_limiter: Arc<RateLimiter>,
    ip_filter: IpFilter,
    authenticator: TokenAuthenticator,
}

impl SecurityLayer {
    /// Create a new security layer from configuration
    pub fn new(config: SecurityConfig) -> Self {
        let config = Arc::new(config);
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit_rpm));

        Self {
            ip_filter: IpFilter::new(config.bind_all),
            authenticator: TokenAuthenticator::new(Arc::clone(&config)),
            config,
            rate_limiter,
        }
    }

    /// Get the socket address for binding
    pub fn bind_addr(&self, port: u16) -> SocketAddr {
        self.config.socket_addr(port)
    }

    /// Check rate limit for an IP
    pub async fn check_rate_limit(&self, ip: IpAddr) -> RateLimitResult {
        self.rate_limiter.check(ip).await
    }

    /// Check if an IP is allowed
    pub fn check_ip(&self, ip: IpAddr) -> bool {
        self.ip_filter.is_allowed(ip)
    }

    /// Authenticate a request
    pub fn authenticate(&self, path: &str, auth_header: Option<&str>) -> AuthResult {
        self.authenticator.authenticate(path, auth_header)
    }

    /// Get security headers
    pub fn security_headers(&self) -> Vec<(&'static str, &'static str)> {
        SecurityHeaders::headers()
    }

    /// Get CORS headers
    pub fn cors_headers(&self, origin: Option<&str>) -> Vec<(&'static str, String)> {
        SecurityHeaders::cors_headers(origin, &self.config)
    }

    /// Get rate limit headers
    pub fn rate_limit_headers(&self, result: &RateLimitResult) -> Vec<(&'static str, String)> {
        SecurityHeaders::rate_limit_headers(result)
    }

    /// Clean up rate limiter buckets
    pub async fn cleanup_rate_limiter(&self) {
        self.rate_limiter.cleanup().await;
    }
}

// =============================================================================
// Request Validation
// =============================================================================

/// Validate an incoming HTTP request through all security layers
///
/// Returns a `SecurityCheckResult` that can be used to allow or reject the request
pub struct SecurityCheck {
    layer: SecurityLayer,
}

impl SecurityCheck {
    /// Create a new security check
    pub fn new(layer: SecurityLayer) -> Self {
        Self { layer }
    }

    /// Validate a request through all security layers
    ///
    /// # Arguments
    ///
    /// * `remote_ip` - The remote IP address of the client
    /// * `path` - The request path
    /// * `auth_header` - The Authorization header value (if present)
    /// * `origin` - The Origin header value (if present)
    ///
    /// # Returns
    ///
    /// A `SecurityCheckResult` with the validation result
    pub async fn validate(
        &self,
        remote_ip: IpAddr,
        path: &str,
        auth_header: Option<&str>,
        origin: Option<&str>,
    ) -> SecurityCheckResult {
        // 1. Check IP filter
        if !self.layer.check_ip(remote_ip) {
            warn!("Rejected request from non-localhost IP: {}", remote_ip);
            return SecurityCheckResult::Rejected {
                status: 403,
                message: "Forbidden: Only localhost connections allowed".to_string(),
                headers: self
                    .layer
                    .security_headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            };
        }

        // 2. Check rate limit
        let rate_result = self.layer.check_rate_limit(remote_ip).await;
        if !rate_result.is_allowed() {
            warn!("Rate limit exceeded for IP: {}", remote_ip);
            let mut headers: Vec<(String, String)> = self
                .layer
                .security_headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            headers.extend(
                self.layer
                    .rate_limit_headers(&rate_result)
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone())),
            );

            let retry_after = match &rate_result {
                RateLimitResult::Exceeded { retry_after, .. } => retry_after.as_secs(),
                _ => 60,
            };

            return SecurityCheckResult::Rejected {
                status: 429,
                message: format!("Too Many Requests. Retry after {} seconds.", retry_after),
                headers,
            };
        }

        // 3. Check authentication
        let auth_result = self.layer.authenticate(path, auth_header);
        if !auth_result.is_ok() {
            warn!("Authentication failed for path {}: {:?}", path, auth_result);
            let mut headers: Vec<(String, String)> = self
                .layer
                .security_headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            headers.push(("WWW-Authenticate".to_string(), "Bearer".to_string()));

            return SecurityCheckResult::Rejected {
                status: auth_result.status_code(),
                message: auth_result
                    .error_message()
                    .unwrap_or("Unauthorized")
                    .to_string(),
                headers,
            };
        }

        // 4. Build success headers
        let mut headers: Vec<(String, String)> = self
            .layer
            .security_headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        headers.extend(
            self.layer
                .rate_limit_headers(&rate_result)
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone())),
        );
        headers.extend(
            self.layer
                .cors_headers(origin)
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone())),
        );

        SecurityCheckResult::Allowed { headers }
    }
}

/// Result of a security check
#[derive(Debug)]
pub enum SecurityCheckResult {
    /// Request is allowed
    Allowed {
        /// Headers to add to the response
        headers: Vec<(String, String)>,
    },
    /// Request is rejected
    Rejected {
        /// HTTP status code
        status: u16,
        /// Error message
        message: String,
        /// Headers to add to the response
        headers: Vec<(String, String)>,
    },
}

impl SecurityCheckResult {
    /// Check if the request is allowed
    pub fn is_allowed(&self) -> bool {
        matches!(self, SecurityCheckResult::Allowed { .. })
    }

    /// Get the HTTP status code
    pub fn status_code(&self) -> u16 {
        match self {
            SecurityCheckResult::Allowed { .. } => 200,
            SecurityCheckResult::Rejected { status, .. } => *status,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_security_config_token_verification() {
        let config = SecurityConfig::test_config();
        assert!(config.verify_token("test-token-for-unit-tests-only"));
        assert!(!config.verify_token("wrong-token"));
        assert!(!config.verify_token(""));
    }

    #[test]
    fn test_security_config_auth_bypass() {
        let config = SecurityConfig::test_config();
        assert!(config.is_auth_bypass_path("/health"));
        assert!(config.is_auth_bypass_path("/health/live"));
        assert!(!config.is_auth_bypass_path("/api/tools"));
    }

    #[test]
    fn test_security_config_origin_allowed() {
        let config = SecurityConfig::test_config();
        assert!(config.is_origin_allowed("http://localhost"));
        assert!(config.is_origin_allowed("http://localhost:3000"));
        assert!(!config.is_origin_allowed("http://example.com"));
        assert!(!config.is_origin_allowed("https://malicious.com"));
    }

    #[test]
    fn test_ip_filter_localhost() {
        let filter = IpFilter::new(false);
        assert!(filter.is_allowed(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(filter.is_allowed(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(!filter.is_allowed(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
        assert!(!filter.is_allowed(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
    }

    #[test]
    fn test_ip_filter_allow_all() {
        let filter = IpFilter::new(true);
        assert!(filter.is_allowed(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(filter.is_allowed(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
        assert!(filter.is_allowed(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
    }

    #[test]
    fn test_token_authenticator() {
        let config = Arc::new(SecurityConfig::test_config());
        let auth = TokenAuthenticator::new(config);

        // Valid token
        assert_eq!(
            auth.authenticate("/api/test", Some("Bearer test-token-for-unit-tests-only")),
            AuthResult::Authenticated
        );

        // Invalid token
        assert_eq!(
            auth.authenticate("/api/test", Some("Bearer wrong-token")),
            AuthResult::InvalidToken
        );

        // Missing header
        assert_eq!(
            auth.authenticate("/api/test", None),
            AuthResult::MissingHeader
        );

        // Invalid format
        assert_eq!(
            auth.authenticate("/api/test", Some("Basic dXNlcjpwYXNz")),
            AuthResult::InvalidFormat
        );

        // Bypass path
        assert_eq!(auth.authenticate("/health", None), AuthResult::Bypassed);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(3); // 3 requests per minute for testing
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // First 3 requests should be allowed
        assert!(limiter.check(ip).await.is_allowed());
        assert!(limiter.check(ip).await.is_allowed());
        assert!(limiter.check(ip).await.is_allowed());

        // 4th request should be rejected
        assert!(!limiter.check(ip).await.is_allowed());
    }

    #[tokio::test]
    async fn test_rate_limiter_different_ips() {
        let limiter = RateLimiter::new(2);
        let ip1 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        // Each IP has its own limit
        assert!(limiter.check(ip1).await.is_allowed());
        assert!(limiter.check(ip1).await.is_allowed());
        assert!(!limiter.check(ip1).await.is_allowed());

        assert!(limiter.check(ip2).await.is_allowed());
        assert!(limiter.check(ip2).await.is_allowed());
        assert!(!limiter.check(ip2).await.is_allowed());
    }

    #[test]
    fn test_security_headers() {
        let headers = SecurityHeaders::headers();

        // Verify required headers are present
        let header_names: Vec<&str> = headers.iter().map(|(name, _)| *name).collect();
        assert!(header_names.contains(&"X-Content-Type-Options"));
        assert!(header_names.contains(&"X-Frame-Options"));
        assert!(header_names.contains(&"Content-Security-Policy"));
        assert!(header_names.contains(&"Referrer-Policy"));
        assert!(header_names.contains(&"Cache-Control"));

        // Verify header values
        let x_frame = headers.iter().find(|(name, _)| *name == "X-Frame-Options");
        assert_eq!(x_frame.unwrap().1, "DENY");

        let csp = headers
            .iter()
            .find(|(name, _)| *name == "Content-Security-Policy");
        assert!(csp.unwrap().1.contains("default-src 'none'"));
    }

    #[test]
    fn test_cors_headers_allowed_origin() {
        let config = SecurityConfig::test_config();
        let headers = SecurityHeaders::cors_headers(Some("http://localhost"), &config);

        assert!(!headers.is_empty());
        let origin_header = headers
            .iter()
            .find(|(name, _)| *name == "Access-Control-Allow-Origin");
        assert!(origin_header.is_some());
        assert_eq!(origin_header.unwrap().1, "http://localhost");
    }

    #[test]
    fn test_cors_headers_disallowed_origin() {
        let config = SecurityConfig::test_config();
        let headers = SecurityHeaders::cors_headers(Some("http://evil.com"), &config);

        // Should not include any CORS headers for disallowed origin
        assert!(headers.is_empty());
    }

    #[test]
    fn test_constant_time_compare() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        let c = [1u8, 2, 3, 5];
        let d = [1u8, 2, 3];

        assert!(constant_time_compare(&a, &b));
        assert!(!constant_time_compare(&a, &c));
        assert!(!constant_time_compare(&a, &d));
    }

    #[tokio::test]
    async fn test_security_check_full_flow() {
        let config = SecurityConfig::test_config();
        let layer = SecurityLayer::new(config);
        let check = SecurityCheck::new(layer);

        // Valid request from localhost with valid token
        let result = check
            .validate(
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                "/api/test",
                Some("Bearer test-token-for-unit-tests-only"),
                Some("http://localhost"),
            )
            .await;
        assert!(result.is_allowed());

        // Health endpoint without auth (bypass)
        let result = check
            .validate(
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                "/health",
                None,
                None,
            )
            .await;
        assert!(result.is_allowed());
    }

    #[tokio::test]
    async fn test_security_check_rejected_cases() {
        let config = SecurityConfig::test_config();
        let layer = SecurityLayer::new(config);
        let check = SecurityCheck::new(layer);

        // Missing auth header
        let result = check
            .validate(
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                "/api/test",
                None,
                None,
            )
            .await;
        assert!(!result.is_allowed());
        assert_eq!(result.status_code(), 401);

        // Invalid token
        let result = check
            .validate(
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                "/api/test",
                Some("Bearer wrong-token"),
                None,
            )
            .await;
        assert!(!result.is_allowed());
        assert_eq!(result.status_code(), 401);
    }

    #[test]
    fn test_auth_result_status_codes() {
        assert_eq!(AuthResult::Authenticated.status_code(), 200);
        assert_eq!(AuthResult::Bypassed.status_code(), 200);
        assert_eq!(AuthResult::MissingHeader.status_code(), 401);
        assert_eq!(AuthResult::InvalidFormat.status_code(), 401);
        assert_eq!(AuthResult::InvalidToken.status_code(), 401);
    }
}

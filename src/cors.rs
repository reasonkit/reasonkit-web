//! CORS (Cross-Origin Resource Sharing) Configuration for ReasonKit Web
//!
//! This module provides a strict CORS policy for the HTTP server, allowing only
//! localhost origins for security. This is essential for MCP HTTP transport
//! and browser-based integrations.
//!
//! # Security Policy
//!
//! - **Allowed Origins**: Only `localhost` and `127.0.0.1` on any port
//! - **Allowed Methods**: GET, POST, OPTIONS (preflight)
//! - **Allowed Headers**: Content-Type, Authorization
//! - **Max Age**: 3600 seconds (1 hour) for preflight caching
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::cors::cors_layer;
//! use axum::Router;
//!
//! let app = Router::new()
//!     .route("/api/mcp", post(mcp_handler))
//!     .layer(cors_layer());
//! ```

use http::{header::HeaderValue, Method};
use std::time::Duration;
use tower_http::cors::{AllowOrigin, CorsLayer};

/// Standard allowed headers for MCP HTTP transport
pub const ALLOWED_HEADERS: [http::header::HeaderName; 2] =
    [http::header::CONTENT_TYPE, http::header::AUTHORIZATION];

/// Standard allowed methods for MCP HTTP transport
pub const ALLOWED_METHODS: [Method; 3] = [Method::GET, Method::POST, Method::OPTIONS];

/// Default max age for preflight cache (1 hour)
pub const DEFAULT_MAX_AGE_SECS: u64 = 3600;

/// Creates a strict CORS layer that only allows localhost origins.
///
/// This is the recommended configuration for development and local MCP servers.
/// For production deployments with specific domain requirements, use
/// [`cors_layer_with_origins`] instead.
///
/// # Security Properties
///
/// - Only allows requests from `http://localhost:*` and `http://127.0.0.1:*`
/// - Blocks all external origins including other private IP ranges
/// - Properly handles preflight OPTIONS requests
/// - Does not expose credentials by default
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::cors::cors_layer;
/// use axum::Router;
///
/// let app = Router::new()
///     .layer(cors_layer());
/// ```
pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            is_localhost_origin(origin)
        }))
        .allow_methods(ALLOWED_METHODS)
        .allow_headers(ALLOWED_HEADERS)
        .max_age(Duration::from_secs(DEFAULT_MAX_AGE_SECS))
}

/// Creates a CORS layer with custom configuration.
///
/// # Arguments
///
/// * `config` - CORS configuration options
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::cors::{cors_layer_with_config, CorsConfig};
///
/// let config = CorsConfig::default()
///     .with_max_age(7200)
///     .with_allow_credentials(true);
///
/// let layer = cors_layer_with_config(config);
/// ```
pub fn cors_layer_with_config(config: CorsConfig) -> CorsLayer {
    let mut layer = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            if config.allow_all_localhost {
                is_localhost_origin(origin)
            } else {
                // Strict mode: no origins allowed by default
                false
            }
        }))
        .allow_methods(config.allowed_methods.clone())
        .allow_headers(config.allowed_headers.clone())
        .max_age(Duration::from_secs(config.max_age_secs));

    if config.allow_credentials {
        layer = layer.allow_credentials(true);
    }

    if config.expose_headers {
        layer = layer.expose_headers([http::header::CONTENT_LENGTH, http::header::CONTENT_TYPE]);
    }

    layer
}

/// Creates a permissive CORS layer for development/testing.
///
/// # Warning
///
/// This configuration is NOT secure for production use. It allows all origins.
/// Use only for local development and testing.
///
/// # Example
///
/// ```rust,no_run
/// use reasonkit_web::cors::cors_layer_permissive;
///
/// // Only for development!
/// #[cfg(debug_assertions)]
/// let layer = cors_layer_permissive();
/// ```
pub fn cors_layer_permissive() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any)
        .max_age(Duration::from_secs(DEFAULT_MAX_AGE_SECS))
}

/// CORS configuration options.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Whether to allow all localhost origins (default: true)
    pub allow_all_localhost: bool,
    /// Whether to allow credentials (cookies, auth headers)
    pub allow_credentials: bool,
    /// Whether to expose response headers to the client
    pub expose_headers: bool,
    /// Maximum age for preflight cache in seconds
    pub max_age_secs: u64,
    /// Allowed HTTP methods
    pub allowed_methods: Vec<Method>,
    /// Allowed request headers
    pub allowed_headers: Vec<http::header::HeaderName>,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allow_all_localhost: true,
            allow_credentials: false,
            expose_headers: false,
            max_age_secs: DEFAULT_MAX_AGE_SECS,
            allowed_methods: ALLOWED_METHODS.to_vec(),
            allowed_headers: ALLOWED_HEADERS.to_vec(),
        }
    }
}

impl CorsConfig {
    /// Create a new CORS configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum age for preflight cache.
    pub fn with_max_age(mut self, secs: u64) -> Self {
        self.max_age_secs = secs;
        self
    }

    /// Enable or disable credentials support.
    pub fn with_allow_credentials(mut self, allow: bool) -> Self {
        self.allow_credentials = allow;
        self
    }

    /// Enable or disable header exposure.
    pub fn with_expose_headers(mut self, expose: bool) -> Self {
        self.expose_headers = expose;
        self
    }

    /// Set allowed HTTP methods.
    pub fn with_methods(mut self, methods: Vec<Method>) -> Self {
        self.allowed_methods = methods;
        self
    }

    /// Set allowed request headers.
    pub fn with_headers(mut self, headers: Vec<http::header::HeaderName>) -> Self {
        self.allowed_headers = headers;
        self
    }

    /// Disable localhost origin allowance (for custom origin handling).
    pub fn with_strict_origins(mut self) -> Self {
        self.allow_all_localhost = false;
        self
    }
}

/// Checks if the given origin is a localhost origin.
///
/// # Valid Origins
///
/// - `http://localhost` (any port)
/// - `http://127.0.0.1` (any port)
/// - `https://localhost` (any port, for secure contexts)
/// - `https://127.0.0.1` (any port)
///
/// # Invalid Origins
///
/// - External domains (e.g., `http://example.com`)
/// - Other private IPs (e.g., `http://192.168.1.1`)
/// - IPv6 localhost (currently not supported)
///
/// # Arguments
///
/// * `origin` - The Origin header value to check
///
/// # Returns
///
/// `true` if the origin is a valid localhost origin, `false` otherwise.
///
/// # Example
///
/// ```rust
/// use http::header::HeaderValue;
/// use reasonkit_web::cors::is_localhost_origin;
///
/// let origin = HeaderValue::from_static("http://localhost:3000");
/// assert!(is_localhost_origin(&origin));
///
/// let external = HeaderValue::from_static("http://example.com");
/// assert!(!is_localhost_origin(&external));
/// ```
pub fn is_localhost_origin(origin: &HeaderValue) -> bool {
    let origin_str = match origin.to_str() {
        Ok(s) => s,
        Err(_) => return false, // Invalid UTF-8, reject
    };

    // Parse the origin to extract host
    // Origin format: scheme://host[:port]
    let origin_lower = origin_str.to_lowercase();

    // Check for localhost patterns
    // http://localhost or http://localhost:PORT
    if origin_lower.starts_with("http://localhost") || origin_lower.starts_with("https://localhost")
    {
        return validate_localhost_format(&origin_lower, "localhost");
    }

    // Check for 127.0.0.1 patterns
    // http://127.0.0.1 or http://127.0.0.1:PORT
    if origin_lower.starts_with("http://127.0.0.1") || origin_lower.starts_with("https://127.0.0.1")
    {
        return validate_localhost_format(&origin_lower, "127.0.0.1");
    }

    // Check for IPv6 localhost [::1]
    if origin_lower.starts_with("http://[::1]") || origin_lower.starts_with("https://[::1]") {
        return validate_ipv6_localhost_format(&origin_lower);
    }

    false
}

/// Validates the format of a localhost origin string.
fn validate_localhost_format(origin: &str, host: &str) -> bool {
    // Find the position after the host
    let scheme_end = if origin.starts_with("https://") {
        8
    } else {
        7 // "http://"
    };

    let after_host = scheme_end + host.len();

    // Check what follows the host
    if origin.len() == after_host {
        // Exact match: http://localhost
        return true;
    }

    let remaining = &origin[after_host..];

    // Should be either end of string, port, or path
    if remaining.starts_with(':') {
        // Port follows - validate it's a number
        let port_str = &remaining[1..];
        // Port might be followed by path
        let port_end = port_str.find('/').unwrap_or(port_str.len());
        let port = &port_str[..port_end];

        // Validate port is numeric and in valid range
        if let Ok(port_num) = port.parse::<u16>() {
            return port_num > 0;
        }
        return false;
    }

    if remaining.starts_with('/') {
        // Path follows directly (no port)
        return true;
    }

    // Invalid format (e.g., "localhostevil.com")
    false
}

/// Validates the format of an IPv6 localhost origin string.
fn validate_ipv6_localhost_format(origin: &str) -> bool {
    // IPv6 localhost format: http://[::1] or http://[::1]:PORT
    let scheme_end = if origin.starts_with("https://") { 8 } else { 7 };

    let after_bracket = origin[scheme_end..].find(']');
    if let Some(pos) = after_bracket {
        let after_host = scheme_end + pos + 1;
        if origin.len() == after_host {
            return true;
        }

        let remaining = &origin[after_host..];
        if remaining.starts_with(':') {
            let port_str = &remaining[1..];
            let port_end = port_str.find('/').unwrap_or(port_str.len());
            let port = &port_str[..port_end];
            if let Ok(port_num) = port.parse::<u16>() {
                return port_num > 0;
            }
            return false;
        }

        if remaining.starts_with('/') {
            return true;
        }
    }

    false
}

/// Result of CORS validation containing diagnostic information.
#[derive(Debug, Clone)]
pub struct CorsValidationResult {
    /// Whether the origin is allowed
    pub allowed: bool,
    /// The origin that was checked
    pub origin: String,
    /// Reason for the decision
    pub reason: String,
}

impl CorsValidationResult {
    /// Create a new validation result.
    pub fn new(allowed: bool, origin: String, reason: String) -> Self {
        Self {
            allowed,
            origin,
            reason,
        }
    }
}

/// Validates an origin and returns detailed information.
///
/// Useful for debugging and logging CORS decisions.
///
/// # Example
///
/// ```rust
/// use reasonkit_web::cors::validate_origin;
///
/// let result = validate_origin("http://localhost:3000");
/// assert!(result.allowed);
/// println!("Reason: {}", result.reason);
/// ```
pub fn validate_origin(origin: &str) -> CorsValidationResult {
    let header_value = match HeaderValue::from_str(origin) {
        Ok(v) => v,
        Err(_) => {
            return CorsValidationResult::new(
                false,
                origin.to_string(),
                "Invalid header value format".to_string(),
            );
        }
    };

    let allowed = is_localhost_origin(&header_value);
    let reason = if allowed {
        "Localhost origin allowed".to_string()
    } else {
        determine_rejection_reason(origin)
    };

    CorsValidationResult::new(allowed, origin.to_string(), reason)
}

/// Determines the specific reason why an origin was rejected.
fn determine_rejection_reason(origin: &str) -> String {
    let origin_lower = origin.to_lowercase();

    if !origin_lower.starts_with("http://") && !origin_lower.starts_with("https://") {
        return "Invalid scheme: must be http:// or https://".to_string();
    }

    if origin_lower.contains("localhost") && !is_valid_localhost_pattern(&origin_lower) {
        return "Invalid localhost format: possible subdomain attack".to_string();
    }

    if origin_lower.contains("127.0.0.1") && !is_valid_loopback_pattern(&origin_lower) {
        return "Invalid 127.0.0.1 format".to_string();
    }

    // Check for other private IPs that we don't allow
    if is_private_ip_origin(&origin_lower) {
        return "Private IP origins other than 127.0.0.1 are not allowed".to_string();
    }

    "External origin not allowed: only localhost origins permitted".to_string()
}

/// Checks if the origin matches valid localhost patterns.
fn is_valid_localhost_pattern(origin: &str) -> bool {
    let patterns = [
        "http://localhost",
        "https://localhost",
        "http://localhost:",
        "https://localhost:",
        "http://localhost/",
        "https://localhost/",
    ];

    for pattern in patterns {
        if origin.starts_with(pattern) {
            return true;
        }
    }

    false
}

/// Checks if the origin matches valid loopback patterns.
fn is_valid_loopback_pattern(origin: &str) -> bool {
    let patterns = [
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://127.0.0.1:",
        "https://127.0.0.1:",
        "http://127.0.0.1/",
        "https://127.0.0.1/",
    ];

    for pattern in patterns {
        if origin.starts_with(pattern) {
            return true;
        }
    }

    false
}

/// Checks if the origin appears to be a private IP (not 127.0.0.1).
fn is_private_ip_origin(origin: &str) -> bool {
    // Common private IP ranges we want to block
    let private_patterns = [
        "192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
        "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
        "172.30.", "172.31.",
    ];

    for pattern in private_patterns {
        if origin.contains(pattern) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Localhost Origin Tests ====================

    #[test]
    fn test_localhost_origin_http() {
        let origin = HeaderValue::from_static("http://localhost");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost should be allowed"
        );
    }

    #[test]
    fn test_localhost_origin_https() {
        let origin = HeaderValue::from_static("https://localhost");
        assert!(
            is_localhost_origin(&origin),
            "https://localhost should be allowed"
        );
    }

    #[test]
    fn test_localhost_origin_with_port() {
        let origin = HeaderValue::from_static("http://localhost:3000");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost:3000 should be allowed"
        );
    }

    #[test]
    fn test_localhost_origin_with_high_port() {
        let origin = HeaderValue::from_static("http://localhost:65535");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost:65535 should be allowed"
        );
    }

    #[test]
    fn test_localhost_origin_with_path() {
        let origin = HeaderValue::from_static("http://localhost/api");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost/api should be allowed"
        );
    }

    #[test]
    fn test_localhost_origin_with_port_and_path() {
        let origin = HeaderValue::from_static("http://localhost:8080/api/v1");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost:8080/api/v1 should be allowed"
        );
    }

    // ==================== 127.0.0.1 Origin Tests ====================

    #[test]
    fn test_loopback_origin_http() {
        let origin = HeaderValue::from_static("http://127.0.0.1");
        assert!(
            is_localhost_origin(&origin),
            "http://127.0.0.1 should be allowed"
        );
    }

    #[test]
    fn test_loopback_origin_https() {
        let origin = HeaderValue::from_static("https://127.0.0.1");
        assert!(
            is_localhost_origin(&origin),
            "https://127.0.0.1 should be allowed"
        );
    }

    #[test]
    fn test_loopback_origin_with_port() {
        let origin = HeaderValue::from_static("http://127.0.0.1:8000");
        assert!(
            is_localhost_origin(&origin),
            "http://127.0.0.1:8000 should be allowed"
        );
    }

    #[test]
    fn test_loopback_origin_with_path() {
        let origin = HeaderValue::from_static("http://127.0.0.1/mcp");
        assert!(
            is_localhost_origin(&origin),
            "http://127.0.0.1/mcp should be allowed"
        );
    }

    // ==================== IPv6 Localhost Tests ====================

    #[test]
    fn test_ipv6_localhost_origin() {
        let origin = HeaderValue::from_static("http://[::1]");
        assert!(
            is_localhost_origin(&origin),
            "http://[::1] should be allowed"
        );
    }

    #[test]
    fn test_ipv6_localhost_origin_with_port() {
        let origin = HeaderValue::from_static("http://[::1]:3000");
        assert!(
            is_localhost_origin(&origin),
            "http://[::1]:3000 should be allowed"
        );
    }

    #[test]
    fn test_ipv6_localhost_origin_https() {
        let origin = HeaderValue::from_static("https://[::1]:8080");
        assert!(
            is_localhost_origin(&origin),
            "https://[::1]:8080 should be allowed"
        );
    }

    // ==================== External Origin Tests (Should Block) ====================

    #[test]
    fn test_external_origin_blocked() {
        let origin = HeaderValue::from_static("http://example.com");
        assert!(
            !is_localhost_origin(&origin),
            "http://example.com should be blocked"
        );
    }

    #[test]
    fn test_external_origin_with_port_blocked() {
        let origin = HeaderValue::from_static("http://evil.com:3000");
        assert!(
            !is_localhost_origin(&origin),
            "http://evil.com:3000 should be blocked"
        );
    }

    #[test]
    fn test_external_https_blocked() {
        let origin = HeaderValue::from_static("https://malicious.org");
        assert!(
            !is_localhost_origin(&origin),
            "https://malicious.org should be blocked"
        );
    }

    // ==================== Subdomain Attack Prevention Tests ====================

    #[test]
    fn test_localhost_subdomain_attack_blocked() {
        let origin = HeaderValue::from_static("http://localhost.evil.com");
        assert!(
            !is_localhost_origin(&origin),
            "http://localhost.evil.com should be blocked (subdomain attack)"
        );
    }

    #[test]
    fn test_localhostevil_blocked() {
        let origin = HeaderValue::from_static("http://localhostevil.com");
        assert!(
            !is_localhost_origin(&origin),
            "http://localhostevil.com should be blocked"
        );
    }

    #[test]
    fn test_subdomain_localhost_blocked() {
        let origin = HeaderValue::from_static("http://sub.localhost.com");
        assert!(
            !is_localhost_origin(&origin),
            "http://sub.localhost.com should be blocked"
        );
    }

    #[test]
    fn test_fake_localhost_blocked() {
        let origin = HeaderValue::from_static("http://my-localhost.com");
        assert!(
            !is_localhost_origin(&origin),
            "http://my-localhost.com should be blocked"
        );
    }

    // ==================== Private IP Tests (Should Block) ====================

    #[test]
    fn test_private_ip_192_blocked() {
        let origin = HeaderValue::from_static("http://192.168.1.1");
        assert!(
            !is_localhost_origin(&origin),
            "http://192.168.1.1 should be blocked"
        );
    }

    #[test]
    fn test_private_ip_10_blocked() {
        let origin = HeaderValue::from_static("http://10.0.0.1:8080");
        assert!(
            !is_localhost_origin(&origin),
            "http://10.0.0.1:8080 should be blocked"
        );
    }

    #[test]
    fn test_private_ip_172_blocked() {
        let origin = HeaderValue::from_static("http://172.16.0.1");
        assert!(
            !is_localhost_origin(&origin),
            "http://172.16.0.1 should be blocked"
        );
    }

    // ==================== Invalid Format Tests ====================

    #[test]
    fn test_no_scheme_blocked() {
        let origin = HeaderValue::from_static("localhost:3000");
        assert!(
            !is_localhost_origin(&origin),
            "localhost:3000 (no scheme) should be blocked"
        );
    }

    #[test]
    fn test_ftp_scheme_blocked() {
        let origin = HeaderValue::from_static("ftp://localhost");
        assert!(
            !is_localhost_origin(&origin),
            "ftp://localhost should be blocked"
        );
    }

    #[test]
    fn test_file_scheme_blocked() {
        let origin = HeaderValue::from_static("file://localhost");
        assert!(
            !is_localhost_origin(&origin),
            "file://localhost should be blocked"
        );
    }

    #[test]
    fn test_invalid_port_blocked() {
        let origin = HeaderValue::from_static("http://localhost:notaport");
        assert!(
            !is_localhost_origin(&origin),
            "http://localhost:notaport should be blocked"
        );
    }

    #[test]
    fn test_port_zero_blocked() {
        let origin = HeaderValue::from_static("http://localhost:0");
        assert!(
            !is_localhost_origin(&origin),
            "http://localhost:0 should be blocked (invalid port)"
        );
    }

    // ==================== CORS Config Tests ====================

    #[test]
    fn test_cors_config_default() {
        let config = CorsConfig::default();
        assert!(config.allow_all_localhost);
        assert!(!config.allow_credentials);
        assert!(!config.expose_headers);
        assert_eq!(config.max_age_secs, DEFAULT_MAX_AGE_SECS);
    }

    #[test]
    fn test_cors_config_builder() {
        let config = CorsConfig::new()
            .with_max_age(7200)
            .with_allow_credentials(true)
            .with_expose_headers(true);

        assert_eq!(config.max_age_secs, 7200);
        assert!(config.allow_credentials);
        assert!(config.expose_headers);
    }

    #[test]
    fn test_cors_config_strict_origins() {
        let config = CorsConfig::new().with_strict_origins();
        assert!(!config.allow_all_localhost);
    }

    // ==================== Validation Result Tests ====================

    #[test]
    fn test_validate_origin_allowed() {
        let result = validate_origin("http://localhost:3000");
        assert!(result.allowed);
        assert_eq!(result.origin, "http://localhost:3000");
        assert!(result.reason.contains("allowed"));
    }

    #[test]
    fn test_validate_origin_blocked_external() {
        let result = validate_origin("http://example.com");
        assert!(!result.allowed);
        assert!(result.reason.contains("External") || result.reason.contains("not allowed"));
    }

    #[test]
    fn test_validate_origin_blocked_private_ip() {
        let result = validate_origin("http://192.168.1.100");
        assert!(!result.allowed);
        assert!(result.reason.contains("Private IP") || result.reason.contains("not allowed"));
    }

    #[test]
    fn test_validate_origin_blocked_subdomain_attack() {
        let result = validate_origin("http://localhost.evil.com");
        assert!(!result.allowed);
    }

    // ==================== Layer Creation Tests ====================

    #[test]
    fn test_cors_layer_creation() {
        let layer = cors_layer();
        // Layer should be created without panicking
        let _ = format!("{:?}", layer);
    }

    #[test]
    fn test_cors_layer_with_config_creation() {
        let config = CorsConfig::new()
            .with_max_age(1800)
            .with_allow_credentials(true);
        let layer = cors_layer_with_config(config);
        let _ = format!("{:?}", layer);
    }

    #[test]
    fn test_cors_layer_permissive_creation() {
        let layer = cors_layer_permissive();
        let _ = format!("{:?}", layer);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_empty_origin_blocked() {
        let origin = HeaderValue::from_static("");
        assert!(
            !is_localhost_origin(&origin),
            "Empty origin should be blocked"
        );
    }

    #[test]
    fn test_case_insensitive_localhost() {
        let origin = HeaderValue::from_static("HTTP://LOCALHOST:3000");
        assert!(
            is_localhost_origin(&origin),
            "HTTP://LOCALHOST:3000 should be allowed (case insensitive)"
        );
    }

    #[test]
    fn test_case_insensitive_loopback() {
        let origin = HeaderValue::from_static("HTTPS://127.0.0.1:8080");
        assert!(
            is_localhost_origin(&origin),
            "HTTPS://127.0.0.1:8080 should be allowed (case insensitive)"
        );
    }

    #[test]
    fn test_localhost_with_trailing_slash() {
        let origin = HeaderValue::from_static("http://localhost/");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost/ should be allowed"
        );
    }

    #[test]
    fn test_port_boundary_1() {
        let origin = HeaderValue::from_static("http://localhost:1");
        assert!(
            is_localhost_origin(&origin),
            "http://localhost:1 should be allowed"
        );
    }

    #[test]
    fn test_common_dev_ports() {
        let ports = ["3000", "5000", "8000", "8080", "9000", "4200", "5173"];
        for port in ports {
            let origin_str = format!("http://localhost:{}", port);
            let origin = HeaderValue::from_str(&origin_str).unwrap();
            assert!(
                is_localhost_origin(&origin),
                "http://localhost:{} should be allowed",
                port
            );
        }
    }
}

//! Page navigation functionality
//!
//! This module handles URL navigation with retry logic, timeout handling,
//! and human-like behavior simulation.

use crate::browser::PageHandle;
use crate::error::{Error, NavigationError, Result};
use std::time::Duration;
use tracing::{debug, info, instrument, warn};

/// Options for page navigation
#[derive(Debug, Clone)]
pub struct NavigationOptions {
    /// Timeout in milliseconds (default: 30000)
    pub timeout_ms: u64,
    /// Wait until condition (default: networkidle0)
    pub wait_until: WaitUntil,
    /// Number of retry attempts (default: 3)
    pub retries: u32,
    /// Delay between retries in ms (default: 1000)
    pub retry_delay_ms: u64,
    /// Simulate human-like behavior (default: true)
    pub human_like: bool,
}

impl Default for NavigationOptions {
    fn default() -> Self {
        Self {
            timeout_ms: 30000,
            wait_until: WaitUntil::NetworkIdle0,
            retries: 3,
            retry_delay_ms: 1000,
            human_like: true,
        }
    }
}

/// Condition to wait for after navigation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitUntil {
    /// Wait until load event fires
    Load,
    /// Wait until DOMContentLoaded event fires
    DomContentLoaded,
    /// Wait until network is idle (0 connections for 500ms)
    NetworkIdle0,
    /// Wait until network is idle (max 2 connections for 500ms)
    NetworkIdle2,
}

/// Result of a navigation operation
#[derive(Debug)]
pub struct NavigationResult {
    /// Final URL after any redirects
    pub final_url: String,
    /// HTTP status code
    pub status: Option<u16>,
    /// Page title
    pub title: Option<String>,
    /// Navigation duration in milliseconds
    pub duration_ms: u64,
}

/// URL validation utilities
pub struct UrlValidator;

impl UrlValidator {
    /// Validate a URL for navigation
    pub fn validate(url: &str) -> std::result::Result<(), String> {
        // Check for empty URL
        if url.is_empty() {
            return Err("URL cannot be empty".to_string());
        }

        // Check for valid protocol
        if !url.starts_with("http://")
            && !url.starts_with("https://")
            && !url.starts_with("file://")
        {
            return Err(format!(
                "URL must start with http://, https://, or file://: {}",
                url
            ));
        }

        // Check URL length (max 2048 characters is common limit)
        if url.len() > 2048 {
            return Err("URL exceeds maximum length of 2048 characters".to_string());
        }

        // Check for localhost/127.0.0.1 (allowed but flagged)
        // This is informational - we still allow it
        let _is_localhost = Self::is_localhost(url);

        Ok(())
    }

    /// Check if URL points to localhost
    pub fn is_localhost(url: &str) -> bool {
        let lower = url.to_lowercase();
        lower.contains("://localhost")
            || lower.contains("://127.0.0.1")
            || lower.contains("://[::1]")
            || lower.contains("://0.0.0.0")
    }

    /// Check if URL is external (not localhost)
    pub fn is_external(url: &str) -> bool {
        !Self::is_localhost(url)
    }

    /// Extract host from URL
    pub fn extract_host(url: &str) -> Option<String> {
        // Simple extraction - find :// then extract until next / or end
        if let Some(protocol_end) = url.find("://") {
            let after_protocol = &url[protocol_end + 3..];
            let host_end = after_protocol.find('/').unwrap_or(after_protocol.len());
            let host_with_port = &after_protocol[..host_end];
            // Remove port if present
            let host = host_with_port
                .rsplit(':')
                .last()
                .or(Some(host_with_port))
                .map(|h| {
                    if host_with_port.contains(':') && !host_with_port.starts_with('[') {
                        // IPv4 with port
                        host_with_port.split(':').next().unwrap_or(host_with_port)
                    } else {
                        h
                    }
                })?;
            Some(host.to_string())
        } else {
            None
        }
    }
}

/// Simple rate limiter for requests
pub struct RateLimiter {
    /// Maximum requests per window
    max_requests: u32,
    /// Window duration in seconds
    window_secs: u64,
    /// Current request count in window
    request_count: u32,
    /// Window start time
    window_start: std::time::Instant,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window_secs,
            request_count: 0,
            window_start: std::time::Instant::now(),
        }
    }

    /// Check if a request is allowed (and count it if so)
    pub fn check(&mut self) -> bool {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.window_start).as_secs();

        // Reset window if expired
        if elapsed >= self.window_secs {
            self.window_start = now;
            self.request_count = 0;
        }

        // Check if under limit
        if self.request_count < self.max_requests {
            self.request_count += 1;
            true
        } else {
            false
        }
    }

    /// Get remaining requests in current window
    pub fn remaining(&self) -> u32 {
        if self.request_count >= self.max_requests {
            0
        } else {
            self.max_requests - self.request_count
        }
    }

    /// Reset the rate limiter
    pub fn reset(&mut self) {
        self.request_count = 0;
        self.window_start = std::time::Instant::now();
    }
}

/// Page navigator with advanced navigation capabilities
pub struct PageNavigator;

impl PageNavigator {
    /// Navigate to a URL with default options
    #[instrument(skip(page))]
    pub async fn goto(
        page: &PageHandle,
        url: &str,
        options: Option<NavigationOptions>,
    ) -> Result<NavigationResult> {
        let opts = options.unwrap_or_default();
        let start = std::time::Instant::now();

        // Validate URL
        if !url.starts_with("http://")
            && !url.starts_with("https://")
            && !url.starts_with("file://")
        {
            return Err(NavigationError::InvalidUrl(format!(
                "URL must start with http://, https://, or file://: {}",
                url
            ))
            .into());
        }

        info!("Navigating to: {}", url);

        let mut last_error = None;
        for attempt in 0..=opts.retries {
            if attempt > 0 {
                warn!("Navigation retry attempt {} of {}", attempt, opts.retries);
                tokio::time::sleep(Duration::from_millis(opts.retry_delay_ms)).await;
            }

            match Self::navigate_once(&page.page, url, &opts).await {
                Ok(result) => {
                    // Update page URL
                    page.set_url(result.final_url.clone()).await;

                    // Apply human-like behavior if enabled
                    if opts.human_like {
                        Self::simulate_human_behavior(&page.page).await?;
                    }

                    let duration_ms = start.elapsed().as_millis() as u64;
                    return Ok(NavigationResult {
                        final_url: result.final_url,
                        status: result.status,
                        title: result.title,
                        duration_ms,
                    });
                }
                Err(e) => {
                    warn!("Navigation attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            NavigationError::LoadFailed("Navigation failed after all retries".to_string()).into()
        }))
    }

    /// Perform a single navigation attempt
    async fn navigate_once(
        page: &chromiumoxide::Page,
        url: &str,
        opts: &NavigationOptions,
    ) -> Result<NavigationResult> {
        // Navigate with timeout
        let timeout = Duration::from_millis(opts.timeout_ms);

        let nav_future = page.goto(url);
        let _response = tokio::time::timeout(timeout, nav_future)
            .await
            .map_err(|_| NavigationError::Timeout(opts.timeout_ms))?
            .map_err(|e| NavigationError::LoadFailed(e.to_string()))?;

        // Wait for page to be ready based on wait_until option
        Self::wait_for_ready(page, opts).await?;

        // Get final URL and title
        let final_url = page
            .url()
            .await
            .map_err(|e| Error::cdp(e.to_string()))?
            .unwrap_or_else(|| url.to_string());

        let title = page
            .evaluate("document.title")
            .await
            .ok()
            .and_then(|v| v.into_value::<String>().ok());

        // Navigation doesn't return status directly in chromiumoxide
        let status: Option<u16> = None;

        debug!("Navigation complete: {} -> {}", url, final_url);

        Ok(NavigationResult {
            final_url,
            status,
            title,
            duration_ms: 0, // Will be set by caller
        })
    }

    /// Wait for page to be ready based on wait_until condition
    async fn wait_for_ready(page: &chromiumoxide::Page, opts: &NavigationOptions) -> Result<()> {
        let script = match opts.wait_until {
            WaitUntil::Load => {
                r#"
                    new Promise(resolve => {
                        if (document.readyState === 'complete') {
                            resolve(true);
                        } else {
                            window.addEventListener('load', () => resolve(true));
                        }
                    })
                "#
            }
            WaitUntil::DomContentLoaded => {
                r#"
                    new Promise(resolve => {
                        if (document.readyState !== 'loading') {
                            resolve(true);
                        } else {
                            document.addEventListener('DOMContentLoaded', () => resolve(true));
                        }
                    })
                "#
            }
            WaitUntil::NetworkIdle0 | WaitUntil::NetworkIdle2 => {
                // For network idle, we'll just wait a short time after load
                // A more sophisticated approach would monitor actual network activity
                r#"
                    new Promise(resolve => {
                        if (document.readyState === 'complete') {
                            setTimeout(() => resolve(true), 500);
                        } else {
                            window.addEventListener('load', () => {
                                setTimeout(() => resolve(true), 500);
                            });
                        }
                    })
                "#
            }
        };

        let timeout = Duration::from_millis(opts.timeout_ms);
        tokio::time::timeout(timeout, page.evaluate(script))
            .await
            .map_err(|_| NavigationError::Timeout(opts.timeout_ms))?
            .map_err(|e| Error::cdp(e.to_string()))?;

        Ok(())
    }

    /// Simulate human-like behavior after navigation
    async fn simulate_human_behavior(page: &chromiumoxide::Page) -> Result<()> {
        // Random small delay
        let delay = rand::random::<u64>() % 500 + 200;
        tokio::time::sleep(Duration::from_millis(delay)).await;

        // Gentle scroll
        let scroll_script = r#"
            window.scrollTo({
                top: Math.random() * 100 + 50,
                behavior: 'smooth'
            });
        "#;

        let _ = page.evaluate(scroll_script).await;

        // Small delay after scroll
        tokio::time::sleep(Duration::from_millis(200)).await;

        Ok(())
    }

    /// Go back in browser history
    #[instrument(skip(page))]
    pub async fn back(page: &PageHandle) -> Result<()> {
        page.page
            .evaluate("window.history.back()")
            .await
            .map_err(|e| Error::cdp(e.to_string()))?;

        tokio::time::sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    /// Go forward in browser history
    #[instrument(skip(page))]
    pub async fn forward(page: &PageHandle) -> Result<()> {
        page.page
            .evaluate("window.history.forward()")
            .await
            .map_err(|e| Error::cdp(e.to_string()))?;

        tokio::time::sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    /// Reload the current page
    #[instrument(skip(page))]
    pub async fn reload(page: &PageHandle) -> Result<()> {
        page.page
            .reload()
            .await
            .map_err(|e| Error::cdp(e.to_string()))?;

        Ok(())
    }

    /// Wait for a specific element to appear
    #[instrument(skip(page))]
    pub async fn wait_for_selector(
        page: &PageHandle,
        selector: &str,
        timeout_ms: u64,
    ) -> Result<()> {
        let script = format!(
            r#"
                new Promise((resolve, reject) => {{
                    const timeout = {};
                    const start = Date.now();

                    function check() {{
                        const el = document.querySelector('{}');
                        if (el) {{
                            resolve(true);
                        }} else if (Date.now() - start > timeout) {{
                            reject(new Error('Timeout waiting for selector'));
                        }} else {{
                            requestAnimationFrame(check);
                        }}
                    }}
                    check();
                }})
            "#,
            timeout_ms,
            selector.replace('\'', "\\'")
        );

        let timeout = Duration::from_millis(timeout_ms + 1000);
        tokio::time::timeout(timeout, page.page.evaluate(script.as_str()))
            .await
            .map_err(|_| NavigationError::Timeout(timeout_ms))?
            .map_err(|e| Error::cdp(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // NavigationOptions Tests
    // ========================================================================

    #[test]
    fn test_navigation_options_default() {
        let opts = NavigationOptions::default();
        assert_eq!(opts.timeout_ms, 30000);
        assert_eq!(opts.retries, 3);
        assert!(opts.human_like);
        assert_eq!(opts.retry_delay_ms, 1000);
    }

    #[test]
    fn test_wait_until_variants() {
        assert_ne!(WaitUntil::Load, WaitUntil::DomContentLoaded);
        assert_eq!(WaitUntil::NetworkIdle0, WaitUntil::NetworkIdle0);
    }

    // ========================================================================
    // URL Validation Tests
    // ========================================================================

    #[test]
    fn test_url_validation_valid_http() {
        assert!(UrlValidator::validate("http://example.com").is_ok());
    }

    #[test]
    fn test_url_validation_valid_https() {
        assert!(UrlValidator::validate("https://example.com").is_ok());
    }

    #[test]
    fn test_url_validation_valid_file() {
        assert!(UrlValidator::validate("file:///path/to/file.html").is_ok());
    }

    #[test]
    fn test_url_validation_empty() {
        let result = UrlValidator::validate("");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_url_validation_no_protocol() {
        let result = UrlValidator::validate("example.com");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must start with"));
    }

    #[test]
    fn test_url_validation_invalid_protocol() {
        let result = UrlValidator::validate("ftp://example.com");
        assert!(result.is_err());
    }

    #[test]
    fn test_url_validation_too_long() {
        let long_url = format!("https://example.com/{}", "a".repeat(3000));
        let result = UrlValidator::validate(&long_url);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("maximum length"));
    }

    // ========================================================================
    // Localhost Check Tests
    // ========================================================================

    #[test]
    fn test_localhost_check_127001() {
        assert!(UrlValidator::is_localhost("http://127.0.0.1:8080"));
        assert!(UrlValidator::is_localhost("https://127.0.0.1/path"));
    }

    #[test]
    fn test_localhost_check_localhost() {
        assert!(UrlValidator::is_localhost("http://localhost:3000"));
        assert!(UrlValidator::is_localhost("https://localhost/api"));
    }

    #[test]
    fn test_localhost_check_ipv6_loopback() {
        assert!(UrlValidator::is_localhost("http://[::1]:8080"));
    }

    #[test]
    fn test_localhost_check_zero_addr() {
        assert!(UrlValidator::is_localhost("http://0.0.0.0:8080"));
    }

    #[test]
    fn test_localhost_check_external() {
        assert!(!UrlValidator::is_localhost("https://example.com"));
        assert!(!UrlValidator::is_localhost("https://google.com"));
        assert!(!UrlValidator::is_localhost("http://192.168.1.1"));
    }

    #[test]
    fn test_is_external() {
        assert!(UrlValidator::is_external("https://example.com"));
        assert!(!UrlValidator::is_external("http://localhost:8080"));
        assert!(!UrlValidator::is_external("http://127.0.0.1"));
    }

    // ========================================================================
    // Host Extraction Tests
    // ========================================================================

    #[test]
    fn test_extract_host_simple() {
        assert_eq!(
            UrlValidator::extract_host("https://example.com/path"),
            Some("example.com".to_string())
        );
    }

    #[test]
    fn test_extract_host_with_port() {
        assert_eq!(
            UrlValidator::extract_host("http://localhost:8080/api"),
            Some("localhost".to_string())
        );
    }

    #[test]
    fn test_extract_host_no_path() {
        assert_eq!(
            UrlValidator::extract_host("https://google.com"),
            Some("google.com".to_string())
        );
    }

    #[test]
    fn test_extract_host_no_protocol() {
        assert_eq!(UrlValidator::extract_host("example.com"), None);
    }

    // ========================================================================
    // Rate Limiter Tests
    // ========================================================================

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let mut limiter = RateLimiter::new(5, 60);

        // First 5 requests should be allowed
        assert!(limiter.check());
        assert!(limiter.check());
        assert!(limiter.check());
        assert!(limiter.check());
        assert!(limiter.check());
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let mut limiter = RateLimiter::new(3, 60);

        // First 3 requests should be allowed
        assert!(limiter.check());
        assert!(limiter.check());
        assert!(limiter.check());

        // 4th request should be blocked
        assert!(!limiter.check());
        assert!(!limiter.check());
    }

    #[test]
    fn test_rate_limiter_remaining() {
        let mut limiter = RateLimiter::new(5, 60);

        assert_eq!(limiter.remaining(), 5);
        limiter.check();
        assert_eq!(limiter.remaining(), 4);
        limiter.check();
        limiter.check();
        assert_eq!(limiter.remaining(), 2);
    }

    #[test]
    fn test_rate_limiter_reset() {
        let mut limiter = RateLimiter::new(3, 60);

        limiter.check();
        limiter.check();
        limiter.check();
        assert_eq!(limiter.remaining(), 0);
        assert!(!limiter.check());

        limiter.reset();
        assert_eq!(limiter.remaining(), 3);
        assert!(limiter.check());
    }

    #[test]
    fn test_rate_limiter_single_request() {
        let mut limiter = RateLimiter::new(1, 60);
        assert!(limiter.check());
        assert!(!limiter.check());
    }

    #[test]
    fn test_rate_limiter_zero_remaining_after_exhaustion() {
        let mut limiter = RateLimiter::new(2, 60);
        limiter.check();
        limiter.check();
        assert_eq!(limiter.remaining(), 0);
    }

    // ========================================================================
    // NavigationResult Tests
    // ========================================================================

    #[test]
    fn test_navigation_result_structure() {
        let result = NavigationResult {
            final_url: "https://example.com".to_string(),
            status: Some(200),
            title: Some("Example".to_string()),
            duration_ms: 150,
        };

        assert_eq!(result.final_url, "https://example.com");
        assert_eq!(result.status, Some(200));
        assert_eq!(result.title, Some("Example".to_string()));
        assert_eq!(result.duration_ms, 150);
    }

    #[test]
    fn test_navigation_result_without_status() {
        let result = NavigationResult {
            final_url: "https://example.com".to_string(),
            status: None,
            title: None,
            duration_ms: 100,
        };

        assert!(result.status.is_none());
        assert!(result.title.is_none());
    }

    // ========================================================================
    // Edge Cases Tests
    // ========================================================================

    #[test]
    fn test_url_validation_with_query_params() {
        assert!(UrlValidator::validate("https://example.com?foo=bar&baz=123").is_ok());
    }

    #[test]
    fn test_url_validation_with_fragment() {
        assert!(UrlValidator::validate("https://example.com#section").is_ok());
    }

    #[test]
    fn test_url_validation_with_auth() {
        assert!(UrlValidator::validate("https://user:pass@example.com").is_ok());
    }

    #[test]
    fn test_localhost_case_insensitive() {
        assert!(UrlValidator::is_localhost("http://LOCALHOST:8080"));
        assert!(UrlValidator::is_localhost("http://LocalHost:8080"));
    }

    #[test]
    fn test_localhost_in_path_not_matched() {
        // localhost in the path should not trigger localhost detection
        assert!(!UrlValidator::is_localhost(
            "https://example.com/localhost/api"
        ));
    }
}

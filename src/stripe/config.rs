//! Stripe Webhook Configuration
//!
//! CONS-003 COMPLIANT: All secrets loaded from environment variables.

use std::env;
use std::time::Duration;

use crate::stripe::error::{StripeWebhookError, StripeWebhookResult};

/// Configuration for Stripe webhook handling
#[derive(Debug, Clone)]
pub struct StripeWebhookConfig {
    /// Webhook signing secret (whsec_...)
    /// NEVER log this value
    webhook_secret: String,

    /// Maximum age for webhook timestamps (replay attack protection)
    /// Stripe recommends 5 minutes (300 seconds)
    pub max_timestamp_age: Duration,

    /// Maximum allowed clock drift (future timestamps)
    pub max_clock_drift: Duration,

    /// Idempotency TTL - how long to remember processed events
    pub idempotency_ttl: Duration,

    /// Maximum number of events to track for idempotency
    pub idempotency_max_entries: usize,

    /// Async processing timeout
    pub processing_timeout: Duration,

    /// Number of retry attempts for failed processing
    pub max_retries: u32,

    /// Base delay for exponential backoff (doubles each retry)
    pub retry_base_delay: Duration,

    /// Whether to log event payloads (DISABLE in production for PII)
    pub log_payloads: bool,
}

impl StripeWebhookConfig {
    /// Create configuration from environment variables
    ///
    /// # Environment Variables
    ///
    /// - `STRIPE_WEBHOOK_SECRET` (required): Webhook signing secret (whsec_...)
    /// - `STRIPE_WEBHOOK_MAX_AGE` (optional): Max timestamp age in seconds (default: 300)
    /// - `STRIPE_WEBHOOK_IDEMPOTENCY_TTL` (optional): Idempotency TTL in seconds (default: 86400)
    /// - `STRIPE_WEBHOOK_PROCESSING_TIMEOUT` (optional): Processing timeout in seconds (default: 30)
    /// - `STRIPE_WEBHOOK_MAX_RETRIES` (optional): Max retry attempts (default: 3)
    /// - `STRIPE_WEBHOOK_LOG_PAYLOADS` (optional): Log payloads - DISABLE IN PROD (default: false)
    ///
    /// # Errors
    ///
    /// Returns `StripeWebhookError::MissingSecret` if `STRIPE_WEBHOOK_SECRET` is not set.
    pub fn from_env() -> StripeWebhookResult<Self> {
        // CONS-003: Secret from environment variable only
        let webhook_secret =
            env::var("STRIPE_WEBHOOK_SECRET").map_err(|_| StripeWebhookError::MissingSecret)?;

        Self::validate_secret(&webhook_secret)?;

        let max_timestamp_age = env::var("STRIPE_WEBHOOK_MAX_AGE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(300)); // 5 minutes

        let idempotency_ttl = env::var("STRIPE_WEBHOOK_IDEMPOTENCY_TTL")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(86400)); // 24 hours

        let processing_timeout = env::var("STRIPE_WEBHOOK_PROCESSING_TIMEOUT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));

        let max_retries = env::var("STRIPE_WEBHOOK_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3);

        let log_payloads = env::var("STRIPE_WEBHOOK_LOG_PAYLOADS")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);

        Ok(Self {
            webhook_secret,
            max_timestamp_age,
            max_clock_drift: Duration::from_secs(60), // 1 minute future tolerance
            idempotency_ttl,
            idempotency_max_entries: 100_000,
            processing_timeout,
            max_retries,
            retry_base_delay: Duration::from_secs(1),
            log_payloads,
        })
    }

    /// Create a test configuration (for testing only)
    #[cfg(test)]
    pub fn test_config() -> Self {
        Self {
            webhook_secret: "whsec_test_secret_for_unit_tests_only_12345".to_string(),
            max_timestamp_age: Duration::from_secs(300),
            max_clock_drift: Duration::from_secs(60),
            idempotency_ttl: Duration::from_secs(3600),
            idempotency_max_entries: 1000,
            processing_timeout: Duration::from_secs(5),
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            log_payloads: true, // OK for tests
        }
    }

    /// Validate the webhook secret format
    fn validate_secret(secret: &str) -> StripeWebhookResult<()> {
        if secret.is_empty() {
            return Err(StripeWebhookError::InvalidSecretFormat(
                "Secret cannot be empty".to_string(),
            ));
        }

        // Stripe webhook secrets start with "whsec_"
        if !secret.starts_with("whsec_") {
            tracing::warn!("STRIPE_WEBHOOK_SECRET does not start with 'whsec_' - may be invalid");
        }

        // Minimum reasonable length
        if secret.len() < 20 {
            return Err(StripeWebhookError::InvalidSecretFormat(
                "Secret too short (minimum 20 characters)".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the webhook signing secret
    ///
    /// # Security Note
    ///
    /// This method returns a reference to the secret. NEVER log this value.
    pub(crate) fn webhook_secret(&self) -> &str {
        &self.webhook_secret
    }

    /// Calculate retry delay with exponential backoff and jitter
    pub fn retry_delay(&self, attempt: u32) -> Duration {
        let base = self.retry_base_delay.as_millis() as u64;
        let delay = base.saturating_mul(2u64.saturating_pow(attempt));

        // Add jitter (10-20% of delay)
        let jitter = delay / 10 + (rand::random::<u64>() % (delay / 10 + 1));
        Duration::from_millis(delay.saturating_add(jitter).min(30_000)) // Cap at 30s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        // Valid secret
        assert!(
            StripeWebhookConfig::validate_secret("whsec_test_secret_12345678901234567890").is_ok()
        );

        // Empty secret
        assert!(matches!(
            StripeWebhookConfig::validate_secret(""),
            Err(StripeWebhookError::InvalidSecretFormat(_))
        ));

        // Too short
        assert!(matches!(
            StripeWebhookConfig::validate_secret("short"),
            Err(StripeWebhookError::InvalidSecretFormat(_))
        ));
    }

    #[test]
    fn test_retry_delay() {
        let config = StripeWebhookConfig::test_config();

        let delay0 = config.retry_delay(0);
        let delay1 = config.retry_delay(1);
        let delay2 = config.retry_delay(2);

        // Each delay should roughly double
        assert!(delay1 > delay0);
        assert!(delay2 > delay1);

        // Should be capped at 30 seconds
        let delay_max = config.retry_delay(20);
        assert!(delay_max <= Duration::from_secs(30));
    }
}

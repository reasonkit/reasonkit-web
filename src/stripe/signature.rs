//! Stripe Webhook Signature Verification
//!
//! Implements HMAC-SHA256 signature verification for Stripe webhooks.
//! Uses constant-time comparison to prevent timing attacks.
//!
//! # Signature Format
//!
//! The `stripe-signature` header contains:
//! - `t`: Unix timestamp
//! - `v1`: HMAC-SHA256 signature (hex-encoded)
//!
//! Example: `t=1614556800,v1=abcdef123456...`
//!
//! # Verification Process
//!
//! 1. Parse the signature header
//! 2. Validate the timestamp is within acceptable range
//! 3. Construct the signed payload: `{timestamp}.{body}`
//! 4. Compute expected signature with HMAC-SHA256
//! 5. Compare signatures using constant-time comparison

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::stripe::config::StripeWebhookConfig;
use crate::stripe::error::{StripeWebhookError, StripeWebhookResult};

type HmacSha256 = Hmac<Sha256>;

/// Parsed stripe-signature header components
#[derive(Debug, Clone)]
pub struct ParsedSignature {
    /// Unix timestamp from header
    pub timestamp: i64,
    /// v1 signature (hex-encoded)
    pub signature: String,
}

/// Signature verifier for Stripe webhooks
#[derive(Clone)]
pub struct SignatureVerifier {
    /// HMAC key derived from webhook secret
    secret: Vec<u8>,
    /// Maximum allowed age for timestamps
    max_age: Duration,
    /// Maximum allowed clock drift
    max_drift: Duration,
}

impl SignatureVerifier {
    /// Create a new signature verifier from config
    pub fn new(config: &StripeWebhookConfig) -> Self {
        Self {
            secret: config.webhook_secret().as_bytes().to_vec(),
            max_age: config.max_timestamp_age,
            max_drift: config.max_clock_drift,
        }
    }

    /// Create a verifier for testing
    #[cfg(test)]
    pub fn test_verifier(secret: &str) -> Self {
        Self {
            secret: secret.as_bytes().to_vec(),
            max_age: Duration::from_secs(300),
            max_drift: Duration::from_secs(60),
        }
    }

    /// Verify the webhook signature
    ///
    /// # Arguments
    ///
    /// * `signature_header` - The `stripe-signature` header value
    /// * `payload` - The raw request body (must be exact bytes received)
    ///
    /// # Returns
    ///
    /// The parsed signature on success, or an error if verification fails.
    ///
    /// # Security
    ///
    /// - Uses constant-time comparison for signature matching
    /// - Validates timestamp to prevent replay attacks
    pub fn verify(
        &self,
        signature_header: &str,
        payload: &[u8],
    ) -> StripeWebhookResult<ParsedSignature> {
        // 1. Parse the signature header
        let parsed = self.parse_signature_header(signature_header)?;

        // 2. Validate timestamp
        self.validate_timestamp(parsed.timestamp)?;

        // 3. Construct signed payload: {timestamp}.{body}
        let signed_payload = self.construct_signed_payload(parsed.timestamp, payload);

        // 4. Compute expected signature
        let expected_signature = self.compute_signature(&signed_payload)?;

        // 5. Constant-time comparison
        if !constant_time_eq(&parsed.signature, &expected_signature) {
            tracing::warn!(
                timestamp = parsed.timestamp,
                "Stripe webhook signature verification failed"
            );
            return Err(StripeWebhookError::SignatureVerificationFailed);
        }

        tracing::debug!(
            timestamp = parsed.timestamp,
            "Stripe webhook signature verified successfully"
        );

        Ok(parsed)
    }

    /// Parse the stripe-signature header
    ///
    /// Format: `t=1614556800,v1=abcdef123456...`
    fn parse_signature_header(&self, header: &str) -> StripeWebhookResult<ParsedSignature> {
        let mut timestamp: Option<i64> = None;
        let mut signature: Option<String> = None;

        for part in header.split(',') {
            let part = part.trim();
            if let Some(value) = part.strip_prefix("t=") {
                timestamp = Some(value.parse::<i64>().map_err(|_| {
                    StripeWebhookError::InvalidSignatureFormat(
                        "Invalid timestamp format".to_string(),
                    )
                })?);
            } else if let Some(value) = part.strip_prefix("v1=") {
                signature = Some(value.to_string());
            }
            // Ignore other versions (v0, etc.) for forward compatibility
        }

        let timestamp = timestamp.ok_or_else(|| {
            StripeWebhookError::InvalidSignatureFormat("Missing timestamp (t=)".to_string())
        })?;

        let signature = signature.ok_or_else(|| {
            StripeWebhookError::InvalidSignatureFormat("Missing v1 signature".to_string())
        })?;

        // Validate signature is valid hex
        if !signature.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(StripeWebhookError::InvalidSignatureFormat(
                "Signature is not valid hex".to_string(),
            ));
        }

        if signature.len() != 64 {
            // SHA-256 produces 32 bytes = 64 hex chars
            return Err(StripeWebhookError::InvalidSignatureFormat(format!(
                "Invalid signature length: {} (expected 64)",
                signature.len()
            )));
        }

        Ok(ParsedSignature {
            timestamp,
            signature,
        })
    }

    /// Validate the timestamp is within acceptable range
    fn validate_timestamp(&self, timestamp: i64) -> StripeWebhookResult<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| StripeWebhookError::InternalError(e.to_string()))?
            .as_secs() as i64;

        let age = now - timestamp;

        // Check if timestamp is too old (replay attack)
        if age > self.max_age.as_secs() as i64 {
            tracing::warn!(
                timestamp,
                now,
                age_seconds = age,
                max_age_seconds = self.max_age.as_secs(),
                "Stripe webhook timestamp too old"
            );
            return Err(StripeWebhookError::TimestampTooOld {
                age_seconds: age,
                max_age_seconds: self.max_age.as_secs() as i64,
            });
        }

        // Check if timestamp is too far in the future (clock skew)
        if age < -(self.max_drift.as_secs() as i64) {
            tracing::warn!(
                timestamp,
                now,
                drift_seconds = -age,
                "Stripe webhook timestamp in future"
            );
            return Err(StripeWebhookError::TimestampInFuture {
                drift_seconds: -age,
            });
        }

        Ok(())
    }

    /// Construct the signed payload as Stripe does
    ///
    /// Format: `{timestamp}.{raw_body}`
    fn construct_signed_payload(&self, timestamp: i64, payload: &[u8]) -> Vec<u8> {
        let mut signed = format!("{}.", timestamp).into_bytes();
        signed.extend_from_slice(payload);
        signed
    }

    /// Compute the HMAC-SHA256 signature
    fn compute_signature(&self, payload: &[u8]) -> StripeWebhookResult<String> {
        let mut mac = HmacSha256::new_from_slice(&self.secret)
            .map_err(|e| StripeWebhookError::InternalError(e.to_string()))?;

        mac.update(payload);
        let result = mac.finalize();
        let signature_bytes = result.into_bytes();

        // Convert to lowercase hex
        Ok(hex::encode(signature_bytes))
    }

    /// Generate a valid signature for testing
    #[cfg(test)]
    pub fn generate_test_signature(&self, payload: &[u8], timestamp: i64) -> String {
        let signed_payload = self.construct_signed_payload(timestamp, payload);
        self.compute_signature(&signed_payload).unwrap()
    }

    /// Generate a valid signature (available in handler module for tests)
    #[doc(hidden)]
    pub fn generate_test_signature_public(&self, payload: &[u8], timestamp: i64) -> String {
        let signed_payload = self.construct_signed_payload(timestamp, payload);
        self.compute_signature(&signed_payload).unwrap()
    }
}

/// Constant-time string comparison to prevent timing attacks
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        result |= x ^ y;
    }
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    #[test]
    fn test_parse_signature_header() {
        let verifier = SignatureVerifier::test_verifier("whsec_test");

        let header =
            "t=1614556800,v1=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";
        let parsed = verifier.parse_signature_header(header).unwrap();

        assert_eq!(parsed.timestamp, 1614556800);
        assert_eq!(
            parsed.signature,
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        );
    }

    #[test]
    fn test_parse_signature_header_missing_timestamp() {
        let verifier = SignatureVerifier::test_verifier("whsec_test");

        let header = "v1=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";
        let result = verifier.parse_signature_header(header);

        assert!(matches!(
            result,
            Err(StripeWebhookError::InvalidSignatureFormat(_))
        ));
    }

    #[test]
    fn test_parse_signature_header_missing_signature() {
        let verifier = SignatureVerifier::test_verifier("whsec_test");

        let header = "t=1614556800";
        let result = verifier.parse_signature_header(header);

        assert!(matches!(
            result,
            Err(StripeWebhookError::InvalidSignatureFormat(_))
        ));
    }

    #[test]
    fn test_verify_valid_signature() {
        let secret = "whsec_test_secret_12345";
        let verifier = SignatureVerifier::test_verifier(secret);

        let payload = b"{\"type\":\"test\"}";
        let timestamp = current_timestamp();

        // Generate a valid signature
        let signature = verifier.generate_test_signature(payload, timestamp);
        let header = format!("t={},v1={}", timestamp, signature);

        // Verify should succeed
        let result = verifier.verify(&header, payload);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_invalid_signature() {
        let verifier = SignatureVerifier::test_verifier("whsec_test_secret");

        let payload = b"{\"type\":\"test\"}";
        let timestamp = current_timestamp();

        // Use a wrong signature
        let header = format!(
            "t={},v1=0000000000000000000000000000000000000000000000000000000000000000",
            timestamp
        );

        let result = verifier.verify(&header, payload);
        assert!(matches!(
            result,
            Err(StripeWebhookError::SignatureVerificationFailed)
        ));
    }

    #[test]
    fn test_verify_old_timestamp() {
        let verifier = SignatureVerifier::test_verifier("whsec_test_secret");

        let payload = b"{\"type\":\"test\"}";
        let old_timestamp = current_timestamp() - 600; // 10 minutes ago

        let signature = verifier.generate_test_signature(payload, old_timestamp);
        let header = format!("t={},v1={}", old_timestamp, signature);

        let result = verifier.verify(&header, payload);
        assert!(matches!(
            result,
            Err(StripeWebhookError::TimestampTooOld { .. })
        ));
    }

    #[test]
    fn test_verify_future_timestamp() {
        let verifier = SignatureVerifier::test_verifier("whsec_test_secret");

        let payload = b"{\"type\":\"test\"}";
        let future_timestamp = current_timestamp() + 120; // 2 minutes in future

        let signature = verifier.generate_test_signature(payload, future_timestamp);
        let header = format!("t={},v1={}", future_timestamp, signature);

        let result = verifier.verify(&header, payload);
        assert!(matches!(
            result,
            Err(StripeWebhookError::TimestampInFuture { .. })
        ));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq("abc123", "abc123"));
        assert!(!constant_time_eq("abc123", "abc124"));
        assert!(!constant_time_eq("abc", "abcd"));
        assert!(!constant_time_eq("", "a"));
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn test_signature_with_modified_payload() {
        let verifier = SignatureVerifier::test_verifier("whsec_test_secret");

        let original_payload = b"{\"type\":\"test\"}";
        let timestamp = current_timestamp();

        // Generate signature for original payload
        let signature = verifier.generate_test_signature(original_payload, timestamp);
        let header = format!("t={},v1={}", timestamp, signature);

        // Try to verify with modified payload
        let modified_payload = b"{\"type\":\"hacked\"}";
        let result = verifier.verify(&header, modified_payload);

        assert!(matches!(
            result,
            Err(StripeWebhookError::SignatureVerificationFailed)
        ));
    }
}

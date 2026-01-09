//! Stripe Webhook Error Types
//!
//! Comprehensive error handling for webhook processing with proper HTTP status mapping.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

/// Stripe webhook error types with HTTP status code mapping
#[derive(Error, Debug)]
pub enum StripeWebhookError {
    // =========================================================================
    // Configuration Errors (500)
    // =========================================================================
    /// Missing webhook signing secret
    #[error("STRIPE_WEBHOOK_SECRET environment variable not set")]
    MissingSecret,

    /// Invalid secret format
    #[error("Invalid webhook secret format: {0}")]
    InvalidSecretFormat(String),

    // =========================================================================
    // Signature Errors (400/401)
    // =========================================================================
    /// Missing stripe-signature header
    #[error("Missing stripe-signature header")]
    MissingSignature,

    /// Invalid signature format
    #[error("Invalid signature format: {0}")]
    InvalidSignatureFormat(String),

    /// Signature verification failed
    #[error("Signature verification failed")]
    SignatureVerificationFailed,

    /// Timestamp too old (replay attack protection)
    #[error("Webhook timestamp too old: {age_seconds}s (max: {max_age_seconds}s)")]
    TimestampTooOld {
        age_seconds: i64,
        max_age_seconds: i64,
    },

    /// Timestamp in future (clock skew)
    #[error("Webhook timestamp in future by {drift_seconds}s")]
    TimestampInFuture { drift_seconds: i64 },

    // =========================================================================
    // Payload Errors (400)
    // =========================================================================
    /// Failed to parse request body
    #[error("Failed to parse request body: {0}")]
    InvalidPayload(String),

    /// Unknown event type
    #[error("Unknown event type: {0}")]
    UnknownEventType(String),

    /// Missing required field in event
    #[error("Missing required field: {0}")]
    MissingField(String),

    // =========================================================================
    // Idempotency (202 - not an error, but handled specially)
    // =========================================================================
    /// Event already processed (idempotent)
    #[error("Event {event_id} already processed")]
    AlreadyProcessed { event_id: String },

    // =========================================================================
    // Processing Errors (500)
    // =========================================================================
    /// Event processing failed
    #[error("Event processing failed: {0}")]
    ProcessingFailed(String),

    /// Database error during processing
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// External service error
    #[error("External service error: {0}")]
    ExternalServiceError(String),

    // =========================================================================
    // Internal Errors (500)
    // =========================================================================
    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for Stripe webhook operations
pub type StripeWebhookResult<T> = std::result::Result<T, StripeWebhookError>;

impl StripeWebhookError {
    /// Get the HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            // 400 Bad Request - client sent malformed data
            Self::InvalidPayload(_)
            | Self::InvalidSignatureFormat(_)
            | Self::UnknownEventType(_)
            | Self::MissingField(_) => StatusCode::BAD_REQUEST,

            // 401 Unauthorized - signature verification failed
            Self::MissingSignature
            | Self::SignatureVerificationFailed
            | Self::TimestampTooOld { .. }
            | Self::TimestampInFuture { .. } => StatusCode::UNAUTHORIZED,

            // 202 Accepted - already processed (idempotent success)
            Self::AlreadyProcessed { .. } => StatusCode::ACCEPTED,

            // 500 Internal Server Error - our fault
            Self::MissingSecret
            | Self::InvalidSecretFormat(_)
            | Self::ProcessingFailed(_)
            | Self::DatabaseError(_)
            | Self::ExternalServiceError(_)
            | Self::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Check if this error should be retried by Stripe
    ///
    /// Stripe will retry webhooks that return 5xx errors.
    /// We return false for client errors (4xx) to prevent retries.
    pub fn should_retry(&self) -> bool {
        matches!(
            self.status_code(),
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE
        )
    }

    /// Get error code for logging/metrics
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::MissingSecret => "MISSING_SECRET",
            Self::InvalidSecretFormat(_) => "INVALID_SECRET_FORMAT",
            Self::MissingSignature => "MISSING_SIGNATURE",
            Self::InvalidSignatureFormat(_) => "INVALID_SIGNATURE_FORMAT",
            Self::SignatureVerificationFailed => "SIGNATURE_VERIFICATION_FAILED",
            Self::TimestampTooOld { .. } => "TIMESTAMP_TOO_OLD",
            Self::TimestampInFuture { .. } => "TIMESTAMP_IN_FUTURE",
            Self::InvalidPayload(_) => "INVALID_PAYLOAD",
            Self::UnknownEventType(_) => "UNKNOWN_EVENT_TYPE",
            Self::MissingField(_) => "MISSING_FIELD",
            Self::AlreadyProcessed { .. } => "ALREADY_PROCESSED",
            Self::ProcessingFailed(_) => "PROCESSING_FAILED",
            Self::DatabaseError(_) => "DATABASE_ERROR",
            Self::ExternalServiceError(_) => "EXTERNAL_SERVICE_ERROR",
            Self::InternalError(_) => "INTERNAL_ERROR",
        }
    }
}

/// Error response body for API clients
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetails {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,
}

impl IntoResponse for StripeWebhookError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_code = self.error_code().to_string();

        // For security, don't expose internal error details
        let message = match &self {
            // Safe to expose
            Self::MissingSignature => self.to_string(),
            Self::InvalidSignatureFormat(_) => "Invalid signature format".to_string(),
            Self::SignatureVerificationFailed => self.to_string(),
            Self::TimestampTooOld { .. } | Self::TimestampInFuture { .. } => {
                "Webhook timestamp validation failed".to_string()
            }
            Self::InvalidPayload(_) => "Invalid request payload".to_string(),
            Self::UnknownEventType(t) => format!("Unknown event type: {}", t),
            Self::MissingField(f) => format!("Missing required field: {}", f),
            Self::AlreadyProcessed { event_id } => {
                format!("Event {} already processed", event_id)
            }
            // Internal errors - generic message
            Self::MissingSecret
            | Self::InvalidSecretFormat(_)
            | Self::ProcessingFailed(_)
            | Self::DatabaseError(_)
            | Self::ExternalServiceError(_)
            | Self::InternalError(_) => "Internal server error".to_string(),
        };

        let body = ErrorResponse {
            error: ErrorDetails {
                code: error_code,
                message,
                retry_after: if self.should_retry() {
                    Some(60) // Suggest retry after 60 seconds
                } else {
                    None
                },
            },
        };

        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            StripeWebhookError::MissingSignature.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            StripeWebhookError::InvalidPayload("test".to_string()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            StripeWebhookError::AlreadyProcessed {
                event_id: "evt_123".to_string()
            }
            .status_code(),
            StatusCode::ACCEPTED
        );
        assert_eq!(
            StripeWebhookError::ProcessingFailed("test".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_should_retry() {
        assert!(!StripeWebhookError::MissingSignature.should_retry());
        assert!(!StripeWebhookError::InvalidPayload("test".to_string()).should_retry());
        assert!(StripeWebhookError::ProcessingFailed("test".to_string()).should_retry());
        assert!(StripeWebhookError::DatabaseError("test".to_string()).should_retry());
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(
            StripeWebhookError::MissingSignature.error_code(),
            "MISSING_SIGNATURE"
        );
        assert_eq!(
            StripeWebhookError::SignatureVerificationFailed.error_code(),
            "SIGNATURE_VERIFICATION_FAILED"
        );
    }
}

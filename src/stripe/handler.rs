//! Axum Handler and Router for Stripe Webhooks
//!
//! This module provides the HTTP layer for receiving and processing Stripe webhooks.
//!
//! # Key Features
//!
//! - **Raw Body Extraction**: Captures the exact bytes for signature verification
//! - **Quick Acknowledgment**: Returns 200/202 immediately, processes asynchronously
//! - **Proper Error Responses**: Returns appropriate HTTP status codes
//!
//! # Endpoint
//!
//! `POST /webhooks/stripe`
//!
//! # Headers
//!
//! Required:
//! - `stripe-signature`: The Stripe webhook signature
//! - `Content-Type: application/json`

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::Serialize;

use crate::stripe::config::StripeWebhookConfig;
use crate::stripe::error::StripeWebhookError;
use crate::stripe::events::StripeEvent;
use crate::stripe::idempotency::{
    IdempotencyMiddleware, IdempotencyStore, InMemoryIdempotencyStore,
};
use crate::stripe::processor::{EventProcessor, SubscriptionHandler};
use crate::stripe::signature::SignatureVerifier;

/// Shared state for the webhook handler
pub struct StripeWebhookState<
    H: SubscriptionHandler,
    S: IdempotencyStore = InMemoryIdempotencyStore,
> {
    /// Signature verifier
    pub verifier: SignatureVerifier,
    /// Idempotency middleware
    pub idempotency: IdempotencyMiddleware<S>,
    /// Event processor
    pub processor: EventProcessor<H, S>,
    /// Configuration
    pub config: StripeWebhookConfig,
    /// Whether to log payloads (DISABLE in production)
    pub log_payloads: bool,
}

impl<H: SubscriptionHandler> StripeWebhookState<H, InMemoryIdempotencyStore> {
    /// Create new state with in-memory idempotency store
    pub fn new(
        config: StripeWebhookConfig,
        handler: Arc<H>,
    ) -> (
        Self,
        crate::stripe::processor::ProcessorHandle<H, InMemoryIdempotencyStore>,
    ) {
        let store = Arc::new(InMemoryIdempotencyStore::from_config(&config));
        Self::with_store(config, handler, store)
    }
}

impl<H: SubscriptionHandler, S: IdempotencyStore> StripeWebhookState<H, S> {
    /// Create new state with custom idempotency store
    pub fn with_store(
        config: StripeWebhookConfig,
        handler: Arc<H>,
        store: Arc<S>,
    ) -> (Self, crate::stripe::processor::ProcessorHandle<H, S>) {
        let verifier = SignatureVerifier::new(&config);
        let idempotency = IdempotencyMiddleware::new(store.clone());
        let (processor, handle) = EventProcessor::new(handler, store, config.clone());
        let log_payloads = config.log_payloads;

        let state = Self {
            verifier,
            idempotency,
            processor,
            config,
            log_payloads,
        };

        (state, handle)
    }
}

/// Success response for webhook acknowledgment
#[derive(Debug, Clone, Serialize)]
pub struct WebhookResponse {
    pub received: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl WebhookResponse {
    fn success(event_id: &str) -> Self {
        Self {
            received: true,
            event_id: Some(event_id.to_string()),
            message: None,
        }
    }

    fn already_processed(event_id: &str) -> Self {
        Self {
            received: true,
            event_id: Some(event_id.to_string()),
            message: Some("Event already processed".to_string()),
        }
    }
}

/// Create the Stripe webhook router
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_web::stripe::{StripeWebhookConfig, StripeWebhookState, stripe_webhook_router};
/// use std::sync::Arc;
///
/// let config = StripeWebhookConfig::from_env()?;
/// let handler = Arc::new(MyHandler);
/// let (state, processor_handle) = StripeWebhookState::new(config, handler);
///
/// // Start the background processor
/// tokio::spawn(async move {
///     processor_handle.run().await;
/// });
///
/// // Create the router
/// let app = stripe_webhook_router(Arc::new(state));
/// ```
pub fn stripe_webhook_router<H: SubscriptionHandler, S: IdempotencyStore>(
    state: Arc<StripeWebhookState<H, S>>,
) -> Router {
    Router::new()
        .route("/webhooks/stripe", post(stripe_webhook_handler))
        .with_state(state)
}

/// Main webhook handler
///
/// Processes incoming Stripe webhook requests:
///
/// 1. Extracts the `stripe-signature` header
/// 2. Verifies the signature using HMAC-SHA256
/// 3. Checks idempotency (have we seen this event?)
/// 4. Queues the event for async processing
/// 5. Returns 200 (new event) or 202 (already processed)
pub async fn stripe_webhook_handler<H: SubscriptionHandler, S: IdempotencyStore>(
    State(state): State<Arc<StripeWebhookState<H, S>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Extract signature header
    let signature = match headers.get("stripe-signature") {
        Some(sig) => match sig.to_str() {
            Ok(s) => s,
            Err(_) => {
                return StripeWebhookError::InvalidSignatureFormat(
                    "Invalid header encoding".to_string(),
                )
                .into_response();
            }
        },
        None => {
            return StripeWebhookError::MissingSignature.into_response();
        }
    };

    // Verify signature
    if let Err(e) = state.verifier.verify(signature, &body) {
        tracing::warn!(error = %e, "Stripe webhook signature verification failed");
        return e.into_response();
    }

    // Parse the event
    let event = match StripeEvent::from_bytes(&body) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to parse Stripe webhook payload");
            return e.into_response();
        }
    };

    // Log the event (if enabled - DISABLE in production for PII)
    if state.log_payloads {
        tracing::debug!(
            event_id = %event.id,
            event_type = %event.event_type,
            livemode = event.livemode,
            "Received Stripe webhook"
        );
    } else {
        tracing::info!(
            event_id = %event.id,
            event_type = %event.event_type,
            "Received Stripe webhook"
        );
    }

    // Check idempotency
    match state.idempotency.should_process(&event.id).await {
        Ok(true) => {
            // New event - queue for processing and return 200
            if let Err(e) = state.processor.queue_event(event.clone()).await {
                tracing::error!(
                    event_id = %event.id,
                    error = %e,
                    "Failed to queue event for processing"
                );
                // Still return 200 - we recorded it, will retry on next delivery
            }

            (StatusCode::OK, Json(WebhookResponse::success(&event.id))).into_response()
        }
        Ok(false) => {
            // Already in progress (should not happen with our implementation)
            (
                StatusCode::ACCEPTED,
                Json(WebhookResponse::already_processed(&event.id)),
            )
                .into_response()
        }
        Err(StripeWebhookError::AlreadyProcessed { event_id }) => {
            // Already processed - return 202 to acknowledge without reprocessing
            tracing::debug!(
                event_id = %event_id,
                "Stripe webhook already processed (idempotent)"
            );
            (
                StatusCode::ACCEPTED,
                Json(WebhookResponse::already_processed(&event_id)),
            )
                .into_response()
        }
        Err(e) => {
            // Other error
            tracing::error!(
                event_id = %event.id,
                error = %e,
                "Idempotency check failed"
            );
            e.into_response()
        }
    }
}

/// Health check endpoint for the webhook handler
pub async fn webhook_health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "healthy",
            "service": "stripe-webhook-handler"
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stripe::processor::NoOpHandler;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    fn create_test_state() -> (
        Arc<StripeWebhookState<NoOpHandler, InMemoryIdempotencyStore>>,
        crate::stripe::processor::ProcessorHandle<NoOpHandler, InMemoryIdempotencyStore>,
    ) {
        let config = StripeWebhookConfig::test_config();
        let handler = Arc::new(NoOpHandler);
        let (state, handle) = StripeWebhookState::new(config, handler);
        (Arc::new(state), handle)
    }

    fn create_valid_webhook_request(
        state: &StripeWebhookState<NoOpHandler, InMemoryIdempotencyStore>,
    ) -> (String, Vec<u8>) {
        let payload = r#"{
            "id": "evt_test_handler_123",
            "type": "customer.subscription.created",
            "created": 1614556800,
            "livemode": false,
            "pending_webhooks": 1,
            "data": {
                "object": {
                    "id": "sub_test_123",
                    "customer": "cus_test_123",
                    "status": "active",
                    "current_period_start": 1614556800,
                    "current_period_end": 1617235200,
                    "cancel_at_period_end": false,
                    "items": {
                        "data": [{
                            "id": "si_test_123",
                            "price": {
                                "id": "price_test_123",
                                "product": "prod_test_123",
                                "unit_amount": 2000,
                                "currency": "usd",
                                "recurring": {
                                    "interval": "month",
                                    "interval_count": 1
                                }
                            },
                            "quantity": 1
                        }]
                    },
                    "metadata": {},
                    "livemode": false
                }
            }
        }"#;

        let timestamp = current_timestamp();
        let signature = state
            .verifier
            .generate_test_signature_public(payload.as_bytes(), timestamp);
        let header = format!("t={},v1={}", timestamp, signature);

        (header, payload.as_bytes().to_vec())
    }

    #[tokio::test]
    async fn test_webhook_handler_success() {
        let (state, _handle) = create_test_state();
        let app = stripe_webhook_router(state.clone());

        let (signature, payload) = create_valid_webhook_request(&state);

        let request = Request::builder()
            .method(Method::POST)
            .uri("/webhooks/stripe")
            .header("content-type", "application/json")
            .header("stripe-signature", signature)
            .body(Body::from(payload))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_webhook_handler_missing_signature() {
        let (state, _handle) = create_test_state();
        let app = stripe_webhook_router(state);

        let request = Request::builder()
            .method(Method::POST)
            .uri("/webhooks/stripe")
            .header("content-type", "application/json")
            // No stripe-signature header
            .body(Body::from(r#"{"id": "test"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_webhook_handler_invalid_signature() {
        let (state, _handle) = create_test_state();
        let app = stripe_webhook_router(state);

        let timestamp = current_timestamp();
        let invalid_signature = format!(
            "t={},v1=0000000000000000000000000000000000000000000000000000000000000000",
            timestamp
        );

        let request = Request::builder()
            .method(Method::POST)
            .uri("/webhooks/stripe")
            .header("content-type", "application/json")
            .header("stripe-signature", invalid_signature)
            .body(Body::from(r#"{"id": "test"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_webhook_handler_invalid_payload() {
        let (state, _handle) = create_test_state();
        let app = stripe_webhook_router(state.clone());

        let payload = b"not valid json";
        let timestamp = current_timestamp();
        let signature = state
            .verifier
            .generate_test_signature_public(payload, timestamp);
        let header = format!("t={},v1={}", timestamp, signature);

        let request = Request::builder()
            .method(Method::POST)
            .uri("/webhooks/stripe")
            .header("content-type", "application/json")
            .header("stripe-signature", header)
            .body(Body::from(payload.to_vec()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_webhook_handler_idempotency() {
        let (state, _handle) = create_test_state();

        // Use unique event IDs for each test run
        let unique_id = format!("evt_idempotency_test_{}", current_timestamp());
        let payload = format!(
            r#"{{
            "id": "{}",
            "type": "customer.subscription.created",
            "created": 1614556800,
            "livemode": false,
            "pending_webhooks": 1,
            "data": {{
                "object": {{
                    "id": "sub_test_123",
                    "customer": "cus_test_123",
                    "status": "active",
                    "current_period_start": 1614556800,
                    "current_period_end": 1617235200,
                    "cancel_at_period_end": false,
                    "items": {{
                        "data": [{{
                            "id": "si_test_123",
                            "price": {{
                                "id": "price_test_123",
                                "product": "prod_test_123",
                                "unit_amount": 2000,
                                "currency": "usd",
                                "recurring": {{
                                    "interval": "month",
                                    "interval_count": 1
                                }}
                            }},
                            "quantity": 1
                        }}]
                    }},
                    "metadata": {{}},
                    "livemode": false
                }}
            }}
        }}"#,
            unique_id
        );

        let timestamp = current_timestamp();
        let signature = state
            .verifier
            .generate_test_signature_public(payload.as_bytes(), timestamp);
        let header = format!("t={},v1={}", timestamp, signature);

        // First request - should return 200
        let app1 = stripe_webhook_router(state.clone());
        let request1 = Request::builder()
            .method(Method::POST)
            .uri("/webhooks/stripe")
            .header("content-type", "application/json")
            .header("stripe-signature", header.clone())
            .body(Body::from(payload.clone()))
            .unwrap();

        let response1 = app1.oneshot(request1).await.unwrap();
        assert_eq!(response1.status(), StatusCode::OK);

        // Second request with same event - should return 202
        let app2 = stripe_webhook_router(state.clone());
        let request2 = Request::builder()
            .method(Method::POST)
            .uri("/webhooks/stripe")
            .header("content-type", "application/json")
            .header("stripe-signature", header)
            .body(Body::from(payload))
            .unwrap();

        let response2 = app2.oneshot(request2).await.unwrap();
        assert_eq!(response2.status(), StatusCode::ACCEPTED);
    }
}

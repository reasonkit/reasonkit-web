//! Stripe Webhook Handler Module
//!
//! This module provides secure, production-ready Stripe webhook handling for SaaS
//! subscription services. It implements:
//!
//! - **Signature Verification**: HMAC-SHA256 validation of the `stripe-signature` header
//! - **Idempotency**: Deduplication of webhook deliveries using event IDs
//! - **Async Processing**: Non-blocking webhook handling with background task execution
//! - **Event Handling**: Support for subscription, invoice, and customer events
//! - **Error Recovery**: Configurable retry logic with exponential backoff
//!
//! # Architecture
//!
//! ```text
//! Request -> Signature Verify -> Idempotency Check -> Ack (200) -> Async Process
//!                   |                    |                              |
//!                   v                    v                              v
//!              400/401              202 (already)              Background Task
//! ```
//!
//! # Security
//!
//! - CONS-003 COMPLIANT: Webhook signing secret loaded from environment
//! - Constant-time signature comparison to prevent timing attacks
//! - Raw body parsing to ensure signature verification works correctly
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit_web::stripe::{
//!     StripeWebhookConfig, StripeWebhookState, stripe_webhook_router,
//!     EventProcessor, SubscriptionHandler,
//! };
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = StripeWebhookConfig::from_env()?;
//!     let processor = Arc::new(MySubscriptionHandler);
//!     let state = StripeWebhookState::new(config, processor);
//!
//!     let app = stripe_webhook_router(Arc::new(state));
//!     // ... serve with axum
//!     Ok(())
//! }
//!
//! struct MySubscriptionHandler;
//!
//! #[async_trait::async_trait]
//! impl SubscriptionHandler for MySubscriptionHandler {
//!     async fn on_subscription_created(&self, subscription: &Subscription) -> anyhow::Result<()> {
//!         // Handle new subscription
//!         Ok(())
//!     }
//!     // ... other handlers
//! }
//! ```

pub mod config;
pub mod error;
pub mod events;
pub mod handler;
pub mod idempotency;
pub mod processor;
pub mod signature;

// Re-export commonly used items
pub use config::StripeWebhookConfig;
pub use error::{StripeWebhookError, StripeWebhookResult};
pub use events::{
    CustomerEvent, InvoiceEvent, StripeEvent, StripeEventType, SubscriptionEvent,
    SubscriptionStatus,
};
pub use handler::{stripe_webhook_handler, stripe_webhook_router, StripeWebhookState};
pub use idempotency::{IdempotencyStore, InMemoryIdempotencyStore};
pub use processor::{EventProcessor, SubscriptionHandler};
pub use signature::SignatureVerifier;

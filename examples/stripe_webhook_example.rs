//! Complete Stripe Webhook Handler Example
//!
//! This example demonstrates how to set up a production-ready Stripe webhook
//! handler for a SaaS subscription service using Axum.
//!
//! # Features Demonstrated
//!
//! - Signature verification (HMAC-SHA256)
//! - Idempotency (duplicate detection)
//! - Async processing (non-blocking webhook response)
//! - Error handling and retry logic
//! - Subscription lifecycle management
//!
//! # Running the Example
//!
//! ```bash
//! STRIPE_WEBHOOK_SECRET=whsec_your_secret cargo run --example stripe_webhook_example
//! ```
//!
//! # Testing with Stripe CLI
//!
//! ```bash
//! stripe listen --forward-to localhost:3000/webhooks/stripe
//! stripe trigger customer.subscription.created
//! ```

use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use reasonkit_web::stripe::{
    stripe_webhook_router, CustomerEvent, InvoiceEvent, StripeWebhookConfig, StripeWebhookState,
    SubscriptionEvent, SubscriptionHandler,
};

// =============================================================================
// Application Handler Implementation
// =============================================================================

/// Your application's subscription handler
///
/// This is where you implement your business logic for handling
/// subscription events. Each method corresponds to a Stripe event type.
struct MySubscriptionHandler {
    // In a real app, you'd have database connections, service clients, etc.
    // db: Arc<DatabasePool>,
    // email_service: Arc<EmailService>,
    // billing_service: Arc<BillingService>,
}

impl MySubscriptionHandler {
    fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl SubscriptionHandler for MySubscriptionHandler {
    /// Handle new subscription created
    ///
    /// Called when a customer completes checkout and a new subscription starts.
    async fn on_subscription_created(&self, event: &SubscriptionEvent) -> Result<()> {
        let sub = &event.subscription;

        tracing::info!(
            subscription_id = %sub.id,
            customer_id = %sub.customer,
            status = ?sub.status,
            "New subscription created"
        );

        // In a real app:
        // 1. Create user account if not exists
        // 2. Provision resources for the plan
        // 3. Send welcome email
        // 4. Update billing records

        // Example: Extract plan from subscription items
        if let Some(item) = sub.items.data.first() {
            let price_id = &item.price.id;
            let product_id = &item.price.product;

            tracing::info!(
                price_id = %price_id,
                product_id = %product_id,
                quantity = item.quantity,
                "Subscription plan details"
            );

            // self.db.create_subscription_record(sub).await?;
            // self.provision_plan_resources(product_id).await?;
        }

        Ok(())
    }

    /// Handle subscription updated
    ///
    /// Called when a subscription changes - plan upgrade/downgrade,
    /// status change, cancellation scheduled, etc.
    async fn on_subscription_updated(&self, event: &SubscriptionEvent) -> Result<()> {
        let sub = &event.subscription;

        tracing::info!(
            subscription_id = %sub.id,
            customer_id = %sub.customer,
            status = ?sub.status,
            cancel_at_period_end = sub.cancel_at_period_end,
            "Subscription updated"
        );

        // Check what changed using previous_attributes
        if let Some(prev) = &event.previous_attributes {
            if let Some(prev_status) = prev.get("status").and_then(|v| v.as_str()) {
                tracing::info!(
                    previous_status = prev_status,
                    new_status = ?sub.status,
                    "Subscription status changed"
                );
            }
        }

        // Handle cancellation scheduling
        if sub.cancel_at_period_end {
            tracing::warn!(
                subscription_id = %sub.id,
                period_end = sub.current_period_end,
                "Subscription scheduled for cancellation"
            );
            // self.email_service.send_retention_email(sub.customer).await?;
        }

        // Handle plan changes
        if sub.status.is_active() {
            // self.update_user_entitlements(sub).await?;
        }

        Ok(())
    }

    /// Handle subscription deleted/canceled
    ///
    /// Called when a subscription actually ends (not just scheduled).
    async fn on_subscription_deleted(&self, event: &SubscriptionEvent) -> Result<()> {
        let sub = &event.subscription;

        tracing::warn!(
            subscription_id = %sub.id,
            customer_id = %sub.customer,
            ended_at = ?sub.ended_at,
            "Subscription deleted"
        );

        // In a real app:
        // 1. Revoke access/entitlements
        // 2. Archive user data (don't delete immediately!)
        // 3. Send cancellation confirmation email
        // 4. Update analytics/churn tracking

        // self.db.mark_subscription_cancelled(sub.id).await?;
        // self.revoke_user_access(sub.customer).await?;
        // self.email_service.send_cancellation_email(sub.customer).await?;

        Ok(())
    }

    /// Handle successful payment
    ///
    /// Called when an invoice is paid successfully.
    async fn on_payment_succeeded(&self, event: &InvoiceEvent) -> Result<()> {
        let invoice = &event.invoice;

        tracing::info!(
            invoice_id = %invoice.id,
            customer_id = %invoice.customer,
            amount = invoice.amount_paid,
            currency = %invoice.currency,
            "Payment succeeded"
        );

        // In a real app:
        // 1. Update billing records
        // 2. Send receipt email
        // 3. Extend subscription period
        // 4. Update MRR/revenue metrics

        // self.db.record_payment(invoice).await?;
        // self.email_service.send_receipt(invoice).await?;
        // self.metrics.record_revenue(invoice.amount_paid).await?;

        Ok(())
    }

    /// Handle failed payment
    ///
    /// Called when a payment attempt fails. Critical for dunning management.
    async fn on_payment_failed(&self, event: &InvoiceEvent) -> Result<()> {
        let invoice = &event.invoice;

        tracing::error!(
            invoice_id = %invoice.id,
            customer_id = %invoice.customer,
            amount_due = invoice.amount_due,
            billing_reason = ?invoice.billing_reason,
            "Payment failed - REQUIRES ATTENTION"
        );

        // In a real app:
        // 1. Send payment failure notification
        // 2. Trigger dunning sequence
        // 3. Maybe pause service (depends on policy)
        // 4. Alert customer success team

        // self.email_service.send_payment_failed_email(invoice).await?;
        // self.dunning_service.start_sequence(invoice.customer).await?;
        // self.alerting.notify_cs_team(invoice).await?;

        // Include hosted invoice URL for easy retry
        if let Some(url) = &invoice.hosted_invoice_url {
            tracing::info!(
                invoice_url = %url,
                "Customer can retry payment at this URL"
            );
        }

        Ok(())
    }

    /// Handle new customer created
    async fn on_customer_created(&self, event: &CustomerEvent) -> Result<()> {
        let customer = &event.customer;

        tracing::info!(
            customer_id = %customer.id,
            email = ?customer.email,
            "New customer created"
        );

        // In a real app:
        // 1. Create user account
        // 2. Link Stripe customer to user
        // 3. Send welcome sequence

        // self.db.create_user_from_customer(customer).await?;

        Ok(())
    }
}

// =============================================================================
// Main Application
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "stripe_webhook_example=debug,reasonkit_web=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration from environment
    // STRIPE_WEBHOOK_SECRET must be set
    let config = StripeWebhookConfig::from_env()
        .expect("STRIPE_WEBHOOK_SECRET environment variable must be set");

    tracing::info!("Stripe webhook configuration loaded");

    // Create your handler
    let handler = Arc::new(MySubscriptionHandler::new());

    // Create webhook state and get the background processor handle
    let (state, processor_handle) = StripeWebhookState::new(config, handler);
    let state = Arc::new(state);

    // Spawn the background processor
    // This processes events asynchronously after the webhook returns 200
    tokio::spawn(async move {
        processor_handle.run().await;
    });
    tracing::info!("Background event processor started");

    // Build the router
    let app = Router::new()
        // Stripe webhook endpoint
        .merge(stripe_webhook_router(state.clone()))
        // You can add other routes here
        .route("/health", axum::routing::get(|| async { "OK" }));

    // Start the server
    let addr = "0.0.0.0:3000";
    let listener = TcpListener::bind(addr).await?;

    tracing::info!("Stripe webhook server listening on {}", addr);
    tracing::info!("Webhook endpoint: POST {}/webhooks/stripe", addr);
    tracing::info!("");
    tracing::info!("Test with Stripe CLI:");
    tracing::info!("  stripe listen --forward-to localhost:3000/webhooks/stripe");
    tracing::info!("  stripe trigger customer.subscription.created");

    axum::serve(listener, app).await?;

    Ok(())
}

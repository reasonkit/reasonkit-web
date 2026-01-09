//! Async Event Processing
//!
//! Handles background processing of webhook events with retry logic.
//! The key design principle: acknowledge the webhook quickly (return 200),
//! then process the event asynchronously.
//!
//! # Architecture
//!
//! ```text
//! Webhook Received
//!       |
//!       v
//! [Verify Signature]
//!       |
//!       v
//! [Check Idempotency] --> Already processed? --> Return 202
//!       |
//!       v
//! [Spawn Background Task] --> Return 200 immediately
//!       |
//!       v
//! [Process Event with Retries]
//!       |
//!       v
//! [Update Idempotency Store]
//! ```

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::stripe::config::StripeWebhookConfig;
use crate::stripe::error::StripeWebhookResult;
use crate::stripe::events::{
    CustomerEvent, InvoiceEvent, StripeEvent, StripeEventType, SubscriptionEvent,
};
use crate::stripe::idempotency::IdempotencyStore;

/// Handler trait for subscription events
#[async_trait::async_trait]
pub trait SubscriptionHandler: Send + Sync + 'static {
    /// Handle new subscription created
    async fn on_subscription_created(&self, event: &SubscriptionEvent) -> anyhow::Result<()>;

    /// Handle subscription updated (plan change, status change, etc.)
    async fn on_subscription_updated(&self, event: &SubscriptionEvent) -> anyhow::Result<()>;

    /// Handle subscription deleted/canceled
    async fn on_subscription_deleted(&self, event: &SubscriptionEvent) -> anyhow::Result<()>;

    /// Handle invoice payment succeeded
    async fn on_payment_succeeded(&self, event: &InvoiceEvent) -> anyhow::Result<()>;

    /// Handle invoice payment failed
    async fn on_payment_failed(&self, event: &InvoiceEvent) -> anyhow::Result<()>;

    /// Handle new customer created
    async fn on_customer_created(&self, event: &CustomerEvent) -> anyhow::Result<()>;
}

/// Event processor that handles webhook events asynchronously
pub struct EventProcessor<H: SubscriptionHandler, S: IdempotencyStore> {
    handler: Arc<H>,
    idempotency_store: Arc<S>,
    config: StripeWebhookConfig,
    /// Channel for background task processing
    task_sender: mpsc::Sender<ProcessingTask>,
}

/// A processing task sent to background workers
struct ProcessingTask {
    event: StripeEvent,
    attempt: u32,
}

impl<H: SubscriptionHandler, S: IdempotencyStore> EventProcessor<H, S> {
    /// Create a new event processor
    pub fn new(
        handler: Arc<H>,
        idempotency_store: Arc<S>,
        config: StripeWebhookConfig,
    ) -> (Self, ProcessorHandle<H, S>) {
        let (tx, rx) = mpsc::channel(1000);

        let processor = Self {
            handler: handler.clone(),
            idempotency_store: idempotency_store.clone(),
            config: config.clone(),
            task_sender: tx,
        };

        let handle = ProcessorHandle {
            handler,
            idempotency_store,
            config,
            task_receiver: rx,
        };

        (processor, handle)
    }

    /// Queue an event for async processing
    ///
    /// This returns immediately after queuing. The actual processing
    /// happens in a background task.
    pub async fn queue_event(&self, event: StripeEvent) -> StripeWebhookResult<()> {
        let task = ProcessingTask { event, attempt: 0 };

        self.task_sender.send(task).await.map_err(|e| {
            crate::stripe::error::StripeWebhookError::InternalError(format!(
                "Failed to queue event: {}",
                e
            ))
        })?;

        Ok(())
    }

    /// Process an event synchronously (for testing or immediate processing)
    pub async fn process_event_sync(&self, event: &StripeEvent) -> StripeWebhookResult<()> {
        process_single_event(&self.handler, &self.idempotency_store, event, &self.config).await
    }
}

/// Handle for running the background processor
pub struct ProcessorHandle<H: SubscriptionHandler, S: IdempotencyStore> {
    handler: Arc<H>,
    idempotency_store: Arc<S>,
    config: StripeWebhookConfig,
    task_receiver: mpsc::Receiver<ProcessingTask>,
}

impl<H: SubscriptionHandler, S: IdempotencyStore> ProcessorHandle<H, S> {
    /// Run the background processor
    ///
    /// This should be spawned as a tokio task:
    ///
    /// ```rust,ignore
    /// tokio::spawn(async move {
    ///     handle.run().await;
    /// });
    /// ```
    pub async fn run(mut self) {
        tracing::info!("Starting Stripe webhook event processor");

        while let Some(task) = self.task_receiver.recv().await {
            let handler = self.handler.clone();
            let store = self.idempotency_store.clone();
            let config = self.config.clone();

            // Spawn each event processing in its own task
            tokio::spawn(async move {
                process_with_retry(handler, store, task.event, &config).await;
            });
        }

        tracing::info!("Stripe webhook event processor shutting down");
    }
}

/// Process a single event with retry logic
async fn process_with_retry<H: SubscriptionHandler, S: IdempotencyStore>(
    handler: Arc<H>,
    store: Arc<S>,
    event: StripeEvent,
    config: &StripeWebhookConfig,
) {
    let event_id = event.id.clone();
    let event_type = event.event_type.clone();

    for attempt in 0..=config.max_retries {
        if attempt > 0 {
            let delay = config.retry_delay(attempt - 1);
            tracing::info!(
                event_id = %event_id,
                event_type = %event_type,
                attempt,
                delay_ms = delay.as_millis(),
                "Retrying event processing"
            );
            tokio::time::sleep(delay).await;
        }

        match process_single_event(&handler, &store, &event, config).await {
            Ok(()) => {
                tracing::info!(
                    event_id = %event_id,
                    event_type = %event_type,
                    attempts = attempt + 1,
                    "Event processed successfully"
                );
                return;
            }
            Err(e) => {
                tracing::warn!(
                    event_id = %event_id,
                    event_type = %event_type,
                    attempt = attempt + 1,
                    max_retries = config.max_retries,
                    error = %e,
                    "Event processing failed"
                );

                if attempt == config.max_retries {
                    // Final failure - mark as failed in idempotency store
                    if let Err(mark_err) = store.mark_failed(&event_id, &e.to_string()).await {
                        tracing::error!(
                            event_id = %event_id,
                            error = %mark_err,
                            "Failed to mark event as failed in idempotency store"
                        );
                    }
                }
            }
        }
    }
}

/// Process a single event
async fn process_single_event<H: SubscriptionHandler, S: IdempotencyStore>(
    handler: &Arc<H>,
    store: &Arc<S>,
    event: &StripeEvent,
    config: &StripeWebhookConfig,
) -> StripeWebhookResult<()> {
    let event_type = event.typed_event_type();

    // Apply timeout to prevent hanging
    let result = timeout(config.processing_timeout, async {
        match event_type {
            StripeEventType::SubscriptionCreated => {
                let sub_event = event.as_subscription()?;
                handler
                    .on_subscription_created(&sub_event)
                    .await
                    .map_err(|e| {
                        crate::stripe::error::StripeWebhookError::ProcessingFailed(e.to_string())
                    })
            }
            StripeEventType::SubscriptionUpdated => {
                let sub_event = event.as_subscription()?;
                handler
                    .on_subscription_updated(&sub_event)
                    .await
                    .map_err(|e| {
                        crate::stripe::error::StripeWebhookError::ProcessingFailed(e.to_string())
                    })
            }
            StripeEventType::SubscriptionDeleted => {
                let sub_event = event.as_subscription()?;
                handler
                    .on_subscription_deleted(&sub_event)
                    .await
                    .map_err(|e| {
                        crate::stripe::error::StripeWebhookError::ProcessingFailed(e.to_string())
                    })
            }
            StripeEventType::InvoicePaymentSucceeded => {
                let invoice_event = event.as_invoice()?;
                handler
                    .on_payment_succeeded(&invoice_event)
                    .await
                    .map_err(|e| {
                        crate::stripe::error::StripeWebhookError::ProcessingFailed(e.to_string())
                    })
            }
            StripeEventType::InvoicePaymentFailed => {
                let invoice_event = event.as_invoice()?;
                handler
                    .on_payment_failed(&invoice_event)
                    .await
                    .map_err(|e| {
                        crate::stripe::error::StripeWebhookError::ProcessingFailed(e.to_string())
                    })
            }
            StripeEventType::CustomerCreated => {
                let customer_event = event.as_customer()?;
                handler
                    .on_customer_created(&customer_event)
                    .await
                    .map_err(|e| {
                        crate::stripe::error::StripeWebhookError::ProcessingFailed(e.to_string())
                    })
            }
            StripeEventType::Unknown => {
                tracing::debug!(
                    event_id = %event.id,
                    event_type = %event.event_type,
                    "Ignoring unknown event type"
                );
                Ok(())
            }
        }
    })
    .await;

    match result {
        Ok(inner_result) => {
            if inner_result.is_ok() {
                // Mark as completed in idempotency store
                store.mark_completed(&event.id).await?;
            }
            inner_result
        }
        Err(_) => Err(crate::stripe::error::StripeWebhookError::ProcessingFailed(
            format!("Processing timed out after {:?}", config.processing_timeout),
        )),
    }
}

/// No-op handler for testing
#[derive(Clone)]
pub struct NoOpHandler;

#[async_trait::async_trait]
impl SubscriptionHandler for NoOpHandler {
    async fn on_subscription_created(&self, _event: &SubscriptionEvent) -> anyhow::Result<()> {
        Ok(())
    }
    async fn on_subscription_updated(&self, _event: &SubscriptionEvent) -> anyhow::Result<()> {
        Ok(())
    }
    async fn on_subscription_deleted(&self, _event: &SubscriptionEvent) -> anyhow::Result<()> {
        Ok(())
    }
    async fn on_payment_succeeded(&self, _event: &InvoiceEvent) -> anyhow::Result<()> {
        Ok(())
    }
    async fn on_payment_failed(&self, _event: &InvoiceEvent) -> anyhow::Result<()> {
        Ok(())
    }
    async fn on_customer_created(&self, _event: &CustomerEvent) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Logging handler that logs all events
#[derive(Clone)]
pub struct LoggingHandler;

#[async_trait::async_trait]
impl SubscriptionHandler for LoggingHandler {
    async fn on_subscription_created(&self, event: &SubscriptionEvent) -> anyhow::Result<()> {
        tracing::info!(
            subscription_id = %event.subscription.id,
            customer_id = %event.subscription.customer,
            status = ?event.subscription.status,
            "Subscription created"
        );
        Ok(())
    }

    async fn on_subscription_updated(&self, event: &SubscriptionEvent) -> anyhow::Result<()> {
        tracing::info!(
            subscription_id = %event.subscription.id,
            customer_id = %event.subscription.customer,
            status = ?event.subscription.status,
            cancel_at_period_end = event.subscription.cancel_at_period_end,
            "Subscription updated"
        );
        Ok(())
    }

    async fn on_subscription_deleted(&self, event: &SubscriptionEvent) -> anyhow::Result<()> {
        tracing::info!(
            subscription_id = %event.subscription.id,
            customer_id = %event.subscription.customer,
            "Subscription deleted"
        );
        Ok(())
    }

    async fn on_payment_succeeded(&self, event: &InvoiceEvent) -> anyhow::Result<()> {
        tracing::info!(
            invoice_id = %event.invoice.id,
            customer_id = %event.invoice.customer,
            amount_paid = event.invoice.amount_paid,
            currency = %event.invoice.currency,
            "Payment succeeded"
        );
        Ok(())
    }

    async fn on_payment_failed(&self, event: &InvoiceEvent) -> anyhow::Result<()> {
        tracing::warn!(
            invoice_id = %event.invoice.id,
            customer_id = %event.invoice.customer,
            amount_due = event.invoice.amount_due,
            currency = %event.invoice.currency,
            "Payment failed"
        );
        Ok(())
    }

    async fn on_customer_created(&self, event: &CustomerEvent) -> anyhow::Result<()> {
        tracing::info!(
            customer_id = %event.customer.id,
            email = ?event.customer.email,
            "Customer created"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stripe::idempotency::InMemoryIdempotencyStore;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    /// Test handler that tracks calls
    struct TestHandler {
        subscription_created_calls: AtomicU32,
        subscription_updated_calls: AtomicU32,
        subscription_deleted_calls: AtomicU32,
        payment_succeeded_calls: AtomicU32,
        payment_failed_calls: AtomicU32,
        customer_created_calls: AtomicU32,
        should_fail: std::sync::atomic::AtomicBool,
    }

    impl TestHandler {
        fn new() -> Self {
            Self {
                subscription_created_calls: AtomicU32::new(0),
                subscription_updated_calls: AtomicU32::new(0),
                subscription_deleted_calls: AtomicU32::new(0),
                payment_succeeded_calls: AtomicU32::new(0),
                payment_failed_calls: AtomicU32::new(0),
                customer_created_calls: AtomicU32::new(0),
                should_fail: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    #[async_trait::async_trait]
    impl SubscriptionHandler for TestHandler {
        async fn on_subscription_created(&self, _event: &SubscriptionEvent) -> anyhow::Result<()> {
            self.subscription_created_calls
                .fetch_add(1, Ordering::SeqCst);
            if self.should_fail.load(Ordering::SeqCst) {
                anyhow::bail!("Simulated failure");
            }
            Ok(())
        }
        async fn on_subscription_updated(&self, _event: &SubscriptionEvent) -> anyhow::Result<()> {
            self.subscription_updated_calls
                .fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_subscription_deleted(&self, _event: &SubscriptionEvent) -> anyhow::Result<()> {
            self.subscription_deleted_calls
                .fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_payment_succeeded(&self, _event: &InvoiceEvent) -> anyhow::Result<()> {
            self.payment_succeeded_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_payment_failed(&self, _event: &InvoiceEvent) -> anyhow::Result<()> {
            self.payment_failed_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_customer_created(&self, _event: &CustomerEvent) -> anyhow::Result<()> {
            self.customer_created_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn create_test_subscription_event() -> StripeEvent {
        let json = r#"{
            "id": "evt_test_123",
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

        StripeEvent::from_bytes(json.as_bytes()).unwrap()
    }

    #[tokio::test]
    async fn test_process_subscription_created() {
        let handler = Arc::new(TestHandler::new());
        let store = Arc::new(InMemoryIdempotencyStore::new(
            Duration::from_secs(3600),
            1000,
        ));
        let config = StripeWebhookConfig::test_config();

        let event = create_test_subscription_event();

        // Record in idempotency store first
        store.check_and_record(&event.id).await.unwrap();

        // Process the event
        process_single_event(&handler, &store, &event, &config)
            .await
            .unwrap();

        assert_eq!(handler.subscription_created_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_processor_queue_and_run() {
        let handler = Arc::new(TestHandler::new());
        let store = Arc::new(InMemoryIdempotencyStore::new(
            Duration::from_secs(3600),
            1000,
        ));
        let config = StripeWebhookConfig::test_config();

        let (processor, handle) = EventProcessor::new(handler.clone(), store.clone(), config);

        // Start the background processor
        let processor_task = tokio::spawn(async move {
            handle.run().await;
        });

        // Queue an event
        let event = create_test_subscription_event();
        store.check_and_record(&event.id).await.unwrap();
        processor.queue_event(event).await.unwrap();

        // Give it time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify handler was called
        assert_eq!(handler.subscription_created_calls.load(Ordering::SeqCst), 1);

        // Cleanup
        processor_task.abort();
    }

    #[tokio::test]
    async fn test_unknown_event_type_ignored() {
        let handler = Arc::new(TestHandler::new());
        let store = Arc::new(InMemoryIdempotencyStore::new(
            Duration::from_secs(3600),
            1000,
        ));
        let config = StripeWebhookConfig::test_config();

        let json = r#"{
            "id": "evt_unknown_123",
            "type": "some.unknown.event",
            "created": 1614556800,
            "livemode": false,
            "pending_webhooks": 1,
            "data": {
                "object": {}
            }
        }"#;

        let event = StripeEvent::from_bytes(json.as_bytes()).unwrap();
        store.check_and_record(&event.id).await.unwrap();

        // Should succeed (unknown events are ignored)
        process_single_event(&handler, &store, &event, &config)
            .await
            .unwrap();

        // No handlers should have been called
        assert_eq!(handler.subscription_created_calls.load(Ordering::SeqCst), 0);
    }
}

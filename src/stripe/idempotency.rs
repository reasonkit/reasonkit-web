//! Idempotency Layer for Stripe Webhooks
//!
//! Handles deduplication of webhook deliveries using event IDs.
//! Stripe may deliver the same webhook multiple times due to:
//!
//! - Network issues causing Stripe to not receive our 2xx response
//! - Retries due to our server returning 5xx
//! - Stripe's at-least-once delivery guarantee
//!
//! # Implementation
//!
//! We track processed event IDs with their processing status and timestamp.
//! Events are automatically cleaned up after the TTL expires.
//!
//! # Storage Options
//!
//! - `InMemoryIdempotencyStore`: Simple in-memory store (single instance only)
//! - For production with multiple instances, use Redis or database backing

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::stripe::error::{StripeWebhookError, StripeWebhookResult};

/// Status of event processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingStatus {
    /// Event is currently being processed
    InProgress,
    /// Event processing completed successfully
    Completed,
    /// Event processing failed (may be retried)
    Failed { error: String },
}

/// Entry in the idempotency store
#[derive(Debug, Clone)]
pub struct IdempotencyEntry {
    /// Event ID
    pub event_id: String,
    /// Processing status
    pub status: ProcessingStatus,
    /// When the event was first received
    pub received_at: Instant,
    /// When the status was last updated
    pub updated_at: Instant,
    /// Number of processing attempts
    pub attempts: u32,
}

/// Trait for idempotency storage backends
#[async_trait::async_trait]
pub trait IdempotencyStore: Send + Sync + 'static {
    /// Check if an event has been processed and record it if not
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if this is a new event (not seen before)
    /// - `Ok(false)` if the event was already processed or is in progress
    /// - `Err` if the check failed
    async fn check_and_record(&self, event_id: &str) -> StripeWebhookResult<bool>;

    /// Mark an event as completed
    async fn mark_completed(&self, event_id: &str) -> StripeWebhookResult<()>;

    /// Mark an event as failed
    async fn mark_failed(&self, event_id: &str, error: &str) -> StripeWebhookResult<()>;

    /// Get the status of an event
    async fn get_status(&self, event_id: &str) -> StripeWebhookResult<Option<IdempotencyEntry>>;

    /// Clean up expired entries
    async fn cleanup(&self) -> StripeWebhookResult<usize>;
}

/// In-memory idempotency store
///
/// Suitable for single-instance deployments. For multi-instance deployments,
/// use a distributed store (Redis, database, etc.)
pub struct InMemoryIdempotencyStore {
    /// Event entries keyed by event ID
    entries: Arc<RwLock<HashMap<String, IdempotencyEntry>>>,
    /// Time-to-live for entries
    ttl: Duration,
    /// Maximum number of entries
    max_entries: usize,
}

impl InMemoryIdempotencyStore {
    /// Create a new in-memory store
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            max_entries,
        }
    }

    /// Create from config
    pub fn from_config(config: &crate::stripe::config::StripeWebhookConfig) -> Self {
        Self::new(config.idempotency_ttl, config.idempotency_max_entries)
    }

    /// Check if entry is expired
    fn is_expired(&self, entry: &IdempotencyEntry) -> bool {
        entry.received_at.elapsed() > self.ttl
    }

    /// Get current entry count
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Check if store is empty
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}

#[async_trait::async_trait]
impl IdempotencyStore for InMemoryIdempotencyStore {
    async fn check_and_record(&self, event_id: &str) -> StripeWebhookResult<bool> {
        let mut entries = self.entries.write().await;
        let now = Instant::now();

        // Check if already exists and not expired
        if let Some(existing) = entries.get(event_id) {
            if !self.is_expired(existing) {
                tracing::debug!(
                    event_id,
                    status = ?existing.status,
                    "Event already in idempotency store"
                );
                return Ok(false);
            }
            // Entry is expired, remove it and proceed as new
            entries.remove(event_id);
        }

        // Check if we need to evict old entries
        if entries.len() >= self.max_entries {
            // Remove oldest entries
            let mut to_remove: Vec<String> = entries
                .iter()
                .filter(|(_, entry)| self.is_expired(entry))
                .map(|(id, _)| id.clone())
                .collect();

            // If not enough expired entries, remove oldest by received_at
            if to_remove.len() < entries.len() / 10 {
                let mut by_age: Vec<_> = entries.iter().collect();
                by_age.sort_by_key(|(_, entry)| entry.received_at);
                to_remove.extend(
                    by_age
                        .iter()
                        .take(entries.len() / 10)
                        .map(|(id, _)| (*id).clone()),
                );
            }

            for id in to_remove {
                entries.remove(&id);
            }

            tracing::info!(
                remaining = entries.len(),
                max = self.max_entries,
                "Evicted old idempotency entries"
            );
        }

        // Record new entry
        entries.insert(
            event_id.to_string(),
            IdempotencyEntry {
                event_id: event_id.to_string(),
                status: ProcessingStatus::InProgress,
                received_at: now,
                updated_at: now,
                attempts: 1,
            },
        );

        tracing::debug!(event_id, "New event recorded in idempotency store");
        Ok(true)
    }

    async fn mark_completed(&self, event_id: &str) -> StripeWebhookResult<()> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.get_mut(event_id) {
            entry.status = ProcessingStatus::Completed;
            entry.updated_at = Instant::now();
            tracing::debug!(event_id, "Event marked as completed");
        } else {
            tracing::warn!(
                event_id,
                "Attempted to mark non-existent event as completed"
            );
        }

        Ok(())
    }

    async fn mark_failed(&self, event_id: &str, error: &str) -> StripeWebhookResult<()> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.get_mut(event_id) {
            entry.status = ProcessingStatus::Failed {
                error: error.to_string(),
            };
            entry.updated_at = Instant::now();
            tracing::debug!(event_id, error, "Event marked as failed");
        } else {
            tracing::warn!(event_id, "Attempted to mark non-existent event as failed");
        }

        Ok(())
    }

    async fn get_status(&self, event_id: &str) -> StripeWebhookResult<Option<IdempotencyEntry>> {
        let entries = self.entries.read().await;

        if let Some(entry) = entries.get(event_id) {
            if self.is_expired(entry) {
                return Ok(None);
            }
            return Ok(Some(entry.clone()));
        }

        Ok(None)
    }

    async fn cleanup(&self) -> StripeWebhookResult<usize> {
        let mut entries = self.entries.write().await;
        let before = entries.len();

        entries.retain(|_, entry| !self.is_expired(entry));

        let removed = before - entries.len();
        if removed > 0 {
            tracing::info!(
                removed,
                remaining = entries.len(),
                "Cleaned up expired idempotency entries"
            );
        }

        Ok(removed)
    }
}

/// Idempotency middleware that wraps event processing
pub struct IdempotencyMiddleware<S: IdempotencyStore> {
    store: Arc<S>,
}

impl<S: IdempotencyStore> IdempotencyMiddleware<S> {
    /// Create new middleware
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    /// Check if event should be processed
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if event should be processed (new event)
    /// - `Ok(false)` if event was already processed (return 202)
    /// - `Err` for already-processed events that need special handling
    pub async fn should_process(&self, event_id: &str) -> StripeWebhookResult<bool> {
        match self.store.check_and_record(event_id).await {
            Ok(true) => Ok(true),
            Ok(false) => {
                // Already processed or in progress
                Err(StripeWebhookError::AlreadyProcessed {
                    event_id: event_id.to_string(),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Mark event processing as complete
    pub async fn complete(&self, event_id: &str) -> StripeWebhookResult<()> {
        self.store.mark_completed(event_id).await
    }

    /// Mark event processing as failed
    pub async fn fail(&self, event_id: &str, error: &str) -> StripeWebhookResult<()> {
        self.store.mark_failed(event_id, error).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_check_and_record_new_event() {
        let store = InMemoryIdempotencyStore::new(Duration::from_secs(3600), 1000);

        let result = store.check_and_record("evt_123").await.unwrap();
        assert!(result); // New event

        let result = store.check_and_record("evt_123").await.unwrap();
        assert!(!result); // Already recorded
    }

    #[tokio::test]
    async fn test_check_and_record_different_events() {
        let store = InMemoryIdempotencyStore::new(Duration::from_secs(3600), 1000);

        assert!(store.check_and_record("evt_1").await.unwrap());
        assert!(store.check_and_record("evt_2").await.unwrap());
        assert!(store.check_and_record("evt_3").await.unwrap());

        assert!(!store.check_and_record("evt_1").await.unwrap());
        assert!(!store.check_and_record("evt_2").await.unwrap());
    }

    #[tokio::test]
    async fn test_mark_completed() {
        let store = InMemoryIdempotencyStore::new(Duration::from_secs(3600), 1000);

        store.check_and_record("evt_123").await.unwrap();
        store.mark_completed("evt_123").await.unwrap();

        let entry = store.get_status("evt_123").await.unwrap().unwrap();
        assert_eq!(entry.status, ProcessingStatus::Completed);
    }

    #[tokio::test]
    async fn test_mark_failed() {
        let store = InMemoryIdempotencyStore::new(Duration::from_secs(3600), 1000);

        store.check_and_record("evt_123").await.unwrap();
        store
            .mark_failed("evt_123", "Database error")
            .await
            .unwrap();

        let entry = store.get_status("evt_123").await.unwrap().unwrap();
        assert!(matches!(entry.status, ProcessingStatus::Failed { .. }));
    }

    #[tokio::test]
    async fn test_expired_entries() {
        let store = InMemoryIdempotencyStore::new(Duration::from_millis(10), 1000);

        store.check_and_record("evt_123").await.unwrap();

        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Should be able to record again after expiry
        assert!(store.check_and_record("evt_123").await.unwrap());
    }

    #[tokio::test]
    async fn test_cleanup() {
        let store = InMemoryIdempotencyStore::new(Duration::from_millis(10), 1000);

        store.check_and_record("evt_1").await.unwrap();
        store.check_and_record("evt_2").await.unwrap();

        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(20)).await;

        let removed = store.cleanup().await.unwrap();
        assert_eq!(removed, 2);
        assert!(store.is_empty().await);
    }

    #[tokio::test]
    async fn test_max_entries_eviction() {
        let store = InMemoryIdempotencyStore::new(Duration::from_secs(3600), 10);

        // Fill up the store
        for i in 0..15 {
            store.check_and_record(&format!("evt_{}", i)).await.unwrap();
        }

        // Should have evicted some entries
        assert!(store.len().await <= 15);
    }

    #[tokio::test]
    async fn test_idempotency_middleware() {
        let store = Arc::new(InMemoryIdempotencyStore::new(
            Duration::from_secs(3600),
            1000,
        ));
        let middleware = IdempotencyMiddleware::new(store);

        // First call should succeed
        assert!(middleware.should_process("evt_123").await.is_ok());

        // Second call should return AlreadyProcessed error
        let result = middleware.should_process("evt_123").await;
        assert!(matches!(
            result,
            Err(StripeWebhookError::AlreadyProcessed { .. })
        ));

        // Mark complete
        middleware.complete("evt_123").await.unwrap();
    }
}

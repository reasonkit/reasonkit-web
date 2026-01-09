//! Stripe Event Types
//!
//! Strongly-typed representations of Stripe webhook events for SaaS subscriptions.

use serde::{Deserialize, Serialize};

use crate::stripe::error::{StripeWebhookError, StripeWebhookResult};

/// Stripe event types we handle
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StripeEventType {
    // Customer events
    #[serde(rename = "customer.created")]
    CustomerCreated,

    // Subscription events
    #[serde(rename = "customer.subscription.created")]
    SubscriptionCreated,
    #[serde(rename = "customer.subscription.updated")]
    SubscriptionUpdated,
    #[serde(rename = "customer.subscription.deleted")]
    SubscriptionDeleted,

    // Invoice events
    #[serde(rename = "invoice.payment_succeeded")]
    InvoicePaymentSucceeded,
    #[serde(rename = "invoice.payment_failed")]
    InvoicePaymentFailed,

    // Catch-all for events we don't explicitly handle
    #[serde(other)]
    Unknown,
}

impl StripeEventType {
    /// Parse event type from string
    pub fn from_str(s: &str) -> Self {
        match s {
            "customer.created" => Self::CustomerCreated,
            "customer.subscription.created" => Self::SubscriptionCreated,
            "customer.subscription.updated" => Self::SubscriptionUpdated,
            "customer.subscription.deleted" => Self::SubscriptionDeleted,
            "invoice.payment_succeeded" => Self::InvoicePaymentSucceeded,
            "invoice.payment_failed" => Self::InvoicePaymentFailed,
            _ => Self::Unknown,
        }
    }

    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CustomerCreated => "customer.created",
            Self::SubscriptionCreated => "customer.subscription.created",
            Self::SubscriptionUpdated => "customer.subscription.updated",
            Self::SubscriptionDeleted => "customer.subscription.deleted",
            Self::InvoicePaymentSucceeded => "invoice.payment_succeeded",
            Self::InvoicePaymentFailed => "invoice.payment_failed",
            Self::Unknown => "unknown",
        }
    }

    /// Check if this is a known event type
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Generic Stripe event envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StripeEvent {
    /// Unique identifier for the event
    pub id: String,

    /// Type of event
    #[serde(rename = "type")]
    pub event_type: String,

    /// Time of event creation (Unix timestamp)
    pub created: i64,

    /// API version used to render data
    #[serde(default)]
    pub api_version: Option<String>,

    /// Whether this is a live mode event
    pub livemode: bool,

    /// Number of times Stripe has attempted to deliver
    #[serde(default)]
    pub pending_webhooks: u32,

    /// Object containing event data
    pub data: EventData,

    /// Request that caused the event (if applicable)
    #[serde(default)]
    pub request: Option<EventRequest>,
}

impl StripeEvent {
    /// Parse from raw JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> StripeWebhookResult<Self> {
        serde_json::from_slice(bytes).map_err(|e| StripeWebhookError::InvalidPayload(e.to_string()))
    }

    /// Get the typed event type
    pub fn typed_event_type(&self) -> StripeEventType {
        StripeEventType::from_str(&self.event_type)
    }

    /// Extract subscription from event data
    pub fn as_subscription(&self) -> StripeWebhookResult<SubscriptionEvent> {
        match self.typed_event_type() {
            StripeEventType::SubscriptionCreated
            | StripeEventType::SubscriptionUpdated
            | StripeEventType::SubscriptionDeleted => {
                let subscription: Subscription =
                    serde_json::from_value(self.data.object.clone())
                        .map_err(|e| StripeWebhookError::InvalidPayload(e.to_string()))?;

                Ok(SubscriptionEvent {
                    event_id: self.id.clone(),
                    event_type: self.typed_event_type(),
                    subscription,
                    previous_attributes: self.data.previous_attributes.clone(),
                })
            }
            _ => Err(StripeWebhookError::InvalidPayload(format!(
                "Event {} is not a subscription event",
                self.event_type
            ))),
        }
    }

    /// Extract invoice from event data
    pub fn as_invoice(&self) -> StripeWebhookResult<InvoiceEvent> {
        match self.typed_event_type() {
            StripeEventType::InvoicePaymentSucceeded | StripeEventType::InvoicePaymentFailed => {
                let invoice: Invoice = serde_json::from_value(self.data.object.clone())
                    .map_err(|e| StripeWebhookError::InvalidPayload(e.to_string()))?;

                Ok(InvoiceEvent {
                    event_id: self.id.clone(),
                    event_type: self.typed_event_type(),
                    invoice,
                })
            }
            _ => Err(StripeWebhookError::InvalidPayload(format!(
                "Event {} is not an invoice event",
                self.event_type
            ))),
        }
    }

    /// Extract customer from event data
    pub fn as_customer(&self) -> StripeWebhookResult<CustomerEvent> {
        match self.typed_event_type() {
            StripeEventType::CustomerCreated => {
                let customer: Customer = serde_json::from_value(self.data.object.clone())
                    .map_err(|e| StripeWebhookError::InvalidPayload(e.to_string()))?;

                Ok(CustomerEvent {
                    event_id: self.id.clone(),
                    event_type: self.typed_event_type(),
                    customer,
                })
            }
            _ => Err(StripeWebhookError::InvalidPayload(format!(
                "Event {} is not a customer event",
                self.event_type
            ))),
        }
    }
}

/// Event data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventData {
    /// The actual event object (subscription, invoice, customer, etc.)
    pub object: serde_json::Value,

    /// Previous values for updated fields (only in *.updated events)
    #[serde(default)]
    pub previous_attributes: Option<serde_json::Value>,
}

/// Request that triggered the event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRequest {
    /// Request ID
    pub id: Option<String>,
    /// Idempotency key used in the request
    pub idempotency_key: Option<String>,
}

// =============================================================================
// Subscription Types
// =============================================================================

/// Subscription event with typed data
#[derive(Debug, Clone)]
pub struct SubscriptionEvent {
    /// The event ID
    pub event_id: String,
    /// Type of subscription event
    pub event_type: StripeEventType,
    /// The subscription object
    pub subscription: Subscription,
    /// Previous values (for updates)
    pub previous_attributes: Option<serde_json::Value>,
}

/// Stripe subscription object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    /// Subscription ID (sub_...)
    pub id: String,
    /// Customer ID (cus_...)
    pub customer: String,
    /// Subscription status
    pub status: SubscriptionStatus,
    /// Current billing period start (Unix timestamp)
    pub current_period_start: i64,
    /// Current billing period end (Unix timestamp)
    pub current_period_end: i64,
    /// Whether subscription will cancel at period end
    #[serde(default)]
    pub cancel_at_period_end: bool,
    /// When the subscription was canceled (if applicable)
    pub canceled_at: Option<i64>,
    /// When the subscription ended (if applicable)
    pub ended_at: Option<i64>,
    /// Trial end date (if applicable)
    pub trial_end: Option<i64>,
    /// Subscription items (plans/prices)
    pub items: SubscriptionItems,
    /// Metadata attached to the subscription
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Whether this is a live mode subscription
    pub livemode: bool,
}

/// Subscription status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionStatus {
    Active,
    PastDue,
    Unpaid,
    Canceled,
    Incomplete,
    IncompleteExpired,
    Trialing,
    Paused,
    #[serde(other)]
    Unknown,
}

impl SubscriptionStatus {
    /// Check if subscription is in a "good" state
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active | Self::Trialing)
    }

    /// Check if subscription requires payment attention
    pub fn requires_payment_action(&self) -> bool {
        matches!(self, Self::PastDue | Self::Unpaid | Self::Incomplete)
    }
}

/// Subscription items container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionItems {
    /// List of subscription items
    pub data: Vec<SubscriptionItem>,
}

/// Individual subscription item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionItem {
    /// Item ID
    pub id: String,
    /// Price object
    pub price: Price,
    /// Quantity
    #[serde(default = "default_quantity")]
    pub quantity: u32,
}

fn default_quantity() -> u32 {
    1
}

/// Price object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Price {
    /// Price ID
    pub id: String,
    /// Product ID
    pub product: String,
    /// Unit amount in cents
    pub unit_amount: Option<i64>,
    /// Currency
    pub currency: String,
    /// Recurring information
    pub recurring: Option<Recurring>,
}

/// Recurring price details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recurring {
    /// Billing interval (day, week, month, year)
    pub interval: String,
    /// Number of intervals
    pub interval_count: u32,
}

// =============================================================================
// Invoice Types
// =============================================================================

/// Invoice event with typed data
#[derive(Debug, Clone)]
pub struct InvoiceEvent {
    /// The event ID
    pub event_id: String,
    /// Type of invoice event
    pub event_type: StripeEventType,
    /// The invoice object
    pub invoice: Invoice,
}

/// Stripe invoice object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    /// Invoice ID (in_...)
    pub id: String,
    /// Customer ID
    pub customer: String,
    /// Associated subscription ID (if any)
    pub subscription: Option<String>,
    /// Invoice status
    pub status: InvoiceStatus,
    /// Total amount in cents
    pub amount_due: i64,
    /// Amount paid in cents
    pub amount_paid: i64,
    /// Amount remaining in cents
    pub amount_remaining: i64,
    /// Currency
    pub currency: String,
    /// Billing reason
    pub billing_reason: Option<String>,
    /// Customer email at time of invoice
    pub customer_email: Option<String>,
    /// Hosted invoice URL
    pub hosted_invoice_url: Option<String>,
    /// Invoice PDF URL
    pub invoice_pdf: Option<String>,
    /// Payment intent ID (if payment attempted)
    pub payment_intent: Option<String>,
    /// When created (Unix timestamp)
    pub created: i64,
    /// Period start
    pub period_start: i64,
    /// Period end
    pub period_end: i64,
    /// Whether this is live mode
    pub livemode: bool,
}

/// Invoice status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvoiceStatus {
    Draft,
    Open,
    Paid,
    Uncollectible,
    Void,
    #[serde(other)]
    Unknown,
}

// =============================================================================
// Customer Types
// =============================================================================

/// Customer event with typed data
#[derive(Debug, Clone)]
pub struct CustomerEvent {
    /// The event ID
    pub event_id: String,
    /// Type of customer event
    pub event_type: StripeEventType,
    /// The customer object
    pub customer: Customer,
}

/// Stripe customer object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Customer {
    /// Customer ID (cus_...)
    pub id: String,
    /// Customer email
    pub email: Option<String>,
    /// Customer name
    pub name: Option<String>,
    /// Customer description
    pub description: Option<String>,
    /// When created (Unix timestamp)
    pub created: i64,
    /// Metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Whether this is live mode
    pub livemode: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_parsing() {
        assert_eq!(
            StripeEventType::from_str("customer.subscription.created"),
            StripeEventType::SubscriptionCreated
        );
        assert_eq!(
            StripeEventType::from_str("invoice.payment_succeeded"),
            StripeEventType::InvoicePaymentSucceeded
        );
        assert_eq!(
            StripeEventType::from_str("unknown.event"),
            StripeEventType::Unknown
        );
    }

    #[test]
    fn test_subscription_status() {
        assert!(SubscriptionStatus::Active.is_active());
        assert!(SubscriptionStatus::Trialing.is_active());
        assert!(!SubscriptionStatus::Canceled.is_active());

        assert!(SubscriptionStatus::PastDue.requires_payment_action());
        assert!(!SubscriptionStatus::Active.requires_payment_action());
    }

    #[test]
    fn test_parse_subscription_event() {
        let json = r#"{
            "id": "evt_1234567890",
            "type": "customer.subscription.created",
            "created": 1614556800,
            "livemode": false,
            "pending_webhooks": 1,
            "data": {
                "object": {
                    "id": "sub_1234567890",
                    "customer": "cus_1234567890",
                    "status": "active",
                    "current_period_start": 1614556800,
                    "current_period_end": 1617235200,
                    "cancel_at_period_end": false,
                    "items": {
                        "data": [{
                            "id": "si_1234567890",
                            "price": {
                                "id": "price_1234567890",
                                "product": "prod_1234567890",
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

        let event = StripeEvent::from_bytes(json.as_bytes()).unwrap();
        assert_eq!(event.id, "evt_1234567890");
        assert_eq!(
            event.typed_event_type(),
            StripeEventType::SubscriptionCreated
        );

        let sub_event = event.as_subscription().unwrap();
        assert_eq!(sub_event.subscription.id, "sub_1234567890");
        assert_eq!(sub_event.subscription.status, SubscriptionStatus::Active);
    }

    #[test]
    fn test_parse_invoice_event() {
        let json = r#"{
            "id": "evt_invoice_1234",
            "type": "invoice.payment_succeeded",
            "created": 1614556800,
            "livemode": false,
            "pending_webhooks": 1,
            "data": {
                "object": {
                    "id": "in_1234567890",
                    "customer": "cus_1234567890",
                    "subscription": "sub_1234567890",
                    "status": "paid",
                    "amount_due": 2000,
                    "amount_paid": 2000,
                    "amount_remaining": 0,
                    "currency": "usd",
                    "created": 1614556800,
                    "period_start": 1614556800,
                    "period_end": 1617235200,
                    "livemode": false
                }
            }
        }"#;

        let event = StripeEvent::from_bytes(json.as_bytes()).unwrap();
        let invoice_event = event.as_invoice().unwrap();

        assert_eq!(invoice_event.invoice.id, "in_1234567890");
        assert_eq!(invoice_event.invoice.status, InvoiceStatus::Paid);
        assert_eq!(invoice_event.invoice.amount_paid, 2000);
    }
}

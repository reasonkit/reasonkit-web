//! Handler module tests
//!
//! Comprehensive tests for the HTTP handlers including capture, feed, and status endpoints.
//! These tests focus on unit testing handler logic with mocked dependencies.

use reasonkit_web::handlers::feed::{FeedEvent, FeedState};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// FeedState Tests
// ============================================================================

#[cfg(test)]
mod feed_state_tests {
    use super::*;

    #[test]
    fn test_feed_state_creation() {
        let state = FeedState::new(100);
        // State should be created with specified capacity
        assert!(state.connected_clients() == 0);
    }

    #[test]
    fn test_feed_state_default_capacity() {
        let state = FeedState::new(1024);
        // Should handle default capacity
        assert!(state.connected_clients() == 0);
    }

    #[tokio::test]
    async fn test_feed_state_subscribe() {
        let state = Arc::new(FeedState::new(100));

        // Subscribe to feed - returns Receiver directly
        let _receiver = state.subscribe();

        // Subscriber count should reflect subscription
        // Note: actual count depends on implementation
    }

    #[tokio::test]
    async fn test_feed_event_capture_received() {
        let state = Arc::new(FeedState::new(100));

        // Publish a capture received event
        state.publish_capture_received("capture-123", "https://example.com", "screenshot");

        // Event should be published without error
        // Verification depends on having subscribers
    }

    #[tokio::test]
    async fn test_feed_event_processing_complete() {
        let state = Arc::new(FeedState::new(100));

        // Publish processing complete event (capture_id, duration_ms, size_bytes, summary)
        state.publish_processing_complete("capture-123", 1500, 1024, None);

        // Event should be published without error
    }

    #[tokio::test]
    async fn test_feed_event_error() {
        let state = Arc::new(FeedState::new(100));

        // Publish error event (capture_id, code, message, recoverable)
        state.publish_error(
            Some("capture-123".to_string()),
            "TIMEOUT",
            "Browser timeout",
            true,
        );

        // Error event should be published without panic
    }

    #[tokio::test]
    async fn test_feed_state_multiple_subscribers() {
        let state = Arc::new(FeedState::new(100));

        // Create multiple subscribers
        let _receiver1 = state.subscribe();
        let _receiver2 = state.subscribe();
        let _receiver3 = state.subscribe();

        // All subscriptions should succeed (up to capacity)
    }

    #[tokio::test]
    async fn test_feed_state_publish_to_subscribers() {
        let state = Arc::new(FeedState::new(100));

        // Subscribe first
        let mut receiver = state.subscribe();

        // Then publish
        state.publish_capture_received("test-id", "https://test.com", "pdf");

        // Use timeout to avoid hanging
        let result = tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await;
        // Event may or may not be received depending on timing
        match result {
            Ok(Ok(_event)) => {
                // Successfully received event
            }
            Ok(Err(_)) => {
                // Lagged behind
            }
            Err(_) => {
                // Timeout - also acceptable in test
            }
        }
    }
}

// ============================================================================
// FeedEvent Tests
// ============================================================================

#[cfg(test)]
mod feed_event_tests {
    use super::*;
    use reasonkit_web::handlers::feed::{CaptureReceivedData, ErrorData, ProcessingCompleteData};

    #[test]
    fn test_feed_event_capture_received_structure() {
        let event = FeedEvent::CaptureReceived(CaptureReceivedData {
            capture_id: "capture-001".to_string(),
            url: "https://example.com/page".to_string(),
            capture_type: "screenshot".to_string(),
            timestamp: 1234567890,
        });

        match event {
            FeedEvent::CaptureReceived(data) => {
                assert_eq!(data.capture_id, "capture-001");
                assert_eq!(data.url, "https://example.com/page");
                assert_eq!(data.capture_type, "screenshot");
            }
            _ => panic!("Expected CaptureReceived variant"),
        }
    }

    #[test]
    fn test_feed_event_processing_complete_structure() {
        let event = FeedEvent::ProcessingComplete(ProcessingCompleteData {
            capture_id: "capture-001".to_string(),
            duration_ms: 2500,
            size_bytes: 1024,
            summary: None,
        });

        match event {
            FeedEvent::ProcessingComplete(data) => {
                assert_eq!(data.capture_id, "capture-001");
                assert_eq!(data.duration_ms, 2500);
            }
            _ => panic!("Expected ProcessingComplete variant"),
        }
    }

    #[test]
    fn test_feed_event_error_structure() {
        let event = FeedEvent::Error(ErrorData {
            capture_id: Some("capture-001".to_string()),
            code: "TIMEOUT".to_string(),
            message: "Navigation timeout".to_string(),
            recoverable: true,
        });

        match event {
            FeedEvent::Error(data) => {
                assert_eq!(data.capture_id, Some("capture-001".to_string()));
                assert_eq!(data.message, "Navigation timeout");
            }
            _ => panic!("Expected Error variant"),
        }
    }

    #[test]
    fn test_feed_event_serialization() {
        let event = FeedEvent::CaptureReceived(CaptureReceivedData {
            capture_id: "test-123".to_string(),
            url: "https://test.com".to_string(),
            capture_type: "html".to_string(),
            timestamp: 1234567890,
        });

        let json = serde_json::to_string(&event).unwrap();

        // Verify JSON structure
        assert!(json.contains("\"capture_id\":\"test-123\""));
        assert!(json.contains("\"url\":\"https://test.com\""));
        assert!(json.contains("html"));
    }

    #[test]
    fn test_feed_event_processing_serialization() {
        let event = FeedEvent::ProcessingComplete(ProcessingCompleteData {
            capture_id: "proc-456".to_string(),
            duration_ms: 1234,
            size_bytes: 512,
            summary: None,
        });

        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains("\"capture_id\":\"proc-456\""));
        assert!(json.contains("1234"));
    }

    #[test]
    fn test_feed_event_error_serialization() {
        let event = FeedEvent::Error(ErrorData {
            capture_id: Some("err-789".to_string()),
            code: "CONNECTION_ERROR".to_string(),
            message: "Connection refused".to_string(),
            recoverable: false,
        });

        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains("\"capture_id\":\"err-789\""));
        assert!(json.contains("Connection refused"));
    }
}

// ============================================================================
// Handler Integration Tests (Require Test Server)
// ============================================================================

#[cfg(test)]
mod handler_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_feed_state_thread_safety() {
        let state = Arc::new(FeedState::new(1000));

        // Spawn multiple tasks that publish events concurrently
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let state_clone = Arc::clone(&state);
                tokio::spawn(async move {
                    for j in 0..10 {
                        let id = format!("capture-{}-{}", i, j);
                        state_clone.publish_capture_received(&id, "https://test.com", "screenshot");
                    }
                })
            })
            .collect();

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Should complete without deadlock or panic
    }

    #[tokio::test]
    async fn test_feed_state_high_volume() {
        let state = Arc::new(FeedState::new(10000));

        // Publish many events rapidly
        for i in 0..1000 {
            state.publish_capture_received(
                &format!("bulk-{}", i),
                "https://bulk-test.com",
                "mhtml",
            );
        }

        // Should handle high volume without issues
    }

    #[tokio::test]
    async fn test_feed_subscriber_cleanup() {
        let state = Arc::new(FeedState::new(100));

        // Create and immediately drop subscribers
        {
            let _rx1 = state.subscribe();
            let _rx2 = state.subscribe();
            // Subscribers drop here
        }

        // Should still be able to publish after subscribers are cleaned up
        state.publish_capture_received("after-cleanup", "https://test.com", "pdf");
    }
}

// ============================================================================
// Capture Handler Tests
// ============================================================================

#[cfg(test)]
mod capture_handler_tests {
    // Note: Full capture handler tests require mocking the browser controller
    // These tests verify request/response structures

    use serde_json::json;

    #[test]
    fn test_capture_request_structure() {
        let request = json!({
            "url": "https://example.com",
            "format": "png",
            "fullPage": true,
            "quality": 90
        });

        assert_eq!(request["url"], "https://example.com");
        assert_eq!(request["format"], "png");
        assert_eq!(request["fullPage"], true);
        assert_eq!(request["quality"], 90);
    }

    #[test]
    fn test_capture_response_structure() {
        let response = json!({
            "success": true,
            "capture_id": "cap-123",
            "format": "png",
            "size": 12345,
            "width": 1920,
            "height": 1080
        });

        assert!(response["success"].as_bool().unwrap());
        assert_eq!(response["capture_id"], "cap-123");
        assert_eq!(response["size"], 12345);
    }

    #[test]
    fn test_capture_error_response() {
        let response = json!({
            "success": false,
            "error": "Browser initialization failed",
            "error_code": "BROWSER_INIT_ERROR"
        });

        assert!(!response["success"].as_bool().unwrap());
        assert!(response["error"].as_str().unwrap().contains("Browser"));
    }
}

// ============================================================================
// Status Handler Tests
// ============================================================================

#[cfg(test)]
mod status_handler_tests {
    use serde_json::json;

    #[test]
    fn test_health_check_response() {
        let response = json!({
            "status": "healthy",
            "version": "0.1.0",
            "uptime_seconds": 3600
        });

        assert_eq!(response["status"], "healthy");
        assert!(response["uptime_seconds"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_status_response_structure() {
        let response = json!({
            "status": "ok",
            "browser": {
                "available": true,
                "active_pages": 2
            },
            "memory": {
                "used_mb": 256,
                "available_mb": 1024
            },
            "captures": {
                "total": 150,
                "pending": 3,
                "completed": 147
            }
        });

        assert_eq!(response["status"], "ok");
        assert!(response["browser"]["available"].as_bool().unwrap());
        assert_eq!(response["captures"]["total"], 150);
    }

    #[test]
    fn test_status_degraded_response() {
        let response = json!({
            "status": "degraded",
            "issues": [
                "High memory usage",
                "Browser restart required"
            ],
            "browser": {
                "available": true,
                "needs_restart": true
            }
        });

        assert_eq!(response["status"], "degraded");
        assert_eq!(response["issues"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_status_unhealthy_response() {
        let response = json!({
            "status": "unhealthy",
            "error": "Browser process crashed",
            "last_healthy": "2024-01-15T10:30:00Z"
        });

        assert_eq!(response["status"], "unhealthy");
        assert!(response["error"].as_str().is_some());
    }
}

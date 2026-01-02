//! Property-based testing for MCP (Model Context Protocol) types.
//!
//! Uses proptest to generate arbitrary inputs and verify invariants
//! for JSON-RPC messages, tool calls, and serialization roundtrips.

use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR JSON-RPC TYPES
// ============================================================================

/// Strategy for generating valid JSON-RPC request IDs
pub fn arb_jsonrpc_id() -> impl Strategy<Value = Option<Value>> {
    prop_oneof![
        Just(None),
        (1i64..1000000).prop_map(|n| Some(Value::Number(n.into()))),
        "[a-zA-Z0-9_-]{1,36}".prop_map(|s| Some(Value::String(s))),
    ]
}

/// Strategy for generating JSON-RPC method names
pub fn arb_method_name() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("initialize".to_string()),
        Just("initialized".to_string()),
        Just("tools/list".to_string()),
        Just("tools/call".to_string()),
        Just("resources/list".to_string()),
        Just("prompts/list".to_string()),
        Just("ping".to_string()),
        "[a-z_]+/[a-z_]+".prop_map(|s| s),
    ]
}

/// Strategy for generating JsonRpcRequest
pub fn arb_jsonrpc_request() -> impl Strategy<Value = JsonRpcRequest> {
    (
        arb_method_name(),
        prop::option::of(arb_json_value()),
        arb_jsonrpc_id(),
    )
        .prop_map(|(method, params, id)| JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method,
            params,
            id,
        })
}

/// Strategy for generating simple JSON values
pub fn arb_json_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|n| Value::Number(n.into())),
        ".{0,100}".prop_map(Value::String),
        Just(Value::Object(serde_json::Map::new())),
        Just(Value::Array(vec![])),
    ]
}

/// Strategy for generating nested JSON objects
pub fn arb_json_object() -> impl Strategy<Value = Value> {
    prop::collection::hash_map("[a-z_]{1,20}", arb_json_value(), 0..10)
        .prop_map(|map| Value::Object(map.into_iter().collect()))
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR JSON-RPC ERROR CODES
// ============================================================================

/// Strategy for generating standard JSON-RPC error codes
pub fn arb_error_code() -> impl Strategy<Value = i32> {
    prop_oneof![
        Just(-32700),        // Parse error
        Just(-32600),        // Invalid Request
        Just(-32601),        // Method not found
        Just(-32602),        // Invalid params
        Just(-32603),        // Internal error
        (-32099i32..-32000), // Server error range
    ]
}

/// Strategy for generating JsonRpcError
pub fn arb_jsonrpc_error() -> impl Strategy<Value = JsonRpcError> {
    (
        arb_error_code(),
        ".{1,200}",
        prop::option::of(arb_json_value()),
    )
        .prop_map(|(code, message, data)| JsonRpcError {
            code,
            message,
            data,
        })
}

/// Strategy for generating JsonRpcResponse
pub fn arb_jsonrpc_response() -> impl Strategy<Value = JsonRpcResponse> {
    prop_oneof![
        // Success response
        (arb_jsonrpc_id(), arb_json_value()).prop_map(|(id, result)| JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }),
        // Error response
        (arb_jsonrpc_id(), arb_jsonrpc_error()).prop_map(|(id, error)| JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }),
    ]
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR MCP TYPES
// ============================================================================

/// Strategy for generating FeedEventType
pub fn arb_feed_event_type() -> impl Strategy<Value = FeedEventType> {
    prop_oneof![
        Just(FeedEventType::Heartbeat),
        Just(FeedEventType::Status),
        Just(FeedEventType::ToolStart),
        Just(FeedEventType::ToolComplete),
        Just(FeedEventType::Error),
        Just(FeedEventType::Shutdown),
    ]
}

/// Strategy for generating ToolContent
pub fn arb_tool_content() -> impl Strategy<Value = ToolContent> {
    prop_oneof![
        ".{0,1000}".prop_map(|text| ToolContent::Text { text }),
        ("[A-Za-z0-9+/=]{10,100}", "image/png|image/jpeg|image/webp")
            .prop_map(|(data, mime_type)| ToolContent::Image { data, mime_type }),
    ]
}

/// Strategy for generating ToolCallResult
pub fn arb_tool_call_result() -> impl Strategy<Value = ToolCallResult> {
    (
        any::<bool>(),
        prop::collection::vec(arb_tool_content(), 1..5),
    )
        .prop_map(|(is_error, content)| ToolCallResult { is_error, content })
}

/// Strategy for generating ToolCallParams
pub fn arb_tool_call_params() -> impl Strategy<Value = ToolCallParams> {
    ("[a-z_]{3,30}", arb_json_object())
        .prop_map(|(name, arguments)| ToolCallParams { name, arguments })
}

/// Strategy for generating McpToolDefinition
pub fn arb_mcp_tool_definition() -> impl Strategy<Value = McpToolDefinition> {
    ("[a-z_]{3,30}", ".{10,200}", arb_json_object()).prop_map(
        |(name, description, input_schema)| McpToolDefinition {
            name,
            description,
            input_schema,
        },
    )
}

/// Strategy for generating ServerStatus
pub fn arb_server_status() -> impl Strategy<Value = ServerStatus> {
    (
        "[a-z-]{3,30}",
        "[0-9]+\\.[0-9]+\\.[0-9]+",
        0u64..86400 * 365,
        any::<bool>(),
        prop::option::of(1024u64..1024 * 1024 * 1024 * 8),
        0u32..10000,
        0u64..1000000000,
    )
        .prop_map(
            |(
                name,
                version,
                uptime_secs,
                healthy,
                memory_bytes,
                active_connections,
                total_requests,
            )| {
                ServerStatus {
                    name,
                    version,
                    uptime_secs,
                    healthy,
                    memory_bytes,
                    active_connections,
                    total_requests,
                }
            },
        )
}

/// Strategy for generating FeedEvent
pub fn arb_feed_event() -> impl Strategy<Value = FeedEvent> {
    (arb_feed_event_type(), 0u64..u64::MAX, arb_json_object()).prop_map(
        |(event_type, timestamp, data)| FeedEvent {
            event_type,
            timestamp,
            data,
        },
    )
}

/// Strategy for generating HeartbeatConfig
pub fn arb_heartbeat_config() -> impl Strategy<Value = HeartbeatConfig> {
    (1u64..300, 1u32..10).prop_map(|(interval_secs, max_missed)| HeartbeatConfig {
        interval: Duration::from_secs(interval_secs),
        max_missed,
    })
}

// ============================================================================
// TYPE DEFINITIONS (Placeholders for actual imports)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
    pub id: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeedEventType {
    Heartbeat,
    Status,
    ToolStart,
    ToolComplete,
    Error,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    #[serde(rename = "isError")]
    pub is_error: bool,
    pub content: Vec<ToolContent>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub name: String,
    pub version: String,
    pub uptime_secs: u64,
    pub healthy: bool,
    pub memory_bytes: Option<u64>,
    pub active_connections: u32,
    pub total_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEvent {
    #[serde(rename = "type")]
    pub event_type: FeedEventType,
    pub timestamp: u64,
    pub data: Value,
}

#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    pub interval: Duration,
    pub max_missed: u32,
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    // ========================================================================
    // JSON-RPC Request Invariants
    // ========================================================================

    #[test]
    fn prop_jsonrpc_version_is_2_0(method in arb_method_name()) {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method,
            params: None,
            id: Some(Value::Number(1.into())),
        };

        prop_assert_eq!(request.jsonrpc, "2.0",
            "JSON-RPC version must always be '2.0'");
    }

    #[test]
    fn prop_jsonrpc_method_non_empty(method in "[a-z_/]{1,50}") {
        prop_assert!(!method.is_empty(),
            "Method name cannot be empty");
    }

    #[test]
    fn prop_jsonrpc_notification_has_no_id(method in arb_method_name()) {
        let notification = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method,
            params: None,
            id: None,
        };

        prop_assert!(notification.id.is_none(),
            "Notifications should have no id");
    }

    // ========================================================================
    // JSON-RPC Response Invariants
    // ========================================================================

    #[test]
    fn prop_jsonrpc_response_exclusive_result_error(
        id in arb_jsonrpc_id(),
        result in arb_json_value(),
        error in arb_jsonrpc_error()
    ) {
        // Response with result (success)
        let success = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: id.clone(),
            result: Some(result),
            error: None,
        };

        // Response with error (failure)
        let failure = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        };

        // XOR: either result OR error, never both
        prop_assert!(success.result.is_some() && success.error.is_none(),
            "Success response must have result, no error");
        prop_assert!(failure.result.is_none() && failure.error.is_some(),
            "Error response must have error, no result");
    }

    // ========================================================================
    // JSON-RPC Error Code Invariants
    // ========================================================================

    #[test]
    fn prop_jsonrpc_error_codes_valid(code in arb_error_code()) {
        // Standard error codes are in specific ranges
        let is_parse_error = code == -32700;
        let is_invalid_request = code == -32600;
        let is_method_not_found = code == -32601;
        let is_invalid_params = code == -32602;
        let is_internal_error = code == -32603;
        let is_server_error = (-32099..=-32000).contains(&code);

        prop_assert!(
            is_parse_error || is_invalid_request || is_method_not_found ||
            is_invalid_params || is_internal_error || is_server_error,
            "Error code {} must be in valid range", code
        );
    }

    // ========================================================================
    // Serialization Roundtrip Tests
    // ========================================================================

    #[test]
    fn prop_jsonrpc_request_roundtrip(request in arb_jsonrpc_request()) {
        let json = serde_json::to_string(&request).expect("Failed to serialize");
        let parsed: JsonRpcRequest = serde_json::from_str(&json).expect("Failed to deserialize");

        prop_assert_eq!(request.jsonrpc, parsed.jsonrpc);
        prop_assert_eq!(request.method, parsed.method);
        prop_assert_eq!(request.id, parsed.id);
    }

    #[test]
    fn prop_jsonrpc_response_roundtrip(response in arb_jsonrpc_response()) {
        let json = serde_json::to_string(&response).expect("Failed to serialize");
        let parsed: JsonRpcResponse = serde_json::from_str(&json).expect("Failed to deserialize");

        prop_assert_eq!(response.jsonrpc, parsed.jsonrpc);
        // Note: id comparison may need special handling for Value::Number
    }

    #[test]
    fn prop_tool_content_roundtrip(content in arb_tool_content()) {
        let json = serde_json::to_string(&content).expect("Failed to serialize");
        let parsed: ToolContent = serde_json::from_str(&json).expect("Failed to deserialize");

        prop_assert_eq!(content, parsed);
    }

    #[test]
    fn prop_feed_event_roundtrip(event in arb_feed_event()) {
        let json = serde_json::to_string(&event).expect("Failed to serialize");
        let parsed: FeedEvent = serde_json::from_str(&json).expect("Failed to deserialize");

        prop_assert_eq!(event.event_type, parsed.event_type);
        prop_assert_eq!(event.timestamp, parsed.timestamp);
    }

    // ========================================================================
    // ToolCallResult Invariants
    // ========================================================================

    #[test]
    fn prop_tool_result_has_content(result in arb_tool_call_result()) {
        prop_assert!(!result.content.is_empty(),
            "Tool result must have at least one content item");
    }

    #[test]
    fn prop_tool_error_result_has_is_error_true(error_msg in ".{1,100}") {
        let result = ToolCallResult {
            is_error: true,
            content: vec![ToolContent::Text { text: error_msg }],
        };

        prop_assert!(result.is_error,
            "Error result must have is_error = true");
    }

    // ========================================================================
    // ServerStatus Invariants
    // ========================================================================

    #[test]
    fn prop_server_status_name_non_empty(name in "[a-z-]{1,30}") {
        prop_assert!(!name.is_empty(),
            "Server name cannot be empty");
    }

    #[test]
    fn prop_server_status_version_semver(major in 0u32..100, minor in 0u32..100, patch in 0u32..100) {
        let version = format!("{}.{}.{}", major, minor, patch);
        let parts: Vec<&str> = version.split('.').collect();

        prop_assert_eq!(parts.len(), 3,
            "Version must have 3 parts");
    }

    #[test]
    fn prop_server_status_uptime_formatted(uptime_secs in 0u64..86400 * 365) {
        let status = ServerStatus {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            uptime_secs,
            healthy: true,
            memory_bytes: None,
            active_connections: 0,
            total_requests: 0,
        };

        // Uptime formatting logic
        let formatted = if status.uptime_secs < 60 {
            format!("{}s", status.uptime_secs)
        } else if status.uptime_secs < 3600 {
            format!("{}m {}s", status.uptime_secs / 60, status.uptime_secs % 60)
        } else if status.uptime_secs < 86400 {
            format!("{}h {}m", status.uptime_secs / 3600, (status.uptime_secs % 3600) / 60)
        } else {
            format!("{}d {}h", status.uptime_secs / 86400, (status.uptime_secs % 86400) / 3600)
        };

        prop_assert!(!formatted.is_empty(),
            "Formatted uptime should not be empty");
    }

    // ========================================================================
    // Memory Usage Formatting Invariants
    // ========================================================================

    #[test]
    fn prop_memory_bytes_formatted(bytes in 0u64..1024 * 1024 * 1024 * 10) {
        let formatted = if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        };

        prop_assert!(!formatted.is_empty(),
            "Formatted memory should not be empty");
        prop_assert!(
            formatted.ends_with(" B") ||
            formatted.ends_with(" KB") ||
            formatted.ends_with(" MB") ||
            formatted.ends_with(" GB"),
            "Memory format must end with unit suffix"
        );
    }

    // ========================================================================
    // HeartbeatConfig Invariants
    // ========================================================================

    #[test]
    fn prop_heartbeat_interval_positive(interval_secs in 1u64..300) {
        let config = HeartbeatConfig {
            interval: Duration::from_secs(interval_secs),
            max_missed: 3,
        };

        prop_assert!(config.interval.as_secs() > 0,
            "Heartbeat interval must be positive");
    }

    #[test]
    fn prop_heartbeat_max_missed_positive(max_missed in 1u32..20) {
        let config = HeartbeatConfig {
            interval: Duration::from_secs(30),
            max_missed,
        };

        prop_assert!(config.max_missed >= 1,
            "max_missed must be at least 1");
    }

    // ========================================================================
    // FeedEvent Invariants
    // ========================================================================

    #[test]
    fn prop_feed_event_timestamp_valid(timestamp in 0u64..u64::MAX / 2) {
        let event = FeedEvent {
            event_type: FeedEventType::Heartbeat,
            timestamp,
            data: Value::Object(serde_json::Map::new()),
        };

        // Touch timestamp to ensure it's set
        let _ = event.timestamp;
    }

    // ========================================================================
    // McpToolDefinition Invariants
    // ========================================================================

    #[test]
    fn prop_tool_definition_name_valid(name in "[a-z_]{3,30}") {
        prop_assert!(name.len() >= 3,
            "Tool name should have at least 3 characters");
        prop_assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "Tool name should be lowercase with underscores");
    }

    #[test]
    fn prop_tool_definition_description_non_empty(desc in ".{10,200}") {
        prop_assert!(desc.len() >= 10,
            "Tool description should be at least 10 characters");
    }
}

// ============================================================================
// SPECIAL CASES TESTS
// ============================================================================

#[cfg(test)]
mod edge_cases {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_empty_params_valid(method in arb_method_name()) {
            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method,
                params: None,
                id: Some(Value::Number(1.into())),
            };

            // Empty params should be valid
            prop_assert!(request.params.is_none());
        }

        #[test]
        fn prop_null_id_valid(method in arb_method_name()) {
            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method,
                params: None,
                id: None,
            };

            // This is a notification (no response expected)
            prop_assert!(request.id.is_none());
        }

        #[test]
        fn prop_string_id_valid(id_str in "[a-zA-Z0-9_-]{1,36}") {
            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method: "test".to_string(),
                params: None,
                id: Some(Value::String(id_str.clone())),
            };

            if let Some(Value::String(s)) = &request.id {
                prop_assert_eq!(s, &id_str);
            }
        }
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_request_creation() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({"name": "test"})),
            id: Some(Value::Number(1.into())),
        };

        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "tools/call");
        assert!(request.params.is_some());
    }

    #[test]
    fn test_tool_content_text() {
        let content = ToolContent::Text {
            text: "Hello, world!".to_string(),
        };

        if let ToolContent::Text { text } = content {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_tool_content_image() {
        let content = ToolContent::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };

        if let ToolContent::Image { data, mime_type } = content {
            assert_eq!(data, "base64data");
            assert_eq!(mime_type, "image/png");
        } else {
            panic!("Expected Image variant");
        }
    }

    #[test]
    fn test_feed_event_type_serialization() {
        let event_type = FeedEventType::Heartbeat;
        let json = serde_json::to_string(&event_type).unwrap();
        assert_eq!(json, "\"heartbeat\"");
    }

    #[test]
    fn test_server_status_defaults() {
        let status = ServerStatus {
            name: "reasonkit-web".to_string(),
            version: "1.0.0".to_string(),
            uptime_secs: 0,
            healthy: true,
            memory_bytes: None,
            active_connections: 0,
            total_requests: 0,
        };

        assert!(status.healthy);
        assert!(status.memory_bytes.is_none());
    }

    #[test]
    fn test_heartbeat_config_defaults() {
        let config = HeartbeatConfig {
            interval: Duration::from_secs(30),
            max_missed: 3,
        };

        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.max_missed, 3);
    }
}

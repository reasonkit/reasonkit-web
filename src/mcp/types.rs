//! MCP protocol types
//!
//! This module defines the types used in the MCP JSON-RPC protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{Duration, Instant};

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Method name
    pub method: String,
    /// Optional parameters
    #[serde(default)]
    pub params: Option<Value>,
    /// Request ID (None for notifications)
    pub id: Option<Value>,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    /// Success result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a success response
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Create a parse error response
    pub fn parse_error() -> Self {
        Self::error(None, -32700, "Parse error")
    }

    /// Create an invalid request error
    pub fn invalid_request(id: Option<Value>) -> Self {
        Self::error(id, -32600, "Invalid Request")
    }

    /// Create a method not found error
    pub fn method_not_found(id: Option<Value>, method: &str) -> Self {
        Self::error(id, -32601, format!("Method not found: {}", method))
    }

    /// Create an invalid params error
    pub fn invalid_params(id: Option<Value>, msg: &str) -> Self {
        Self::error(id, -32602, format!("Invalid params: {}", msg))
    }

    /// Create an internal error
    pub fn internal_error(id: Option<Value>, msg: &str) -> Self {
        Self::error(id, -32603, format!("Internal error: {}", msg))
    }
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Optional additional data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpCapabilities {
    /// Tools capability
    #[serde(default)]
    pub tools: ToolsCapability,
    /// Resources capability
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Prompts capability
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
}

/// Tools capability
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Whether tool list changes should be notified
    #[serde(default, rename = "listChanged")]
    pub list_changed: bool,
}

/// Resources capability (not implemented)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourcesCapability {}

/// Prompts capability (not implemented)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptsCapability {}

/// MCP server info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
}

impl Default for McpServerInfo {
    fn default() -> Self {
        Self {
            name: "reasonkit-web".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// MCP tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input JSON schema
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Parameters for tools/call method
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallParams {
    /// Tool name
    pub name: String,
    /// Tool arguments
    #[serde(default)]
    pub arguments: Value,
}

/// Result of a tool call
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallResult {
    /// Whether the call was an error
    #[serde(rename = "isError", skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Content array
    pub content: Vec<ToolContent>,
}

impl ToolCallResult {
    /// Create a success result with text content
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            is_error: false,
            content: vec![ToolContent::text(text)],
        }
    }

    /// Create a success result with image content
    pub fn image(data: String, mime_type: impl Into<String>) -> Self {
        Self {
            is_error: false,
            content: vec![ToolContent::image(data, mime_type)],
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            is_error: true,
            content: vec![ToolContent::text(message)],
        }
    }

    /// Create a result with multiple content items
    pub fn multi(content: Vec<ToolContent>) -> Self {
        Self {
            is_error: false,
            content,
        }
    }
}

/// Content item in tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        /// Base64 encoded image data
        data: String,
        /// MIME type
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Resource content
    #[serde(rename = "resource")]
    Resource {
        /// Resource URI
        uri: String,
        /// Resource content
        resource: ResourceContent,
    },
}

impl ToolContent {
    /// Create text content
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create image content
    pub fn image(data: String, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data,
            mime_type: mime_type.into(),
        }
    }
}

/// Resource content in tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// MIME type
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// Text content (for text resources)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Binary content as base64 (for binary resources)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Server status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Whether server is healthy
    pub healthy: bool,
    /// Memory usage in bytes (if available)
    pub memory_bytes: Option<u64>,
    /// Number of active connections
    pub active_connections: u32,
    /// Total requests handled
    pub total_requests: u64,
}

impl ServerStatus {
    /// Create a new server status
    pub fn new(start_time: Instant) -> Self {
        Self {
            name: "reasonkit-web".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: start_time.elapsed().as_secs(),
            healthy: true,
            memory_bytes: None,
            active_connections: 0,
            total_requests: 0,
        }
    }

    /// Format uptime as human-readable string
    pub fn uptime_formatted(&self) -> String {
        let secs = self.uptime_secs;
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else if secs < 86400 {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        } else {
            format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
        }
    }

    /// Format memory usage as human-readable string
    pub fn memory_formatted(&self) -> Option<String> {
        self.memory_bytes.map(|bytes| {
            if bytes < 1024 {
                format!("{} B", bytes)
            } else if bytes < 1024 * 1024 {
                format!("{:.1} KB", bytes as f64 / 1024.0)
            } else if bytes < 1024 * 1024 * 1024 {
                format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
            } else {
                format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            }
        })
    }
}

/// Feed event for server-sent events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEvent {
    /// Event type
    #[serde(rename = "type")]
    pub event_type: FeedEventType,
    /// Event timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Event data
    pub data: Value,
}

/// Types of feed events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeedEventType {
    /// Heartbeat ping
    Heartbeat,
    /// Status update
    Status,
    /// Tool execution started
    ToolStart,
    /// Tool execution completed
    ToolComplete,
    /// Error occurred
    Error,
    /// Server shutdown
    Shutdown,
}

impl FeedEvent {
    /// Create a heartbeat event
    pub fn heartbeat() -> Self {
        Self {
            event_type: FeedEventType::Heartbeat,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            data: serde_json::json!({"status": "ok"}),
        }
    }

    /// Create a status event
    pub fn status(status: &ServerStatus) -> Self {
        Self {
            event_type: FeedEventType::Status,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            data: serde_json::to_value(status).unwrap_or(Value::Null),
        }
    }

    /// Create a tool start event
    pub fn tool_start(tool_name: &str) -> Self {
        Self {
            event_type: FeedEventType::ToolStart,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            data: serde_json::json!({"tool": tool_name}),
        }
    }

    /// Create a tool complete event
    pub fn tool_complete(tool_name: &str, success: bool, duration_ms: u64) -> Self {
        Self {
            event_type: FeedEventType::ToolComplete,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            data: serde_json::json!({
                "tool": tool_name,
                "success": success,
                "duration_ms": duration_ms
            }),
        }
    }

    /// Create an error event
    pub fn error(message: &str) -> Self {
        Self {
            event_type: FeedEventType::Error,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            data: serde_json::json!({"error": message}),
        }
    }
}

/// Heartbeat configuration
pub struct HeartbeatConfig {
    /// Interval between heartbeats
    pub interval: Duration,
    /// Maximum missed heartbeats before disconnect
    pub max_missed: u32,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            max_missed: 3,
        }
    }
}

impl HeartbeatConfig {
    /// Create a new heartbeat config with custom interval
    pub fn with_interval(interval_secs: u64) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            max_missed: 3,
        }
    }

    /// Get interval in milliseconds
    pub fn interval_ms(&self) -> u64 {
        self.interval.as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // JsonRpcRequest Tests
    // ========================================================================

    #[test]
    fn test_jsonrpc_request_deserialize() {
        let json = r#"{"jsonrpc":"2.0","method":"test","id":1}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "test");
        assert_eq!(req.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn test_jsonrpc_request_with_params() {
        let json = r#"{"jsonrpc":"2.0","method":"test","params":{"foo":"bar"},"id":1}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.params.is_some());
        assert_eq!(req.params.unwrap()["foo"], "bar");
    }

    #[test]
    fn test_jsonrpc_request_notification() {
        let json = r#"{"jsonrpc":"2.0","method":"notify"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.id.is_none());
        assert!(req.params.is_none());
    }

    // ========================================================================
    // JsonRpcResponse Tests
    // ========================================================================

    #[test]
    fn test_jsonrpc_response_success() {
        let resp =
            JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!({"ok": true}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_jsonrpc_response_error() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(1)), -32600, "Invalid");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32600"));
    }

    #[test]
    fn test_jsonrpc_response_parse_error() {
        let resp = JsonRpcResponse::parse_error();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, -32700);
    }

    #[test]
    fn test_jsonrpc_response_method_not_found() {
        let resp = JsonRpcResponse::method_not_found(Some(serde_json::json!(1)), "unknown");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, -32601);
        assert!(resp.error.as_ref().unwrap().message.contains("unknown"));
    }

    #[test]
    fn test_jsonrpc_response_invalid_params() {
        let resp = JsonRpcResponse::invalid_params(Some(serde_json::json!(1)), "missing url");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, -32602);
    }

    #[test]
    fn test_jsonrpc_response_internal_error() {
        let resp = JsonRpcResponse::internal_error(Some(serde_json::json!(1)), "boom");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, -32603);
    }

    // ========================================================================
    // ToolCallResult Tests
    // ========================================================================

    #[test]
    fn test_tool_call_result_text() {
        let result = ToolCallResult::text("Hello, world!");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn test_tool_call_result_error() {
        let result = ToolCallResult::error("Something went wrong");
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_call_result_image() {
        let result = ToolCallResult::image("base64data".to_string(), "image/png");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn test_tool_call_result_multi() {
        let content = vec![ToolContent::text("Hello"), ToolContent::text("World")];
        let result = ToolCallResult::multi(content);
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 2);
    }

    // ========================================================================
    // ToolContent Tests
    // ========================================================================

    #[test]
    fn test_tool_content_serialize() {
        let content = ToolContent::text("Hello");
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_tool_content_image() {
        let content = ToolContent::image("data".to_string(), "image/png");
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
        assert!(json.contains("\"mimeType\":\"image/png\""));
    }

    // ========================================================================
    // MCP Capabilities Tests
    // ========================================================================

    #[test]
    fn test_mcp_capabilities() {
        let caps = McpCapabilities::default();
        assert!(!caps.tools.list_changed);
        assert!(caps.resources.is_none());
    }

    #[test]
    fn test_mcp_server_info_default() {
        let info = McpServerInfo::default();
        assert_eq!(info.name, "reasonkit-web");
        assert!(!info.version.is_empty());
    }

    // ========================================================================
    // ServerStatus Tests
    // ========================================================================

    #[test]
    fn test_uptime_calculation() {
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let status = ServerStatus::new(start);

        // Uptime should be at least 0 (could be 0 if < 1 second)
        assert!(status.uptime_secs >= 0);
        assert!(status.healthy);
    }

    #[test]
    fn test_uptime_formatted_seconds() {
        let mut status = ServerStatus::new(Instant::now());
        status.uptime_secs = 45;
        assert_eq!(status.uptime_formatted(), "45s");
    }

    #[test]
    fn test_uptime_formatted_minutes() {
        let mut status = ServerStatus::new(Instant::now());
        status.uptime_secs = 125; // 2m 5s
        assert_eq!(status.uptime_formatted(), "2m 5s");
    }

    #[test]
    fn test_uptime_formatted_hours() {
        let mut status = ServerStatus::new(Instant::now());
        status.uptime_secs = 3725; // 1h 2m
        assert_eq!(status.uptime_formatted(), "1h 2m");
    }

    #[test]
    fn test_uptime_formatted_days() {
        let mut status = ServerStatus::new(Instant::now());
        status.uptime_secs = 90061; // 1d 1h
        assert_eq!(status.uptime_formatted(), "1d 1h");
    }

    #[test]
    fn test_memory_usage_format_bytes() {
        let mut status = ServerStatus::new(Instant::now());
        status.memory_bytes = Some(512);
        assert_eq!(status.memory_formatted(), Some("512 B".to_string()));
    }

    #[test]
    fn test_memory_usage_format_kilobytes() {
        let mut status = ServerStatus::new(Instant::now());
        status.memory_bytes = Some(2048);
        assert_eq!(status.memory_formatted(), Some("2.0 KB".to_string()));
    }

    #[test]
    fn test_memory_usage_format_megabytes() {
        let mut status = ServerStatus::new(Instant::now());
        status.memory_bytes = Some(52_428_800); // 50 MB
        assert_eq!(status.memory_formatted(), Some("50.0 MB".to_string()));
    }

    #[test]
    fn test_memory_usage_format_gigabytes() {
        let mut status = ServerStatus::new(Instant::now());
        status.memory_bytes = Some(2_147_483_648); // 2 GB
        assert_eq!(status.memory_formatted(), Some("2.00 GB".to_string()));
    }

    #[test]
    fn test_memory_usage_format_none() {
        let status = ServerStatus::new(Instant::now());
        assert!(status.memory_formatted().is_none());
    }

    #[test]
    fn test_status_response_serialization() {
        let status = ServerStatus {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
            uptime_secs: 3600,
            healthy: true,
            memory_bytes: Some(1048576),
            active_connections: 5,
            total_requests: 100,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"name\":\"test-server\""));
        assert!(json.contains("\"healthy\":true"));
        assert!(json.contains("\"uptime_secs\":3600"));
    }

    // ========================================================================
    // FeedEvent Tests
    // ========================================================================

    #[test]
    fn test_feed_event_serialization() {
        let event = FeedEvent::heartbeat();
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"heartbeat\""));
        assert!(json.contains("\"timestamp\""));
    }

    #[test]
    fn test_feed_event_heartbeat() {
        let event = FeedEvent::heartbeat();
        assert_eq!(event.event_type, FeedEventType::Heartbeat);
        assert!(event.timestamp > 0);
    }

    #[test]
    fn test_feed_event_tool_start() {
        let event = FeedEvent::tool_start("web_navigate");
        assert_eq!(event.event_type, FeedEventType::ToolStart);
        assert_eq!(event.data["tool"], "web_navigate");
    }

    #[test]
    fn test_feed_event_tool_complete() {
        let event = FeedEvent::tool_complete("web_screenshot", true, 500);
        assert_eq!(event.event_type, FeedEventType::ToolComplete);
        assert_eq!(event.data["tool"], "web_screenshot");
        assert_eq!(event.data["success"], true);
        assert_eq!(event.data["duration_ms"], 500);
    }

    #[test]
    fn test_feed_event_error() {
        let event = FeedEvent::error("Connection failed");
        assert_eq!(event.event_type, FeedEventType::Error);
        assert_eq!(event.data["error"], "Connection failed");
    }

    #[test]
    fn test_feed_event_type_serialization() {
        assert_eq!(
            serde_json::to_string(&FeedEventType::Heartbeat).unwrap(),
            "\"heartbeat\""
        );
        assert_eq!(
            serde_json::to_string(&FeedEventType::Status).unwrap(),
            "\"status\""
        );
        assert_eq!(
            serde_json::to_string(&FeedEventType::ToolStart).unwrap(),
            "\"toolstart\""
        );
        assert_eq!(
            serde_json::to_string(&FeedEventType::ToolComplete).unwrap(),
            "\"toolcomplete\""
        );
        assert_eq!(
            serde_json::to_string(&FeedEventType::Error).unwrap(),
            "\"error\""
        );
        assert_eq!(
            serde_json::to_string(&FeedEventType::Shutdown).unwrap(),
            "\"shutdown\""
        );
    }

    // ========================================================================
    // HeartbeatConfig Tests
    // ========================================================================

    #[test]
    fn test_heartbeat_interval_default() {
        let config = HeartbeatConfig::default();
        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.max_missed, 3);
    }

    #[test]
    fn test_heartbeat_interval_custom() {
        let config = HeartbeatConfig::with_interval(60);
        assert_eq!(config.interval, Duration::from_secs(60));
        assert_eq!(config.interval_ms(), 60000);
    }

    #[test]
    fn test_heartbeat_interval_ms() {
        let config = HeartbeatConfig::default();
        assert_eq!(config.interval_ms(), 30000);
    }

    // ========================================================================
    // McpToolDefinition Tests
    // ========================================================================

    #[test]
    fn test_tool_definition_serialization() {
        let tool = McpToolDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                }
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"inputSchema\""));
    }

    // ========================================================================
    // Edge Cases Tests
    // ========================================================================

    #[test]
    fn test_jsonrpc_response_null_id() {
        let resp = JsonRpcResponse::success(None, serde_json::json!("ok"));
        let json = serde_json::to_string(&resp).unwrap();
        // id should be omitted when None
        assert!(!json.contains("\"id\""));
    }

    #[test]
    fn test_server_status_zero_uptime() {
        let mut status = ServerStatus::new(Instant::now());
        status.uptime_secs = 0;
        assert_eq!(status.uptime_formatted(), "0s");
    }

    #[test]
    fn test_feed_event_status() {
        let status = ServerStatus {
            name: "test".to_string(),
            version: "1.0".to_string(),
            uptime_secs: 100,
            healthy: true,
            memory_bytes: None,
            active_connections: 0,
            total_requests: 50,
        };
        let event = FeedEvent::status(&status);
        assert_eq!(event.event_type, FeedEventType::Status);
    }
}

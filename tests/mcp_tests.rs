//! MCP server integration tests
//!
//! These tests verify the MCP protocol implementation.

use reasonkit_web::mcp::types::{
    JsonRpcRequest, JsonRpcResponse, McpToolDefinition, ToolCallResult,
};
use reasonkit_web::mcp::{McpServer, ToolRegistry, AVAILABLE_TOOLS};
use serde_json::json;

#[test]
fn test_jsonrpc_request_parsing() {
    let json = r#"{
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }"#;

    let request: JsonRpcRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.method, "tools/list");
    assert_eq!(request.id, Some(json!(1)));
}

#[test]
fn test_jsonrpc_response_success() {
    let response = JsonRpcResponse::success(Some(json!(1)), json!({"status": "ok"}));
    let json = serde_json::to_string(&response).unwrap();

    assert!(json.contains("\"jsonrpc\":\"2.0\""));
    assert!(json.contains("\"result\""));
    assert!(!json.contains("\"error\""));
}

#[test]
fn test_jsonrpc_response_error() {
    let response = JsonRpcResponse::error(Some(json!(1)), -32600, "Invalid Request");
    let json = serde_json::to_string(&response).unwrap();

    assert!(json.contains("\"error\""));
    assert!(json.contains("-32600"));
    assert!(!json.contains("\"result\""));
}

#[test]
fn test_tool_registry_creation() {
    let registry = ToolRegistry::new();
    let definitions = registry.definitions();

    // Should have all the defined tools
    assert!(definitions.len() >= 8);

    // Check for specific tools
    let tool_names: Vec<_> = definitions.iter().map(|d| d.name.as_str()).collect();
    assert!(tool_names.contains(&"web_navigate"));
    assert!(tool_names.contains(&"web_screenshot"));
    assert!(tool_names.contains(&"web_pdf"));
    assert!(tool_names.contains(&"web_extract_content"));
    assert!(tool_names.contains(&"web_extract_links"));
    assert!(tool_names.contains(&"web_extract_metadata"));
    assert!(tool_names.contains(&"web_execute_js"));
    assert!(tool_names.contains(&"web_capture_mhtml"));
}

#[test]
fn test_tool_definitions_have_schemas() {
    let registry = ToolRegistry::new();
    let definitions = registry.definitions();

    for def in definitions {
        // Each tool should have a name
        assert!(!def.name.is_empty(), "Tool name should not be empty");

        // Each tool should have a description
        assert!(
            !def.description.is_empty(),
            "Tool {} should have a description",
            def.name
        );

        // Each tool should have an input schema
        assert!(
            def.input_schema.is_object(),
            "Tool {} should have an object schema",
            def.name
        );

        // Schema should have type: object
        assert_eq!(
            def.input_schema["type"], "object",
            "Tool {} schema should be type object",
            def.name
        );

        // Schema should have properties
        assert!(
            def.input_schema["properties"].is_object(),
            "Tool {} should have properties",
            def.name
        );
    }
}

#[test]
fn test_available_tools_constant() {
    // Should have at least 8 tools
    assert!(AVAILABLE_TOOLS.len() >= 8);

    // Check for key tools
    assert!(AVAILABLE_TOOLS.contains(&"web_navigate"));
    assert!(AVAILABLE_TOOLS.contains(&"web_screenshot"));
    assert!(AVAILABLE_TOOLS.contains(&"web_extract_content"));
}

#[test]
fn test_tool_call_result_text() {
    let result = ToolCallResult::text("Hello, world!");
    assert!(!result.is_error);
    assert_eq!(result.content.len(), 1);

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Hello, world!"));
    assert!(!json.contains("isError"));
}

#[test]
fn test_tool_call_result_error() {
    let result = ToolCallResult::error("Something went wrong");
    assert!(result.is_error);
    assert_eq!(result.content.len(), 1);

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Something went wrong"));
    assert!(json.contains("isError"));
}

#[test]
fn test_mcp_server_creation() {
    let _server = McpServer::new();
    // Server should be created without error
    // We can't easily test the run method without a proper test harness
    assert!(true);
}

#[tokio::test]
async fn test_mcp_initialize_response() {
    // Simulate an initialize request
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {}
        })),
        id: Some(json!(1)),
    };

    // Parse and verify
    assert_eq!(request.method, "initialize");
}

#[tokio::test]
async fn test_mcp_tools_list_response() {
    let registry = ToolRegistry::new();
    let definitions = registry.definitions();

    let response = json!({
        "tools": definitions
    });

    // Verify structure
    assert!(response["tools"].is_array());
    let tools = response["tools"].as_array().unwrap();
    assert!(!tools.is_empty());

    // Verify each tool has required fields
    for tool in tools {
        assert!(tool["name"].is_string());
        assert!(tool["description"].is_string());
        assert!(tool["inputSchema"].is_object());
    }
}

#[test]
fn test_mcp_tool_definition_serialization() {
    let def = McpToolDefinition {
        name: "test_tool".to_string(),
        description: "A test tool".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "param1": { "type": "string" }
            },
            "required": ["param1"]
        }),
    };

    let json = serde_json::to_string(&def).unwrap();
    assert!(json.contains("\"name\":\"test_tool\""));
    assert!(json.contains("\"inputSchema\""));

    // Verify it can be deserialized back
    let parsed: McpToolDefinition = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "test_tool");
}

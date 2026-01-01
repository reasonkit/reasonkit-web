//! MCP stdio server implementation
//!
//! This module implements the MCP server that communicates over stdio,
//! handling JSON-RPC requests and dispatching to registered tools.
//!
//! # Security
//!
//! The server supports optional token-based authentication via the
//! `REASONKIT_MCP_TOKEN` environment variable. When set, all requests
//! must include a valid authentication token in the request parameters
//! or the request will be rejected with an authentication error.
//!
//! ## Authentication Methods
//!
//! 1. **Request params**: Include `auth_token` in the request params object
//! 2. **Environment variable**: Set `REASONKIT_MCP_TOKEN` to enable authentication
//!
//! If `REASONKIT_MCP_TOKEN` is not set, authentication is disabled (backwards-compatible).

use crate::error::Result;
use crate::mcp::tools::ToolRegistry;
use crate::mcp::types::{
    JsonRpcRequest, JsonRpcResponse, McpCapabilities, McpServerInfo, ToolCallParams,
};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

/// Environment variable name for the MCP authentication token
const MCP_TOKEN_ENV_VAR: &str = "REASONKIT_MCP_TOKEN";

/// JSON-RPC error code for authentication failure (using -32000 range for server errors)
const AUTH_ERROR_CODE: i32 = -32001;

/// MCP server state
pub struct McpServer {
    /// Tool registry
    tools: ToolRegistry,
    /// Server info
    info: McpServerInfo,
    /// Whether the server has been initialized
    initialized: RwLock<bool>,
    /// Optional authentication token (loaded from REASONKIT_MCP_TOKEN env var)
    /// When Some, all requests must include a matching auth_token in params
    auth_token: Option<String>,
}

impl McpServer {
    /// Create a new MCP server
    ///
    /// Loads authentication token from `REASONKIT_MCP_TOKEN` environment variable if set.
    /// When a token is configured, all incoming requests must include a matching
    /// `auth_token` field in their params object.
    pub fn new() -> Self {
        let auth_token = std::env::var(MCP_TOKEN_ENV_VAR)
            .ok()
            .filter(|t| !t.is_empty());

        if auth_token.is_some() {
            info!(
                "MCP server authentication enabled via {}",
                MCP_TOKEN_ENV_VAR
            );
        } else {
            warn!(
                "MCP server running without authentication. Set {} to enable.",
                MCP_TOKEN_ENV_VAR
            );
        }

        Self {
            tools: ToolRegistry::new(),
            info: McpServerInfo::default(),
            initialized: RwLock::new(false),
            auth_token,
        }
    }

    /// Create a new MCP server with a specific authentication token
    ///
    /// This method is primarily for testing purposes. In production, use `new()`
    /// which loads the token from the environment variable.
    pub fn with_auth_token(token: impl Into<String>) -> Self {
        let token = token.into();
        let auth_token = if token.is_empty() { None } else { Some(token) };

        Self {
            tools: ToolRegistry::new(),
            info: McpServerInfo::default(),
            initialized: RwLock::new(false),
            auth_token,
        }
    }

    /// Check if authentication is enabled
    pub fn is_auth_enabled(&self) -> bool {
        self.auth_token.is_some()
    }

    /// Validate authentication for an incoming request
    ///
    /// # Authentication Logic
    ///
    /// 1. If no auth_token is configured (None), authentication passes (backwards-compatible)
    /// 2. If auth_token is configured, the request params must contain a matching `auth_token` field
    /// 3. Token comparison uses constant-time comparison to prevent timing attacks
    ///
    /// # Returns
    ///
    /// - `Ok(())` if authentication succeeds or is not required
    /// - `Err(JsonRpcResponse)` with authentication error if validation fails
    fn validate_auth(
        &self,
        request: &JsonRpcRequest,
    ) -> std::result::Result<(), Box<JsonRpcResponse>> {
        let expected_token = match &self.auth_token {
            Some(token) => token,
            None => return Ok(()), // No auth required if token not configured
        };

        // Extract auth_token from request params
        let provided_token = request
            .params
            .as_ref()
            .and_then(|p| p.get("auth_token"))
            .and_then(|v| v.as_str());

        match provided_token {
            Some(token) => {
                // Use constant-time comparison to prevent timing attacks
                if constant_time_compare(token, expected_token) {
                    debug!("Authentication successful for method: {}", request.method);
                    Ok(())
                } else {
                    warn!(
                        method = %request.method,
                        "Authentication failed: invalid token"
                    );
                    Err(Box::new(JsonRpcResponse::error(
                        request.id.clone(),
                        AUTH_ERROR_CODE,
                        "Authentication failed: invalid token",
                    )))
                }
            }
            None => {
                warn!(
                    method = %request.method,
                    "Authentication failed: missing auth_token in params"
                );
                Err(Box::new(JsonRpcResponse::error(
                    request.id.clone(),
                    AUTH_ERROR_CODE,
                    "Authentication required: missing auth_token in params",
                )))
            }
        }
    }

    /// Run the MCP server (blocking)
    #[instrument(skip(self))]
    pub async fn run(&self) -> Result<()> {
        info!(
            "Starting MCP server: {} v{}",
            self.info.name, self.info.version
        );

        if self.is_auth_enabled() {
            info!("Authentication is ENABLED - all requests require valid auth_token");
        } else {
            warn!("Authentication is DISABLED - accepting all requests");
        }

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    error!("Failed to read line: {}", e);
                    continue;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            let response = self.handle_line(&line).await;

            if let Some(resp) = response {
                let json = serde_json::to_string(&resp).unwrap_or_else(|e| {
                    error!("Failed to serialize response: {}", e);
                    r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Internal error"}}"#
                        .to_string()
                });

                debug!("Sending: {}", json);

                if let Err(e) = writeln!(stdout, "{}", json) {
                    error!("Failed to write response: {}", e);
                }
                if let Err(e) = stdout.flush() {
                    error!("Failed to flush stdout: {}", e);
                }
            }
        }

        info!("MCP server shutting down");
        Ok(())
    }

    /// Handle a single line of input
    async fn handle_line(&self, line: &str) -> Option<JsonRpcResponse> {
        // Try to parse as JSON-RPC request
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to parse request: {}", e);
                return Some(JsonRpcResponse::parse_error());
            }
        };

        // Handle the request
        self.handle_request(request).await
    }

    /// Handle a JSON-RPC request
    #[instrument(skip(self, request))]
    async fn handle_request(&self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = request.id.clone();
        let method = request.method.as_str();

        info!("Handling method: {}", method);

        // Validate authentication BEFORE processing any method
        // This prevents unauthenticated access to any server functionality
        if let Err(auth_error) = self.validate_auth(&request) {
            return Some(*auth_error);
        }

        let result = match method {
            // Lifecycle methods
            "initialize" => self.handle_initialize(request.params).await,
            "initialized" => {
                // Notification, no response needed
                return None;
            }
            "shutdown" => self.handle_shutdown().await,

            // Tool methods
            "tools/list" => self.handle_tools_list().await,
            "tools/call" => self.handle_tools_call(request.params).await,

            // Ping (for testing)
            "ping" => Ok(json!({ "pong": true })),

            // Unknown method
            _ => {
                warn!("Unknown method: {}", method);
                return Some(JsonRpcResponse::method_not_found(id, method));
            }
        };

        Some(match result {
            Ok(value) => JsonRpcResponse::success(id, value),
            Err(e) => JsonRpcResponse::internal_error(id, &e.to_string()),
        })
    }

    /// Handle initialize request
    async fn handle_initialize(&self, params: Option<Value>) -> Result<Value> {
        info!("Handling initialize");

        // Validate protocol version if provided
        if let Some(ref p) = params {
            if let Some(version) = p.get("protocolVersion").and_then(|v| v.as_str()) {
                debug!("Client protocol version: {}", version);
                // We support MCP protocol version 2024-11-05 and earlier
            }
        }

        *self.initialized.write().await = true;

        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": McpCapabilities::default(),
            "serverInfo": self.info
        }))
    }

    /// Handle shutdown request
    async fn handle_shutdown(&self) -> Result<Value> {
        info!("Handling shutdown");
        *self.initialized.write().await = false;
        Ok(json!(null))
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self) -> Result<Value> {
        let definitions = self.tools.definitions();
        Ok(json!({
            "tools": definitions
        }))
    }

    /// Handle tools/call request
    async fn handle_tools_call(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| crate::error::Error::generic("Missing params"))?;

        let tool_params: ToolCallParams = serde_json::from_value(params)
            .map_err(|e| crate::error::Error::generic(format!("Invalid params: {}", e)))?;

        let result = self
            .tools
            .execute(&tool_params.name, tool_params.arguments)
            .await;

        Ok(serde_json::to_value(result)?)
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Constant-time string comparison to prevent timing attacks
///
/// This function compares two strings in constant time, regardless of where
/// they differ. This prevents attackers from using timing information to
/// gradually discover the correct token character by character.
///
/// # Security Note
///
/// This is a critical security function. The comparison must:
/// 1. Always compare all bytes (no early exit)
/// 2. Take the same amount of time regardless of input
/// 3. Return a simple boolean result
fn constant_time_compare(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    // If lengths differ, we still need to compare to avoid timing leak
    // but we know the result will be false
    if a_bytes.len() != b_bytes.len() {
        // Still do a comparison to maintain constant time behavior
        // Use a dummy comparison against self
        let mut _dummy: u8 = 0;
        for byte in a_bytes.iter() {
            _dummy |= *byte; // Always 0, but compiler shouldn't optimize out
        }
        return false;
    }

    // XOR all bytes and accumulate differences
    let mut result: u8 = 0;
    for (x, y) in a_bytes.iter().zip(b_bytes.iter()) {
        result |= x ^ y;
    }

    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_compare_equal() {
        assert!(constant_time_compare("secret123", "secret123"));
        assert!(constant_time_compare("", ""));
        assert!(constant_time_compare("a", "a"));
    }

    #[test]
    fn test_constant_time_compare_unequal() {
        assert!(!constant_time_compare("secret123", "secret124"));
        assert!(!constant_time_compare("secret123", "Secret123"));
        assert!(!constant_time_compare("abc", "def"));
    }

    #[test]
    fn test_constant_time_compare_different_lengths() {
        assert!(!constant_time_compare("short", "longer"));
        assert!(!constant_time_compare("longer", "short"));
        assert!(!constant_time_compare("abc", ""));
    }

    #[tokio::test]
    async fn test_mcp_server_new() {
        // Clear any env var that might be set
        std::env::remove_var(MCP_TOKEN_ENV_VAR);
        let server = McpServer::new();
        assert_eq!(server.info.name, "reasonkit-web");
        assert!(!server.is_auth_enabled());
    }

    #[tokio::test]
    async fn test_mcp_server_with_auth_token() {
        let server = McpServer::with_auth_token("test-secret-token");
        assert!(server.is_auth_enabled());
    }

    #[tokio::test]
    async fn test_mcp_server_with_empty_auth_token() {
        let server = McpServer::with_auth_token("");
        assert!(!server.is_auth_enabled());
    }

    #[tokio::test]
    async fn test_validate_auth_no_token_configured() {
        let server = McpServer::with_auth_token("");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: None,
            id: Some(json!(1)),
        };

        assert!(server.validate_auth(&request).is_ok());
    }

    #[tokio::test]
    async fn test_validate_auth_valid_token() {
        let server = McpServer::with_auth_token("my-secret-token");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: Some(json!({ "auth_token": "my-secret-token" })),
            id: Some(json!(1)),
        };

        assert!(server.validate_auth(&request).is_ok());
    }

    #[tokio::test]
    async fn test_validate_auth_invalid_token() {
        let server = McpServer::with_auth_token("my-secret-token");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: Some(json!({ "auth_token": "wrong-token" })),
            id: Some(json!(1)),
        };

        let result = server.validate_auth(&request);
        assert!(result.is_err());
        let err_response = result.unwrap_err();
        assert!(err_response.error.is_some());
        assert_eq!(err_response.error.as_ref().unwrap().code, AUTH_ERROR_CODE);
        assert!(err_response
            .error
            .as_ref()
            .unwrap()
            .message
            .contains("invalid token"));
    }

    #[tokio::test]
    async fn test_validate_auth_missing_token() {
        let server = McpServer::with_auth_token("my-secret-token");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: None,
            id: Some(json!(1)),
        };

        let result = server.validate_auth(&request);
        assert!(result.is_err());
        let err_response = result.unwrap_err();
        assert!(err_response.error.is_some());
        assert_eq!(err_response.error.as_ref().unwrap().code, AUTH_ERROR_CODE);
        assert!(err_response
            .error
            .as_ref()
            .unwrap()
            .message
            .contains("missing auth_token"));
    }

    #[tokio::test]
    async fn test_validate_auth_token_in_params_but_not_string() {
        let server = McpServer::with_auth_token("my-secret-token");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: Some(json!({ "auth_token": 12345 })), // Number, not string
            id: Some(json!(1)),
        };

        let result = server.validate_auth(&request);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_request_with_auth_required() {
        let server = McpServer::with_auth_token("secret");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: None, // No auth token
            id: Some(json!(1)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.as_ref().unwrap().code, AUTH_ERROR_CODE);
    }

    #[tokio::test]
    async fn test_handle_request_with_valid_auth() {
        let server = McpServer::with_auth_token("secret");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: Some(json!({ "auth_token": "secret" })),
            id: Some(json!(1)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        assert!(response.result.unwrap()["pong"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_handle_ping() {
        std::env::remove_var(MCP_TOKEN_ENV_VAR);
        let server = McpServer::new();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: None,
            id: Some(json!(1)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        assert!(response.result.unwrap()["pong"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        std::env::remove_var(MCP_TOKEN_ENV_VAR);
        let server = McpServer::new();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": "2024-11-05"
            })),
            id: Some(json!(1)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["capabilities"].is_object());
        assert!(result["serverInfo"].is_object());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        std::env::remove_var(MCP_TOKEN_ENV_VAR);
        let server = McpServer::new();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/list".to_string(),
            params: None,
            id: Some(json!(2)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert!(result["tools"].is_array());
        assert!(!result["tools"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        std::env::remove_var(MCP_TOKEN_ENV_VAR);
        let server = McpServer::new();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "unknown/method".to_string(),
            params: None,
            id: Some(json!(3)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_handle_notification() {
        std::env::remove_var(MCP_TOKEN_ENV_VAR);
        let server = McpServer::new();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "initialized".to_string(),
            params: None,
            id: None, // Notification
        };

        let response = server.handle_request(request).await;
        assert!(response.is_none()); // Notifications don't get responses
    }

    #[tokio::test]
    async fn test_handle_initialize_with_auth() {
        let server = McpServer::with_auth_token("init-secret");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": "2024-11-05",
                "auth_token": "init-secret"
            })),
            id: Some(json!(1)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
    }

    #[tokio::test]
    async fn test_handle_tools_list_with_auth() {
        let server = McpServer::with_auth_token("list-secret");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/list".to_string(),
            params: Some(json!({ "auth_token": "list-secret" })),
            id: Some(json!(2)),
        };

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert!(result["tools"].is_array());
    }
}

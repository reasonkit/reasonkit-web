//! Cross-Crate Integration Tests for reasonkit-web
//!
//! Tests for the web sensing layer and MCP endpoint server.
//!
//! # Test Categories
//!
//! 1. **MCP Server**: Protocol initialization, tool handling
//! 2. **Content Processing**: HTML cleaning, text extraction
//! 3. **Capture Buffer**: In-memory capture storage
//! 4. **CORS Security**: Origin validation
//! 5. **Error Handling**: Error types, status codes
//! 6. **Shutdown**: Graceful shutdown handling
//!
//! # Running Tests
//!
//! ```bash
//! cargo test --package reasonkit-web --test cross_crate_integration_tests
//! ```

// ============================================================================
// MODULE: MCP Server Tests
// ============================================================================

mod mcp_server_tests {
    use reasonkit_web::mcp::{
        JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpCapabilities, McpServerInfo,
        McpToolDefinition, ToolCallParams, ToolCallResult, ToolContent,
    };

    /// Test: MCP server info structure
    #[test]
    fn test_mcp_server_info() {
        let info = McpServerInfo {
            name: "reasonkit-web".to_string(),
            version: reasonkit_web::VERSION.to_string(),
        };

        assert_eq!(info.name, "reasonkit-web");
        assert!(!info.version.is_empty());
    }

    /// Test: MCP server info default
    #[test]
    fn test_mcp_server_info_default() {
        let info = McpServerInfo::default();

        assert_eq!(info.name, "reasonkit-web");
        assert!(!info.version.is_empty());
    }

    /// Test: MCP capabilities structure
    #[test]
    fn test_mcp_capabilities() {
        let capabilities = McpCapabilities::default();

        // Should have default tools capability
        assert!(!capabilities.tools.list_changed);
    }

    /// Test: Tool definition structure
    #[test]
    fn test_tool_definition() {
        let tool_def = McpToolDefinition {
            name: "web_capture".to_string(),
            description: "Capture a web page".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to capture"
                    }
                },
                "required": ["url"]
            }),
        };

        assert_eq!(tool_def.name, "web_capture");
        assert!(tool_def.description.contains("Capture"));
        assert!(tool_def.input_schema["properties"]["url"].is_object());
    }

    /// Test: JSON-RPC request structure
    #[test]
    fn test_jsonrpc_request() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "web_capture",
                "arguments": {
                    "url": "https://example.com"
                }
            })),
        };

        assert_eq!(request.method, "tools/call");
        assert!(request.params.is_some());

        // Verify serialization
        let serialized = serde_json::to_string(&request);
        assert!(serialized.is_ok(), "Request should serialize");
    }

    /// Test: JSON-RPC response helpers
    #[test]
    fn test_jsonrpc_response_helpers() {
        // Success response
        let success = JsonRpcResponse::success(
            Some(serde_json::json!(1)),
            serde_json::json!({"content": "test"}),
        );
        assert!(success.result.is_some());
        assert!(success.error.is_none());

        // Error response
        let error = JsonRpcResponse::error(Some(serde_json::json!(2)), -32600, "Invalid Request");
        assert!(error.result.is_none());
        assert!(error.error.is_some());
        assert_eq!(error.error.as_ref().unwrap().code, -32600);

        // Parse error
        let parse_err = JsonRpcResponse::parse_error();
        assert!(parse_err.error.is_some());
        assert_eq!(parse_err.error.as_ref().unwrap().code, -32700);

        // Method not found
        let not_found = JsonRpcResponse::method_not_found(None, "unknown");
        assert!(not_found.error.is_some());
        assert_eq!(not_found.error.as_ref().unwrap().code, -32601);
    }

    /// Test: JSON-RPC error codes
    #[test]
    fn test_jsonrpc_error_codes() {
        let parse_error = JsonRpcError {
            code: -32700,
            message: "Parse error".to_string(),
            data: None,
        };
        assert_eq!(parse_error.code, -32700);

        let invalid_request = JsonRpcError {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        };
        assert_eq!(invalid_request.code, -32600);
    }

    /// Test: Tool call params structure
    #[test]
    fn test_tool_call_params() {
        let params = ToolCallParams {
            name: "web_extract".to_string(),
            arguments: serde_json::json!({
                "url": "https://example.com",
                "selector": "main"
            }),
        };

        assert_eq!(params.name, "web_extract");
        assert_eq!(params.arguments["selector"], "main");
    }

    /// Test: Tool result structure
    #[test]
    fn test_tool_result() {
        let result = ToolCallResult {
            content: vec![ToolContent::Text {
                text: "Extracted content".to_string(),
            }],
            is_error: false,
        };

        assert!(!result.content.is_empty());
        assert!(!result.is_error);

        // Verify serialization
        let serialized = serde_json::to_string(&result);
        assert!(serialized.is_ok());
    }

    /// Test: Tool content types
    #[test]
    fn test_tool_content_types() {
        // Text content
        let text_content = ToolContent::Text {
            text: "Extracted content".to_string(),
        };

        match text_content {
            ToolContent::Text { text } => {
                assert_eq!(text, "Extracted content");
            }
            _ => panic!("Expected text content"),
        }
    }
}

// ============================================================================
// MODULE: Content Processing Tests
// ============================================================================

mod content_processing_tests {
    use reasonkit_web::processing::ContentProcessor;

    /// Test: Content processor with defaults
    #[test]
    fn test_content_processor_defaults() {
        let processor = ContentProcessor::with_defaults();

        let html = r#"<html><body><p>Hello World</p></body></html>"#;
        let result = processor.process(html);

        assert!(result.text.contains("Hello World"));
        assert!(result.word_count > 0);
    }

    /// Test: HTML tag stripping
    #[test]
    fn test_html_tag_stripping() {
        let processor = ContentProcessor::with_defaults();

        let html = r#"<html>
            <head><title>Test</title></head>
            <body>
                <h1>Heading</h1>
                <p>Paragraph with <strong>bold</strong> text.</p>
            </body>
        </html>"#;

        let result = processor.process(html);

        assert!(result.text.contains("Heading"));
        assert!(result.text.contains("Paragraph"));
        assert!(!result.text.contains("<h1>"));
        assert!(!result.text.contains("<p>"));
    }

    /// Test: Script and style removal
    #[test]
    fn test_script_style_removal() {
        let processor = ContentProcessor::with_defaults();

        let html = r#"<html>
            <head>
                <script>alert('malicious');</script>
                <style>.hidden { display: none; }</style>
            </head>
            <body>
                <p>Clean content</p>
                <script>console.log('inline script');</script>
            </body>
        </html>"#;

        let result = processor.process(html);

        assert!(result.text.contains("Clean content"));
        assert!(!result.text.contains("alert"));
        assert!(!result.text.contains("malicious"));
    }

    /// Test: Processing time tracking
    #[test]
    fn test_processing_time_tracking() {
        let processor = ContentProcessor::with_defaults();

        let html = r#"<p>Some content to process</p>"#;
        let result = processor.process(html);

        assert!(
            result.processing_time_us > 0,
            "Should track processing time"
        );
    }

    /// Test: Word count accuracy
    #[test]
    fn test_word_count_accuracy() {
        let processor = ContentProcessor::with_defaults();

        let html = r#"<p>One two three four five</p>"#;
        let result = processor.process(html);

        assert_eq!(result.word_count, 5, "Should count 5 words");
    }

    /// Test: Empty input handling
    #[test]
    fn test_empty_input_handling() {
        let processor = ContentProcessor::with_defaults();

        let result = processor.process("");

        assert!(result.text.is_empty() || result.word_count == 0);
    }

    /// Test: Large content processing
    #[test]
    fn test_large_content_processing() {
        let processor = ContentProcessor::with_defaults();

        // Generate large HTML
        let mut html = String::from("<html><body>");
        for i in 0..1000 {
            html.push_str(&format!("<p>Paragraph {} with some content.</p>", i));
        }
        html.push_str("</body></html>");

        let result = processor.process(&html);

        assert!(result.word_count > 1000, "Should process large content");
    }
}

// ============================================================================
// MODULE: Capture Buffer Tests
// ============================================================================

mod capture_buffer_tests {
    use reasonkit_web::buffer::{CaptureBuffer, CaptureRecord};

    /// Test: Capture record creation
    #[test]
    fn test_capture_record_creation() {
        let record = CaptureRecord::new(
            "https://example.com".to_string(),
            "<html>...</html>".to_string(), // content
            "Extracted text".to_string(),   // processed_content
            1234,
        );

        assert_eq!(record.url, "https://example.com");
        assert!(record.content.contains("html"));
        assert!(record.processed_content.contains("Extracted"));
        assert_eq!(record.processing_time_us, 1234);
    }

    /// Test: Buffer push and get
    #[tokio::test]
    async fn test_buffer_push_and_get() {
        let buffer = CaptureBuffer::new();

        let record = CaptureRecord::new(
            "https://test.com".to_string(),
            "<html>Test</html>".to_string(),
            "Test content".to_string(),
            100,
        );

        buffer.push(record).await;

        let recent = buffer.get_recent(10).await;
        assert!(!recent.is_empty(), "Buffer should have records");
        assert_eq!(recent[0].url, "https://test.com");
    }

    /// Test: Buffer push multiple records
    #[tokio::test]
    async fn test_buffer_push_multiple() {
        let buffer = CaptureBuffer::new();

        for i in 0..10 {
            let record = CaptureRecord::new(
                format!("https://example{}.com", i),
                format!("<html>{}</html>", i),
                format!("Text {}", i),
                100,
            );
            buffer.push(record).await;
        }

        let recent = buffer.get_recent(100).await;
        assert!(!recent.is_empty(), "Buffer should have records");
    }
}

// ============================================================================
// MODULE: CORS Security Tests
// ============================================================================

mod cors_security_tests {
    use http::HeaderValue;
    use reasonkit_web::cors::{is_localhost_origin, CorsConfig};

    /// Test: Localhost origin validation
    #[test]
    fn test_localhost_origin_validation() {
        // Valid localhost origins
        let localhost = HeaderValue::from_static("http://localhost");
        assert!(is_localhost_origin(&localhost));

        let localhost_port = HeaderValue::from_static("http://localhost:3000");
        assert!(is_localhost_origin(&localhost_port));

        let loopback = HeaderValue::from_static("http://127.0.0.1");
        assert!(is_localhost_origin(&loopback));

        let loopback_port = HeaderValue::from_static("http://127.0.0.1:8080");
        assert!(is_localhost_origin(&loopback_port));

        // Invalid origins
        let external = HeaderValue::from_static("http://example.com");
        assert!(!is_localhost_origin(&external));

        let attacker = HeaderValue::from_static("https://attacker.com");
        assert!(!is_localhost_origin(&attacker));
    }

    /// Test: CORS config builder
    #[test]
    fn test_cors_config_builder() {
        let config = CorsConfig::new()
            .with_max_age(7200)
            .with_allow_credentials(true);

        assert_eq!(config.max_age_secs, 7200);
        assert!(config.allow_credentials);
    }

    /// Test: Default CORS config
    #[test]
    fn test_default_cors_config() {
        let config = CorsConfig::new();

        // Should have reasonable defaults
        assert!(config.max_age_secs > 0);
    }
}

// ============================================================================
// MODULE: Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use reasonkit_web::error::{RequestContext, WebError};

    /// Test: Error creation helpers
    #[test]
    fn test_error_creation() {
        let missing_field = WebError::missing_field("url");
        assert!(
            missing_field.to_string().to_lowercase().contains("url")
                || missing_field.to_string().to_lowercase().contains("missing")
        );

        let too_large = WebError::content_too_large(1024 * 1024 * 10, 1024 * 1024);
        let err_str = too_large.to_string().to_lowercase();
        assert!(
            err_str.contains("large") || err_str.contains("size") || err_str.contains("content")
        );
    }

    /// Test: HTTP status code mapping
    #[test]
    fn test_status_code_mapping() {
        let missing_field = WebError::missing_field("test");
        assert_eq!(missing_field.status_code(), 400);

        let too_large = WebError::content_too_large(100, 50);
        assert_eq!(too_large.status_code(), 413);
    }

    /// Test: Request context
    #[test]
    fn test_request_context() {
        let ctx = RequestContext::new();

        // Should have a request ID
        assert!(!ctx.request_id.is_empty());
    }

    /// Test: Error JSON serialization
    #[test]
    fn test_error_json_serialization() {
        let ctx = RequestContext::new();
        let error = WebError::missing_field("test_field");

        let json = error.to_json_with_request_id(&ctx.request_id);

        // Verify the JSON is a valid serde_json::Value and contains request_id
        let json_str = serde_json::to_string(&json).expect("Should serialize");
        assert!(json_str.contains(&ctx.request_id));
    }
}

// ============================================================================
// MODULE: Shutdown Controller Tests
// ============================================================================

mod shutdown_tests {
    use reasonkit_web::shutdown::{ShutdownController, ShutdownState};

    /// Test: Shutdown controller creation
    #[test]
    fn test_shutdown_controller_creation() {
        let controller = ShutdownController::new();

        assert!(!controller.is_shutting_down());
        assert_eq!(controller.state(), ShutdownState::Running);
    }

    /// Test: Shutdown state transition
    #[tokio::test]
    async fn test_shutdown_state_transition() {
        let controller = ShutdownController::new();

        assert!(!controller.is_shutting_down());

        controller.initiate_shutdown().await;

        assert!(controller.is_shutting_down());
        assert_eq!(controller.state(), ShutdownState::Stopped);
    }

    /// Test: Connection guard
    #[test]
    fn test_connection_guard() {
        let controller = ShutdownController::new();

        // Guard increments connection count
        let guard = controller.connection_guard();
        assert!(controller.active_connections() > 0);

        // Guard drop decrements
        drop(guard);
        assert_eq!(controller.active_connections(), 0);
    }

    /// Test: Multiple connection guards
    #[test]
    fn test_multiple_connection_guards() {
        let controller = ShutdownController::new();

        let _guard1 = controller.connection_guard();
        let _guard2 = controller.connection_guard();
        let _guard3 = controller.connection_guard();

        assert_eq!(controller.active_connections(), 3);

        drop(_guard1);
        assert_eq!(controller.active_connections(), 2);

        drop(_guard2);
        drop(_guard3);
        assert_eq!(controller.active_connections(), 0);
    }
}

// ============================================================================
// MODULE: Request Tracing Tests
// ============================================================================

mod tracing_tests {
    use reasonkit_web::generate_request_id;

    /// Test: Request ID generation
    #[test]
    fn test_request_id_generation() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();

        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
        assert_ne!(id1, id2, "Request IDs should be unique");
    }
}

// ============================================================================
// MODULE: SSE Feed Tests
// ============================================================================

mod sse_feed_tests {
    use reasonkit_web::handlers::feed::FeedState;
    use std::sync::Arc;

    /// Test: Feed state creation
    #[test]
    fn test_feed_state_creation() {
        let state = FeedState::new(1024);

        // State should be initialized - just verify construction works
        let _ = state;
    }

    /// Test: Event publishing
    #[test]
    fn test_event_publishing() {
        let state = Arc::new(FeedState::new(1024));

        state.publish_capture_received("capture-123", "https://example.com", "screenshot");

        // Event should be published without panic
    }
}

// ============================================================================
// MODULE: Metrics Tests
// ============================================================================

mod metrics_tests {
    use reasonkit_web::metrics::Metrics;

    /// Test: Metrics structure
    #[test]
    fn test_metrics_structure() {
        let metrics = Metrics::new();

        // Should have initial values - verify construction works
        let _ = metrics;
    }
}

// ============================================================================
// MODULE: Version and Constants Tests
// ============================================================================

mod version_tests {
    /// Test: Version constant
    #[test]
    fn test_version_constant() {
        assert!(!reasonkit_web::VERSION.is_empty());
        assert!(reasonkit_web::VERSION.contains('.'), "Should be semver");
    }

    /// Test: Name constant
    #[test]
    fn test_name_constant() {
        assert_eq!(reasonkit_web::NAME, "reasonkit-web");
    }
}

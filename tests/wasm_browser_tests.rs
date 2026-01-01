//! WASM Browser Tests for ReasonKit Web
//!
//! Comprehensive browser-based testing using wasm-bindgen-test.
//! Run with: `wasm-pack test --headless --chrome`
//!
//! Test Categories:
//! - DOM interaction and manipulation
//! - Fetch API for HTTP requests
//! - WebSocket connectivity and messaging
//! - Browser storage APIs
//! - Event handling

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// =============================================================================
// DOM INTERACTION TESTS
// =============================================================================

mod dom_tests {
    use super::*;
    use web_sys::{Document, Element, HtmlElement, Window};

    /// Helper to get window and document
    fn get_window_document() -> (Window, Document) {
        let window = web_sys::window().expect("no global window exists");
        let document = window.document().expect("should have a document on window");
        (window, document)
    }

    #[wasm_bindgen_test]
    fn test_document_exists() {
        let (window, document) = get_window_document();
        assert!(window.document().is_some());
        assert!(document.body().is_some());
    }

    #[wasm_bindgen_test]
    fn test_create_element() {
        let (_window, document) = get_window_document();

        let div = document
            .create_element("div")
            .expect("should create div element");

        assert_eq!(div.tag_name().to_lowercase(), "div");
    }

    #[wasm_bindgen_test]
    fn test_set_element_attributes() {
        let (_window, document) = get_window_document();

        let div = document
            .create_element("div")
            .expect("should create div element");

        div.set_attribute("id", "test-element")
            .expect("should set id attribute");
        div.set_attribute("class", "rk-container")
            .expect("should set class attribute");
        div.set_attribute("data-rk-component", "browser-connector")
            .expect("should set data attribute");

        assert_eq!(div.get_attribute("id"), Some("test-element".to_string()));
        assert_eq!(div.get_attribute("class"), Some("rk-container".to_string()));
        assert_eq!(
            div.get_attribute("data-rk-component"),
            Some("browser-connector".to_string())
        );
    }

    #[wasm_bindgen_test]
    fn test_append_child_to_body() {
        let (_window, document) = get_window_document();
        let body = document.body().expect("document should have a body");

        let div = document
            .create_element("div")
            .expect("should create div element");
        div.set_attribute("id", "rk-wasm-test-div")
            .expect("should set id");

        body.append_child(&div)
            .expect("should append child to body");

        let retrieved = document.get_element_by_id("rk-wasm-test-div");
        assert!(retrieved.is_some());

        // Cleanup
        body.remove_child(&div).expect("should remove child");
    }

    #[wasm_bindgen_test]
    fn test_inner_html_manipulation() {
        let (_window, document) = get_window_document();
        let body = document.body().expect("document should have a body");

        let container = document
            .create_element("div")
            .expect("should create container");
        container
            .set_attribute("id", "rk-html-test")
            .expect("should set id");

        body.append_child(&container)
            .expect("should append container");

        // Set innerHTML
        container.set_inner_html("<span>ReasonKit Test</span>");

        assert!(container.inner_html().contains("ReasonKit Test"));

        // Cleanup
        body.remove_child(&container).expect("should cleanup");
    }

    #[wasm_bindgen_test]
    fn test_query_selector() {
        let (_window, document) = get_window_document();
        let body = document.body().expect("document should have a body");

        let div = document.create_element("div").expect("create div");
        div.set_attribute("class", "rk-query-test")
            .expect("set class");
        div.set_inner_html("Query Target");

        body.append_child(&div).expect("append");

        let result = document
            .query_selector(".rk-query-test")
            .expect("query should not fail");
        assert!(result.is_some());

        // Cleanup
        body.remove_child(&div).expect("cleanup");
    }

    #[wasm_bindgen_test]
    fn test_query_selector_all() {
        let (_window, document) = get_window_document();
        let body = document.body().expect("document should have a body");

        // Create multiple elements
        for i in 0..3 {
            let div = document.create_element("div").expect("create div");
            div.set_attribute("class", "rk-multi-test")
                .expect("set class");
            div.set_attribute("id", &format!("rk-multi-{}", i))
                .expect("set id");
            body.append_child(&div).expect("append");
        }

        let node_list = document
            .query_selector_all(".rk-multi-test")
            .expect("query all should not fail");

        assert_eq!(node_list.length(), 3);

        // Cleanup
        for i in 0..3 {
            if let Some(el) = document.get_element_by_id(&format!("rk-multi-{}", i)) {
                body.remove_child(&el).expect("cleanup");
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_element_style_manipulation() {
        let (_window, document) = get_window_document();
        let body = document.body().expect("document should have a body");

        let div = document.create_element("div").expect("create div");
        div.set_attribute("id", "rk-style-test").expect("set id");

        body.append_child(&div).expect("append");

        // Cast to HtmlElement for style access
        let html_div: HtmlElement = div.clone().dyn_into().expect("should be HtmlElement");
        let style = html_div.style();

        style
            .set_property("background-color", "#06b6d4")
            .expect("set bg color");
        style
            .set_property("color", "#f9fafb")
            .expect("set text color");
        style.set_property("padding", "16px").expect("set padding");

        assert_eq!(
            style.get_property_value("background-color").ok(),
            Some("rgb(6, 182, 212)".to_string()).or(Some("#06b6d4".to_string()))
        );

        // Cleanup
        body.remove_child(&div).expect("cleanup");
    }

    #[wasm_bindgen_test]
    fn test_class_list_operations() {
        let (_window, document) = get_window_document();

        let div = document.create_element("div").expect("create div");
        let class_list = div.class_list();

        class_list.add_1("rk-active").expect("add class");
        assert!(class_list.contains("rk-active"));

        class_list
            .add_2("rk-visible", "rk-primary")
            .expect("add multiple");
        assert!(class_list.contains("rk-visible"));
        assert!(class_list.contains("rk-primary"));

        class_list.remove_1("rk-active").expect("remove class");
        assert!(!class_list.contains("rk-active"));

        class_list.toggle("rk-toggled").expect("toggle class");
        assert!(class_list.contains("rk-toggled"));

        class_list.toggle("rk-toggled").expect("toggle again");
        assert!(!class_list.contains("rk-toggled"));
    }

    #[wasm_bindgen_test]
    fn test_text_content() {
        let (_window, document) = get_window_document();

        let p = document.create_element("p").expect("create p");
        p.set_text_content(Some("ReasonKit Browser Connector - Text Content Test"));

        assert_eq!(
            p.text_content(),
            Some("ReasonKit Browser Connector - Text Content Test".to_string())
        );
    }

    #[wasm_bindgen_test]
    fn test_parent_child_relationships() {
        let (_window, document) = get_window_document();
        let body = document.body().expect("document should have a body");

        let parent = document.create_element("div").expect("create parent");
        parent
            .set_attribute("id", "rk-parent-test")
            .expect("set id");

        let child = document.create_element("span").expect("create child");
        child.set_text_content(Some("Child Element"));

        parent.append_child(&child).expect("append child");
        body.append_child(&parent).expect("append parent to body");

        assert!(child.parent_element().is_some());
        assert_eq!(
            child.parent_element().unwrap().id(),
            "rk-parent-test".to_string()
        );

        // Cleanup
        body.remove_child(&parent).expect("cleanup");
    }
}

// =============================================================================
// FETCH API TESTS
// =============================================================================

mod fetch_tests {
    use super::*;
    use js_sys::{Array, Object, Promise, Reflect};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    /// Helper to create a fetch request
    fn create_request(url: &str, method: &str) -> Result<Request, JsValue> {
        let mut opts = RequestInit::new();
        opts.method(method);
        opts.mode(RequestMode::Cors);

        Request::new_with_str_and_init(url, &opts)
    }

    #[wasm_bindgen_test]
    async fn test_fetch_get_request() {
        let window = web_sys::window().expect("no global window");

        // Using httpbin for testing - a reliable echo service
        let request =
            create_request("https://httpbin.org/get", "GET").expect("should create request");

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("fetch should succeed");

        let resp: Response = resp_value.dyn_into().expect("response should be Response");

        assert!(resp.ok());
        assert_eq!(resp.status(), 200);
    }

    #[wasm_bindgen_test]
    async fn test_fetch_post_request_with_json() {
        let window = web_sys::window().expect("no global window");

        let mut opts = RequestInit::new();
        opts.method("POST");
        opts.mode(RequestMode::Cors);

        // Create JSON body
        let body = JsValue::from_str(r#"{"test": "reasonkit", "value": 42}"#);
        opts.body(Some(&body));

        // Set headers
        let headers = web_sys::Headers::new().expect("create headers");
        headers
            .set("Content-Type", "application/json")
            .expect("set content-type");
        opts.headers(&headers.into());

        let request = Request::new_with_str_and_init("https://httpbin.org/post", &opts)
            .expect("create request");

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("fetch should succeed");

        let resp: Response = resp_value.dyn_into().expect("response cast");

        assert!(resp.ok());
        assert_eq!(resp.status(), 200);

        // Parse response JSON
        let json_promise = resp.json().expect("get json promise");
        let json_value = JsFuture::from(json_promise).await.expect("parse json");

        // Verify the response contains our data
        let data = Reflect::get(&json_value, &JsValue::from_str("data")).expect("get data field");
        assert!(data.is_string());
    }

    #[wasm_bindgen_test]
    async fn test_fetch_response_headers() {
        let window = web_sys::window().expect("no global window");

        let request = create_request("https://httpbin.org/response-headers?X-RK-Test=true", "GET")
            .expect("create request");

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("fetch should succeed");

        let resp: Response = resp_value.dyn_into().expect("response cast");
        let headers = resp.headers();

        // httpbin echoes the query params as response headers
        let rk_header = headers.get("X-RK-Test").expect("get header");
        assert!(rk_header.is_some());
    }

    #[wasm_bindgen_test]
    async fn test_fetch_text_response() {
        let window = web_sys::window().expect("no global window");

        let request =
            create_request("https://httpbin.org/robots.txt", "GET").expect("create request");

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("fetch should succeed");

        let resp: Response = resp_value.dyn_into().expect("response cast");
        assert!(resp.ok());

        let text_promise = resp.text().expect("get text promise");
        let text_value = JsFuture::from(text_promise).await.expect("get text");

        let text = text_value.as_string().expect("convert to string");
        assert!(!text.is_empty());
    }

    #[wasm_bindgen_test]
    async fn test_fetch_status_codes() {
        let window = web_sys::window().expect("no global window");

        // Test 404 status
        let request =
            create_request("https://httpbin.org/status/404", "GET").expect("create request");

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("fetch should complete");

        let resp: Response = resp_value.dyn_into().expect("response cast");

        assert!(!resp.ok());
        assert_eq!(resp.status(), 404);
    }

    #[wasm_bindgen_test]
    async fn test_fetch_with_timeout_abort() {
        let window = web_sys::window().expect("no global window");

        // Create an AbortController for timeout
        let abort_controller = web_sys::AbortController::new().expect("create abort controller");
        let signal = abort_controller.signal();

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);
        opts.signal(Some(&signal));

        let request = Request::new_with_str_and_init("https://httpbin.org/delay/10", &opts)
            .expect("create request");

        // Abort after 100ms (simulating timeout)
        let controller_clone = abort_controller.clone();
        let timeout_closure = Closure::once(move || {
            controller_clone.abort();
        });

        window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                timeout_closure.as_ref().unchecked_ref(),
                100,
            )
            .expect("set timeout");
        timeout_closure.forget();

        // The fetch should be aborted
        let result = JsFuture::from(window.fetch_with_request(&request)).await;

        // Should fail due to abort
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    async fn test_fetch_redirect_handling() {
        let window = web_sys::window().expect("no global window");

        // httpbin redirects to /get
        let request =
            create_request("https://httpbin.org/redirect/1", "GET").expect("create request");

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("fetch should succeed");

        let resp: Response = resp_value.dyn_into().expect("response cast");

        // Should follow redirect and return 200
        assert!(resp.ok());
        assert_eq!(resp.status(), 200);
        assert!(resp.redirected());
    }
}

// =============================================================================
// WEBSOCKET TESTS
// =============================================================================

mod websocket_tests {
    use super::*;
    use js_sys::Array;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

    /// WebSocket echo server for testing
    const WS_ECHO_URL: &str = "wss://echo.websocket.events";

    #[wasm_bindgen_test]
    async fn test_websocket_connection() {
        let ws = WebSocket::new(WS_ECHO_URL).expect("should create websocket");

        assert_eq!(ws.ready_state(), WebSocket::CONNECTING);

        // Wait for connection
        let connected = Rc::new(RefCell::new(false));
        let connected_clone = connected.clone();

        let onopen = Closure::once(Box::new(move |_: JsValue| {
            *connected_clone.borrow_mut() = true;
        }) as Box<dyn FnOnce(JsValue)>);

        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // Wait for connection (simple polling with timeout)
        let window = web_sys::window().expect("window");
        for _ in 0..50 {
            if *connected.borrow() || ws.ready_state() == WebSocket::OPEN {
                break;
            }
            let promise = js_sys::Promise::new(&mut |resolve, _| {
                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 100)
                    .expect("set timeout");
            });
            JsFuture::from(promise).await.expect("wait");
        }

        assert_eq!(ws.ready_state(), WebSocket::OPEN);

        // Clean up
        ws.close().expect("close websocket");
    }

    #[wasm_bindgen_test]
    async fn test_websocket_send_receive() {
        let ws = WebSocket::new(WS_ECHO_URL).expect("create websocket");

        let received = Rc::new(RefCell::new(None::<String>));
        let received_clone = received.clone();
        let ready = Rc::new(RefCell::new(false));
        let ready_clone = ready.clone();

        // Set up message handler
        let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Some(text) = e.data().as_string() {
                *received_clone.borrow_mut() = Some(text);
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        // Set up open handler
        let ws_clone = ws.clone();
        let onopen = Closure::once(Box::new(move |_: JsValue| {
            *ready_clone.borrow_mut() = true;
            ws_clone
                .send_with_str("ReasonKit WASM Test Message")
                .expect("send message");
        }) as Box<dyn FnOnce(JsValue)>);

        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // Wait for response
        let window = web_sys::window().expect("window");
        for _ in 0..100 {
            if received.borrow().is_some() {
                break;
            }
            let promise = js_sys::Promise::new(&mut |resolve, _| {
                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 100)
                    .expect("set timeout");
            });
            JsFuture::from(promise).await.expect("wait");
        }

        let msg = received.borrow();
        assert!(msg.is_some());
        // Echo server returns the same message
        assert!(msg.as_ref().unwrap().contains("ReasonKit"));

        ws.close().expect("close");
    }

    #[wasm_bindgen_test]
    async fn test_websocket_binary_message() {
        let ws = WebSocket::new(WS_ECHO_URL).expect("create websocket");
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        let received_binary = Rc::new(RefCell::new(false));
        let received_clone = received_binary.clone();
        let ready = Rc::new(RefCell::new(false));
        let ready_clone = ready.clone();

        // Set up message handler for binary
        let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
            if e.data().is_instance_of::<js_sys::ArrayBuffer>() {
                *received_clone.borrow_mut() = true;
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        // Set up open handler
        let ws_clone = ws.clone();
        let onopen = Closure::once(Box::new(move |_: JsValue| {
            *ready_clone.borrow_mut() = true;
            // Send binary data
            let data: [u8; 4] = [0x52, 0x4B, 0x21, 0x00]; // "RK!"
            ws_clone.send_with_u8_array(&data).expect("send binary");
        }) as Box<dyn FnOnce(JsValue)>);

        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // Wait for response
        let window = web_sys::window().expect("window");
        for _ in 0..100 {
            if *received_binary.borrow() {
                break;
            }
            let promise = js_sys::Promise::new(&mut |resolve, _| {
                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 100)
                    .expect("set timeout");
            });
            JsFuture::from(promise).await.expect("wait");
        }

        assert!(*received_binary.borrow());

        ws.close().expect("close");
    }

    #[wasm_bindgen_test]
    async fn test_websocket_close_event() {
        let ws = WebSocket::new(WS_ECHO_URL).expect("create websocket");

        let closed = Rc::new(RefCell::new(false));
        let closed_clone = closed.clone();
        let close_code = Rc::new(RefCell::new(0u16));
        let close_code_clone = close_code.clone();

        // Set up close handler
        let onclose = Closure::wrap(Box::new(move |e: CloseEvent| {
            *closed_clone.borrow_mut() = true;
            *close_code_clone.borrow_mut() = e.code();
        }) as Box<dyn FnMut(CloseEvent)>);

        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
        onclose.forget();

        // Set up open handler to close immediately
        let ws_clone = ws.clone();
        let onopen = Closure::once(Box::new(move |_: JsValue| {
            ws_clone
                .close_with_code_and_reason(1000, "Test complete")
                .expect("close with code");
        }) as Box<dyn FnOnce(JsValue)>);

        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // Wait for close
        let window = web_sys::window().expect("window");
        for _ in 0..50 {
            if *closed.borrow() {
                break;
            }
            let promise = js_sys::Promise::new(&mut |resolve, _| {
                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 100)
                    .expect("set timeout");
            });
            JsFuture::from(promise).await.expect("wait");
        }

        assert!(*closed.borrow());
        assert_eq!(*close_code.borrow(), 1000); // Normal closure
    }

    #[wasm_bindgen_test]
    fn test_websocket_url_parsing() {
        // Valid WebSocket URLs
        assert!(WebSocket::new("wss://example.com").is_ok());
        assert!(WebSocket::new("ws://localhost:8080").is_ok());
        assert!(WebSocket::new("wss://api.reasonkit.sh/ws").is_ok());

        // Invalid URLs should fail
        assert!(WebSocket::new("http://example.com").is_err());
        assert!(WebSocket::new("not-a-url").is_err());
    }

    #[wasm_bindgen_test]
    fn test_websocket_ready_states() {
        // Verify WebSocket constant values
        assert_eq!(WebSocket::CONNECTING, 0);
        assert_eq!(WebSocket::OPEN, 1);
        assert_eq!(WebSocket::CLOSING, 2);
        assert_eq!(WebSocket::CLOSED, 3);
    }
}

// =============================================================================
// BROWSER STORAGE TESTS
// =============================================================================

mod storage_tests {
    use super::*;
    use web_sys::Storage;

    fn get_local_storage() -> Storage {
        web_sys::window()
            .expect("window")
            .local_storage()
            .expect("local storage result")
            .expect("local storage")
    }

    fn get_session_storage() -> Storage {
        web_sys::window()
            .expect("window")
            .session_storage()
            .expect("session storage result")
            .expect("session storage")
    }

    #[wasm_bindgen_test]
    fn test_local_storage_set_get() {
        let storage = get_local_storage();

        storage
            .set_item("rk-test-key", "rk-test-value")
            .expect("set item");

        let value = storage.get_item("rk-test-key").expect("get item");
        assert_eq!(value, Some("rk-test-value".to_string()));

        // Cleanup
        storage.remove_item("rk-test-key").expect("remove item");
    }

    #[wasm_bindgen_test]
    fn test_local_storage_remove() {
        let storage = get_local_storage();

        storage
            .set_item("rk-remove-test", "value")
            .expect("set item");
        storage.remove_item("rk-remove-test").expect("remove item");

        let value = storage.get_item("rk-remove-test").expect("get item");
        assert!(value.is_none());
    }

    #[wasm_bindgen_test]
    fn test_local_storage_clear() {
        let storage = get_local_storage();

        // Set multiple items
        storage.set_item("rk-clear-1", "v1").expect("set 1");
        storage.set_item("rk-clear-2", "v2").expect("set 2");

        let initial_length = storage.length().expect("length");
        assert!(initial_length >= 2);

        storage.clear().expect("clear");

        let final_length = storage.length().expect("length after clear");
        assert_eq!(final_length, 0);
    }

    #[wasm_bindgen_test]
    fn test_session_storage_basic() {
        let storage = get_session_storage();

        storage
            .set_item("rk-session-key", "session-value")
            .expect("set");

        let value = storage.get_item("rk-session-key").expect("get");
        assert_eq!(value, Some("session-value".to_string()));

        storage.remove_item("rk-session-key").expect("cleanup");
    }

    #[wasm_bindgen_test]
    fn test_storage_key_iteration() {
        let storage = get_local_storage();
        storage.clear().expect("clear first");

        // Set known items
        storage.set_item("rk-iter-0", "val0").expect("set 0");
        storage.set_item("rk-iter-1", "val1").expect("set 1");
        storage.set_item("rk-iter-2", "val2").expect("set 2");

        let length = storage.length().expect("get length");
        assert_eq!(length, 3);

        // Iterate by index
        let mut keys = Vec::new();
        for i in 0..length {
            if let Some(key) = storage.key(i).expect("get key") {
                keys.push(key);
            }
        }

        assert_eq!(keys.len(), 3);
        assert!(keys.iter().any(|k| k.starts_with("rk-iter")));

        storage.clear().expect("cleanup");
    }

    #[wasm_bindgen_test]
    fn test_storage_json_serialization() {
        let storage = get_local_storage();

        // Store JSON string
        let json_data = r#"{"component":"browser-connector","version":"0.1.0","enabled":true}"#;
        storage
            .set_item("rk-json-config", json_data)
            .expect("set json");

        let retrieved = storage.get_item("rk-json-config").expect("get json");
        assert_eq!(retrieved, Some(json_data.to_string()));

        // Parse and verify (using js_sys)
        let parsed: serde_json::Value =
            serde_json::from_str(&retrieved.unwrap()).expect("parse json");
        assert_eq!(parsed["component"], "browser-connector");
        assert_eq!(parsed["version"], "0.1.0");
        assert_eq!(parsed["enabled"], true);

        storage.remove_item("rk-json-config").expect("cleanup");
    }
}

// =============================================================================
// EVENT HANDLING TESTS
// =============================================================================

mod event_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::JsCast;
    use web_sys::{CustomEvent, CustomEventInit, Event, EventTarget, HtmlElement, MouseEvent};

    #[wasm_bindgen_test]
    fn test_custom_event_creation() {
        let mut init = CustomEventInit::new();
        init.detail(&JsValue::from_str("ReasonKit Custom Event Data"));
        init.bubbles(true);
        init.cancelable(true);

        let event = CustomEvent::new_with_event_init_dict("rk-custom-event", &init)
            .expect("create custom event");

        assert_eq!(event.type_(), "rk-custom-event");
        assert!(event.bubbles());
        assert!(event.cancelable());
    }

    #[wasm_bindgen_test]
    fn test_event_listener_add_remove() {
        let (_window, document) = (
            web_sys::window().expect("window"),
            web_sys::window()
                .expect("window")
                .document()
                .expect("document"),
        );
        let body = document.body().expect("body");

        let div = document.create_element("div").expect("create div");
        div.set_attribute("id", "rk-event-target").expect("set id");
        body.append_child(&div).expect("append");

        let triggered = Rc::new(RefCell::new(false));
        let triggered_clone = triggered.clone();

        let callback = Closure::wrap(Box::new(move |_: Event| {
            *triggered_clone.borrow_mut() = true;
        }) as Box<dyn FnMut(Event)>);

        div.add_event_listener_with_callback("click", callback.as_ref().unchecked_ref())
            .expect("add listener");

        // Dispatch event
        let event = Event::new("click").expect("create click event");
        div.dispatch_event(&event).expect("dispatch event");

        assert!(*triggered.borrow());

        // Remove listener
        div.remove_event_listener_with_callback("click", callback.as_ref().unchecked_ref())
            .expect("remove listener");

        callback.forget();

        body.remove_child(&div).expect("cleanup");
    }

    #[wasm_bindgen_test]
    fn test_event_propagation_stop() {
        let (_window, document) = (
            web_sys::window().expect("window"),
            web_sys::window()
                .expect("window")
                .document()
                .expect("document"),
        );
        let body = document.body().expect("body");

        let outer = document.create_element("div").expect("create outer");
        outer.set_attribute("id", "rk-outer").expect("set id");

        let inner = document.create_element("div").expect("create inner");
        inner.set_attribute("id", "rk-inner").expect("set id");

        outer.append_child(&inner).expect("append inner");
        body.append_child(&outer).expect("append outer");

        let outer_triggered = Rc::new(RefCell::new(false));
        let outer_triggered_clone = outer_triggered.clone();

        // Outer listener
        let outer_callback = Closure::wrap(Box::new(move |_: Event| {
            *outer_triggered_clone.borrow_mut() = true;
        }) as Box<dyn FnMut(Event)>);

        outer
            .add_event_listener_with_callback("click", outer_callback.as_ref().unchecked_ref())
            .expect("add outer listener");

        // Inner listener that stops propagation
        let inner_callback = Closure::wrap(Box::new(move |e: Event| {
            e.stop_propagation();
        }) as Box<dyn FnMut(Event)>);

        inner
            .add_event_listener_with_callback("click", inner_callback.as_ref().unchecked_ref())
            .expect("add inner listener");

        // Dispatch on inner
        let event = Event::new("click").expect("create click");
        inner.dispatch_event(&event).expect("dispatch");

        // Outer should NOT be triggered due to stopPropagation
        assert!(!*outer_triggered.borrow());

        outer_callback.forget();
        inner_callback.forget();

        body.remove_child(&outer).expect("cleanup");
    }

    #[wasm_bindgen_test]
    fn test_event_prevent_default() {
        let event = Event::new("submit").expect("create submit event");

        // Initially cancelable
        assert!(!event.default_prevented());

        event.prevent_default();

        assert!(event.default_prevented());
    }

    #[wasm_bindgen_test]
    fn test_mouse_event_creation() {
        let init = web_sys::MouseEventInit::new();

        let event =
            MouseEvent::new_with_mouse_event_init_dict("click", &init).expect("create mouse event");

        assert_eq!(event.type_(), "click");
        assert_eq!(event.button(), 0); // Left button
    }

    #[wasm_bindgen_test]
    fn test_custom_event_with_detail() {
        let mut init = CustomEventInit::new();

        // Create a JS object for detail
        let detail = js_sys::Object::new();
        js_sys::Reflect::set(&detail, &"action".into(), &"capture".into()).expect("set action");
        js_sys::Reflect::set(
            &detail,
            &"timestamp".into(),
            &JsValue::from_f64(1234567890.0),
        )
        .expect("set timestamp");

        init.detail(&detail);

        let event =
            CustomEvent::new_with_event_init_dict("rk-action", &init).expect("create event");

        let retrieved_detail = event.detail();
        assert!(retrieved_detail.is_object());

        let action = js_sys::Reflect::get(&retrieved_detail, &"action".into()).expect("get action");
        assert_eq!(action.as_string(), Some("capture".to_string()));
    }
}

// =============================================================================
// BROWSER CONNECTOR INTEGRATION TESTS
// =============================================================================

mod browser_connector_tests {
    use super::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::console;

    /// Simulated browser connector state
    #[wasm_bindgen]
    pub struct BrowserConnectorState {
        connected: bool,
        url: String,
        captured_elements: u32,
    }

    #[wasm_bindgen]
    impl BrowserConnectorState {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                connected: false,
                url: String::new(),
                captured_elements: 0,
            }
        }

        pub fn connect(&mut self, url: &str) -> bool {
            self.url = url.to_string();
            self.connected = true;
            true
        }

        pub fn disconnect(&mut self) {
            self.connected = false;
            self.url.clear();
        }

        pub fn is_connected(&self) -> bool {
            self.connected
        }

        pub fn get_url(&self) -> String {
            self.url.clone()
        }

        pub fn capture_element(&mut self, _selector: &str) -> u32 {
            self.captured_elements += 1;
            self.captured_elements
        }

        pub fn get_captured_count(&self) -> u32 {
            self.captured_elements
        }
    }

    #[wasm_bindgen_test]
    fn test_connector_state_initialization() {
        let state = BrowserConnectorState::new();

        assert!(!state.is_connected());
        assert!(state.get_url().is_empty());
        assert_eq!(state.get_captured_count(), 0);
    }

    #[wasm_bindgen_test]
    fn test_connector_connect_disconnect() {
        let mut state = BrowserConnectorState::new();

        assert!(state.connect("https://example.com"));
        assert!(state.is_connected());
        assert_eq!(state.get_url(), "https://example.com");

        state.disconnect();
        assert!(!state.is_connected());
        assert!(state.get_url().is_empty());
    }

    #[wasm_bindgen_test]
    fn test_connector_element_capture() {
        let mut state = BrowserConnectorState::new();
        state.connect("https://example.com");

        let count1 = state.capture_element(".header");
        let count2 = state.capture_element(".content");
        let count3 = state.capture_element(".footer");

        assert_eq!(count1, 1);
        assert_eq!(count2, 2);
        assert_eq!(count3, 3);
        assert_eq!(state.get_captured_count(), 3);
    }

    #[wasm_bindgen_test]
    fn test_console_logging() {
        // Verify console API is accessible
        console::log_1(&"ReasonKit WASM Test: Console logging works".into());
        console::warn_1(&"ReasonKit WASM Test: Warning logging works".into());
        console::info_1(&"ReasonKit WASM Test: Info logging works".into());

        // Console group
        console::group_1(&"ReasonKit Test Group".into());
        console::log_1(&"Inside group".into());
        console::group_end();

        // Console time
        console::time_with_label("rk-operation");
        // Simulated operation
        let _ = (0..1000).sum::<i32>();
        console::time_end_with_label("rk-operation");
    }

    #[wasm_bindgen_test]
    async fn test_performance_api() {
        let window = web_sys::window().expect("window");
        let performance = window.performance().expect("performance API");

        let start = performance.now();

        // Simulate some work
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .expect("timeout");
        });
        JsFuture::from(promise).await.expect("wait");

        let end = performance.now();
        let duration = end - start;

        assert!(duration >= 10.0);
        assert!(duration < 1000.0); // Sanity check
    }

    #[wasm_bindgen_test]
    fn test_navigator_user_agent() {
        let window = web_sys::window().expect("window");
        let navigator = window.navigator();
        let user_agent = navigator.user_agent().expect("user agent");

        assert!(!user_agent.is_empty());
        // In wasm-pack test, this should contain "HeadlessChrome" or similar
        console::log_1(&format!("User Agent: {}", user_agent).into());
    }

    #[wasm_bindgen_test]
    fn test_location_api() {
        let window = web_sys::window().expect("window");
        let location = window.location();

        let protocol = location.protocol().expect("protocol");
        let host = location.host().expect("host");

        // In test environment
        assert!(!protocol.is_empty());
        console::log_1(&format!("Location: {}://{}", protocol, host).into());
    }
}

//! WASM Test Utilities for ReasonKit Web
//!
//! Helper functions and utilities for browser-based WASM testing.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Document, Element, Window};

/// Get the global window object
pub fn window() -> Window {
    web_sys::window().expect("no global window exists")
}

/// Get the document object
pub fn document() -> Document {
    window().document().expect("should have a document")
}

/// Create an element with optional ID and class
pub fn create_element(tag: &str, id: Option<&str>, class: Option<&str>) -> Element {
    let doc = document();
    let el = doc.create_element(tag).expect("should create element");

    if let Some(id_val) = id {
        el.set_attribute("id", id_val).expect("should set id");
    }

    if let Some(class_val) = class {
        el.set_attribute("class", class_val)
            .expect("should set class");
    }

    el
}

/// Append element to body and return cleanup closure
pub fn append_to_body(element: &Element) -> impl FnOnce() {
    let body = document().body().expect("should have body");
    body.append_child(element).expect("should append");

    let el_clone = element.clone();
    move || {
        if let Some(parent) = el_clone.parent_element() {
            parent.remove_child(&el_clone).ok();
        }
    }
}

/// Wait for specified milliseconds
pub async fn wait_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        window()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms)
            .expect("set timeout");
    });
    JsFuture::from(promise).await.expect("wait");
}

/// Wait for element to appear in DOM
pub async fn wait_for_element(selector: &str, timeout_ms: i32) -> Option<Element> {
    let interval = 50;
    let iterations = timeout_ms / interval;

    for _ in 0..iterations {
        if let Ok(Some(el)) = document().query_selector(selector) {
            return Some(el);
        }
        wait_ms(interval).await;
    }

    None
}

/// Wait for element to contain specific text
pub async fn wait_for_text(selector: &str, text: &str, timeout_ms: i32) -> bool {
    let interval = 50;
    let iterations = timeout_ms / interval;

    for _ in 0..iterations {
        if let Ok(Some(el)) = document().query_selector(selector) {
            if let Some(content) = el.text_content() {
                if content.contains(text) {
                    return true;
                }
            }
        }
        wait_ms(interval).await;
    }

    false
}

/// Create a test container with unique ID
pub fn create_test_container(test_name: &str) -> Element {
    let id = format!("rk-test-{}", test_name);
    let container = create_element("div", Some(&id), Some("rk-test-container"));

    // Style the container
    if let Ok(html_el) = container.clone().dyn_into::<web_sys::HtmlElement>() {
        let style = html_el.style();
        style.set_property("display", "none").ok();
    }

    let _cleanup = append_to_body(&container);
    container
}

/// Assert element exists with optional text content check
pub fn assert_element_exists(selector: &str, expected_text: Option<&str>) {
    let el = document()
        .query_selector(selector)
        .expect("query should not fail")
        .expect(&format!("Element '{}' should exist", selector));

    if let Some(text) = expected_text {
        let content = el.text_content().unwrap_or_default();
        assert!(
            content.contains(text),
            "Element '{}' should contain '{}', but contains '{}'",
            selector,
            text,
            content
        );
    }
}

/// Assert element does not exist
pub fn assert_element_not_exists(selector: &str) {
    let result = document()
        .query_selector(selector)
        .expect("query should not fail");
    assert!(result.is_none(), "Element '{}' should not exist", selector);
}

/// Get computed style property
pub fn get_computed_style(element: &Element, property: &str) -> Option<String> {
    let win = window();
    win.get_computed_style(element)
        .ok()
        .flatten()
        .and_then(|style| style.get_property_value(property).ok())
        .filter(|v| !v.is_empty())
}

/// Trigger click event on element
pub fn click_element(element: &Element) {
    let event = web_sys::MouseEvent::new("click").expect("create click event");
    element.dispatch_event(&event).expect("dispatch click");
}

/// Set input value
pub fn set_input_value(element: &Element, value: &str) {
    if let Ok(input) = element.clone().dyn_into::<web_sys::HtmlInputElement>() {
        input.set_value(value);

        // Dispatch input event
        let event = web_sys::Event::new("input").expect("create input event");
        input.dispatch_event(&event).expect("dispatch input");
    }
}

/// Get current timestamp in milliseconds
pub fn timestamp_ms() -> f64 {
    window().performance().expect("performance API").now()
}

/// Measure execution time of async operation
pub async fn measure_async<F, T>(f: F) -> (T, f64)
where
    F: std::future::Future<Output = T>,
{
    let start = timestamp_ms();
    let result = f.await;
    let end = timestamp_ms();
    (result, end - start)
}

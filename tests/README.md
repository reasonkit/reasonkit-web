# WASM Browser Tests for ReasonKit Web

Comprehensive browser-based testing suite using `wasm-bindgen-test`.

## Prerequisites

1. **Install wasm-pack**:
   ```bash
   cargo install wasm-pack
   ```

2. **Install Chrome/Chromium** (for headless testing):
   ```bash
   # Ubuntu/Debian
   sudo apt install chromium-browser

   # macOS
   brew install --cask chromium

   # Or use Chrome
   ```

## Running Tests

### Headless Chrome (Recommended for CI)

```bash
cd reasonkit-web
wasm-pack test --headless --chrome
```

### Headless Firefox

```bash
wasm-pack test --headless --firefox
```

### Interactive Browser (Debug)

```bash
wasm-pack test --chrome
# Opens browser with test UI - useful for debugging
```

### All Browsers

```bash
wasm-pack test --headless --chrome --firefox
```

## Test Categories

### DOM Interaction Tests (`dom_tests`)
- Element creation and manipulation
- Attribute handling
- Class list operations
- Parent/child relationships
- Query selectors
- Style manipulation

### Fetch API Tests (`fetch_tests`)
- GET/POST requests
- JSON response handling
- Headers and status codes
- Timeout/abort handling
- Redirect following

### WebSocket Tests (`websocket_tests`)
- Connection establishment
- Text message send/receive
- Binary message handling
- Close events and codes
- URL validation

### Storage Tests (`storage_tests`)
- LocalStorage operations
- SessionStorage operations
- JSON serialization
- Key iteration

### Event Tests (`event_tests`)
- Custom event creation
- Event listener add/remove
- Propagation control
- preventDefault handling

### Browser Connector Tests (`browser_connector_tests`)
- Connector state management
- Console API access
- Performance API
- Navigator/Location APIs

## Test Utilities

The `wasm_test_utils.rs` module provides helpers:

```rust
use wasm_test_utils::*;

// Wait for element
let el = wait_for_element(".my-class", 5000).await;

// Wait for text content
wait_for_text("#output", "Success", 3000).await;

// Create test containers
let container = create_test_container("my-test");

// Measure performance
let (result, duration_ms) = measure_async(async_operation()).await;
```

## CI Integration

Example GitHub Actions workflow:

```yaml
name: WASM Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable
        with:
          targets: wasm32-unknown-unknown

      - name: Install wasm-pack
        run: cargo install wasm-pack

      - name: Install Chrome
        uses: browser-actions/setup-chrome@latest

      - name: Run WASM tests
        run: |
          cd reasonkit-web
          wasm-pack test --headless --chrome
```

## Troubleshooting

### Tests hang
- Ensure Chrome/Firefox is installed
- Check for blocking async operations
- Verify network access for fetch tests

### CORS errors in fetch tests
- Tests use httpbin.org which supports CORS
- Local servers need proper CORS headers

### WebSocket connection failures
- echo.websocket.events may be temporarily unavailable
- Tests include timeout handling

## Adding New Tests

```rust
#[wasm_bindgen_test]
async fn test_my_feature() {
    // Async test with browser APIs
    let window = web_sys::window().expect("window");
    // ...
}

#[wasm_bindgen_test]
fn test_sync_feature() {
    // Synchronous test
    assert!(true);
}
```

## Performance Benchmarking

Use the `measure_async` utility for timing:

```rust
#[wasm_bindgen_test]
async fn test_performance() {
    let (_, duration) = measure_async(async {
        // Operation to measure
        fetch_something().await
    }).await;

    assert!(duration < 1000.0, "Should complete in under 1s");
}
```

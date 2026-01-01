# Changelog

All notable changes to `reasonkit-web` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-31

### Added

#### Browser Automation

- **Chromiumoxide Integration**: Headless Chromium control
- **Page Navigation**: URL loading, history navigation
- **Content Extraction**: HTML, text, and structured data extraction
- **Screenshot Capture**: Full page and element screenshots
- **JavaScript Execution**: Arbitrary script execution in page context

#### Anti-Detection Features

- Stealth mode with fingerprint randomization
- WebDriver detection bypass
- Navigator property spoofing
- Canvas fingerprint noise injection
- WebGL renderer masking

#### MCP Sidecar Architecture

- Model Context Protocol server implementation
- Tool registration and discovery
- Structured request/response handling
- Session management

#### CLI Interface

- `reasonkit-web` binary
- Configuration via command-line arguments
- Environment variable support
- Logging with tracing-subscriber

### Optional Features

- `full-stealth`: Enhanced anti-detection measures

### Performance

- Async/await with Tokio runtime
- Efficient memory management
- Connection reuse and pooling

### Documentation

- API documentation
- Usage examples
- Integration guides

---

[0.1.0]: https://github.com/reasonkit/reasonkit-web/releases/tag/reasonkit-web-v0.1.0

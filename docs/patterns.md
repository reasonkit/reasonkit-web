# ReasonKit Web Integration Patterns

> **Version:** 0.1.0
> **Focus:** Common use cases and architectural patterns for integrating ReasonKit Web.

## Pattern 1: The Research Agent

This pattern uses ReasonKit Web as the primary information gathering tool for a research agent. The agent alternates between searching/navigating and reading/extracting.

### Workflow

1.  **Search:** Agent uses `web_navigate` to a search engine (e.g., Google, Bing).
2.  **Analyze Results:** Agent uses `web_extract_links` to find relevant result URLs.
3.  **Deep Dive:** For each relevant URL:
    - `web_navigate` to the URL.
    - `web_extract_content` (Markdown format) to read the page.
    - `web_extract_metadata` to get author/date info.
4.  **Synthesize:** Agent combines extracted content into a summary.

### Example Sequence (JSON-RPC)

```json
// 1. Navigate to search
{"method": "tools/call", "params": {"name": "web_navigate", "arguments": {"url": "https://www.google.com/search?q=rust+mcp+server"}}}

// 2. Extract links
{"method": "tools/call", "params": {"name": "web_extract_links", "arguments": {"url": "current", "selector": "#search"}}}

// 3. Navigate to result
{"method": "tools/call", "params": {"name": "web_navigate", "arguments": {"url": "https://modelcontextprotocol.io"}}}

// 4. Extract content
{"method": "tools/call", "params": {"name": "web_extract_content", "arguments": {"url": "current", "format": "markdown"}}}
```

## Pattern 2: The Visual Validator

This pattern is useful for frontend testing or design validation agents. It relies heavily on screenshots and visual data.

### Workflow

1.  **Navigate:** Go to the target web application.
2.  **Capture State:** Take a `web_screenshot` of the initial state.
3.  **Action:** Use `web_execute_js` to trigger an interaction (e.g., click a button, fill a form).
4.  **Wait:** Implicitly handled by `web_execute_js` promise resolution or explicit `waitFor` in navigation.
5.  **Verify:** Take another `web_screenshot` to verify the UI change.

### Best Practices

- Use `fullPage: true` for design reviews.
- Use specific `selector` screenshots for component testing.
- Combine with a Vision-Language Model (VLM) like Claude 3.5 Sonnet to analyze the images.

## Pattern 3: The Archivist

This pattern is for compliance, auditing, or data preservation agents. It focuses on capturing high-fidelity records of web pages.

### Workflow

1.  **Discovery:** Agent identifies a list of URLs to archive.
2.  **Forensic Capture:** For each URL:
    - `web_navigate` to ensure the page loads.
    - `web_capture_mhtml` to get a single-file archive of all resources (HTML, CSS, Images).
    - `web_pdf` to get a printable, immutable document version.
    - `web_extract_metadata` to log the timestamp and original metadata.
3.  **Storage:** Save the artifacts (MHTML, PDF, JSON metadata) to long-term storage (S3, reasonkit-mem, etc.).

## Pattern 4: The Data Scraper (Structured)

This pattern extracts structured data (tables, lists, specific fields) from unstructured web pages.

### Workflow

1.  **Navigate:** Go to the page containing data.
2.  **Schema Injection:** Agent constructs a JavaScript function to traverse the DOM and extract specific fields into a JSON object.
3.  **Execution:** Use `web_execute_js` to run the extraction script.
    - _Why JS?_ It's often more reliable/precise for structured data than converting the whole page to Markdown and asking the LLM to parse it back out.
4.  **Validation:** Agent validates the returned JSON structure.

### Example JS Payload

```javascript
// Passed to web_execute_js
Array.from(document.querySelectorAll("table.data tr")).map((row) => {
  const cells = row.querySelectorAll("td");
  return {
    id: cells[0]?.innerText,
    name: cells[1]?.innerText,
    status: cells[2]?.innerText,
  };
});
```

## Pattern 5: The Session Manager (Authenticated)

Handling authenticated sessions (login walls).

### Approaches

1.  **Pre-authenticated Profile:**
    - Launch Chrome manually with a specific user data directory.
    - Log in to the required services.
    - Point `reasonkit-web` to use that existing user data directory via environment variables or arguments (if supported by your specific deployment) or by ensuring the `CHROME_PATH` uses the profile.
    - _Note:_ Currently, `reasonkit-web` starts fresh sessions by default. For persistent sessions, you may need to modify the browser launch arguments in `src/browser/mod.rs` to point to a user data dir.

2.  **Agent Login:**
    - Agent navigates to login page.
    - Agent uses `web_execute_js` to fill username/password fields (retrieved securely from env/secrets, NEVER hardcoded).
    - Agent submits form.
    - Agent handles 2FA (if possible, or flags for human intervention).

---

## Error Handling Patterns

- **Retry Logic:** If `web_navigate` fails (timeout/network), implement an exponential backoff retry in the agent logic.
- **Fallback:** If `web_extract_content` (Markdown) is messy/empty, try `web_extract_content` (Text) or `web_screenshot` + OCR.
- **Stealth:** If blocked (403/Captcha), ensure the underlying browser is using stealth plugins (ReasonKit Web does this by default, but aggressive blocking may require slower interactions).

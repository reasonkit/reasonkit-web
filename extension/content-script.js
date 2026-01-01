/**
 * ReasonKit Web Connector - Content Script
 *
 * This content script:
 * 1. Loads the WASM module
 * 2. Initializes DOM observation
 * 3. Captures page content on changes
 * 4. Sends captures to the MCP server
 *
 * @version 0.1.0
 * @license Apache-2.0
 */

'use strict';

(async function() {
  // Prevent double initialization
  if (window.__REASONKIT_INITIALIZED__) {
    console.debug('[ReasonKit] Already initialized, skipping');
    return;
  }
  window.__REASONKIT_INITIALIZED__ = true;

  console.log('[ReasonKit] Content script loaded');

  // Global state
  let wasmModule = null;
  let isObserving = false;
  let lastCaptureTime = 0;
  const MIN_CAPTURE_INTERVAL_MS = 1000; // Minimum 1 second between captures

  /**
   * Load and initialize the WASM module.
   */
  async function initializeWasm() {
    try {
      // Dynamic import of WASM module
      const wasmUrl = chrome.runtime.getURL('pkg/reasonkit_web_rs.js');
      const module = await import(wasmUrl);

      // Initialize WASM
      await module.default();

      wasmModule = module;
      console.log('[ReasonKit] WASM module initialized');

      return true;
    } catch (error) {
      console.error('[ReasonKit] Failed to load WASM module:', error);
      return false;
    }
  }

  /**
   * Get configuration from background script.
   */
  async function getConfig() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: 'GET_CONFIG' }, (response) => {
        resolve(response.config || {});
      });
    });
  }

  /**
   * Get auth token from background script.
   */
  async function getAuthToken() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: 'GET_AUTH_TOKEN' }, (response) => {
        resolve(response.token);
      });
    });
  }

  /**
   * Notify background of capture completion.
   */
  async function notifyCaptureComplete(result) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({
        type: 'CAPTURE_COMPLETE',
        result: result
      }, resolve);
    });
  }

  /**
   * Capture the current page state.
   */
  async function capturePage() {
    // Rate limiting
    const now = Date.now();
    if (now - lastCaptureTime < MIN_CAPTURE_INTERVAL_MS) {
      console.debug('[ReasonKit] Rate limited, skipping capture');
      return null;
    }
    lastCaptureTime = now;

    // Check if WASM is available
    if (!wasmModule) {
      console.warn('[ReasonKit] WASM module not loaded');
      return null;
    }

    try {
      const url = window.location.href;
      const domContent = document.documentElement.outerHTML;

      console.debug('[ReasonKit] Capturing page:', url.substring(0, 50) + '...');

      const resultJson = await wasmModule.capture_page(url, domContent);
      const result = JSON.parse(resultJson);

      console.debug('[ReasonKit] Capture result:', result);

      // Notify background script
      await notifyCaptureComplete(result);

      return result;
    } catch (error) {
      console.error('[ReasonKit] Capture failed:', error);

      const errorResult = {
        capture_id: '',
        success: false,
        error: error.message,
        content_size_bytes: 0,
        capture_duration_ms: 0,
        server_response_ms: null
      };

      await notifyCaptureComplete(errorResult);
      return errorResult;
    }
  }

  /**
   * Set up MutationObserver for DOM changes.
   */
  function setupMutationObserver(debounceMs) {
    let debounceTimer = null;

    const observer = new MutationObserver((mutations) => {
      // Clear existing timer
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }

      // Debounce captures
      debounceTimer = setTimeout(() => {
        console.debug('[ReasonKit] DOM mutations detected:', mutations.length);
        capturePage();
      }, debounceMs);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true,
      attributeFilter: ['class', 'id', 'href', 'src']
    });

    console.log('[ReasonKit] MutationObserver started');
    return observer;
  }

  /**
   * Set up Page Visibility API handler.
   */
  function setupVisibilityHandler(observer) {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        console.debug('[ReasonKit] Page became visible');
        if (!isObserving && observer) {
          observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            characterData: true
          });
          isObserving = true;
        }
        // Capture on visibility restore
        capturePage();
      } else {
        console.debug('[ReasonKit] Page became hidden');
        if (isObserving && observer) {
          observer.disconnect();
          isObserving = false;
        }
      }
    });
  }

  /**
   * Main initialization.
   */
  async function initialize() {
    try {
      // Load configuration
      const config = await getConfig();
      console.log('[ReasonKit] Configuration loaded:', config.server_url);

      // Load WASM module
      const wasmLoaded = await initializeWasm();
      if (!wasmLoaded) {
        console.error('[ReasonKit] Failed to initialize WASM, running in degraded mode');
        // Could implement fallback JavaScript-only capture here
        return;
      }

      // Configure WASM module
      if (wasmModule && wasmModule.configure) {
        wasmModule.configure(JSON.stringify(config));
      }

      // Set auth token
      const token = await getAuthToken();
      if (token && wasmModule && wasmModule.set_auth_token) {
        wasmModule.set_auth_token(token);
        console.log('[ReasonKit] Auth token configured');
      }

      // Initial page capture
      await capturePage();

      // Set up DOM observation
      const observer = setupMutationObserver(config.capture_debounce_ms || 500);
      isObserving = true;

      // Set up visibility handler
      setupVisibilityHandler(observer);

      console.log('[ReasonKit] Initialization complete');
    } catch (error) {
      console.error('[ReasonKit] Initialization failed:', error);
    }
  }

  // Start initialization
  initialize();
})();

/**
 * ReasonKit Web Connector - Background Service Worker
 *
 * This service worker manages:
 * - Authentication token storage
 * - Extension lifecycle
 * - Message passing between content scripts and popup
 *
 * @version 0.1.0
 * @license Apache-2.0
 */

'use strict';

// Default configuration
const DEFAULT_CONFIG = {
  server_url: 'http://127.0.0.1:9753',
  capture_debounce_ms: 500,
  max_dom_size_bytes: 5242880, // 5MB
  exclude_selectors: [],
  include_only_selectors: null,
  enable_compression: true,
  compression_threshold_bytes: 10240 // 10KB
};

/**
 * Initialize the extension on install.
 */
chrome.runtime.onInstalled.addListener(async (details) => {
  console.log('[ReasonKit] Extension installed:', details.reason);

  // Set default configuration
  const existing = await chrome.storage.local.get('reasonkit_config');
  if (!existing.reasonkit_config) {
    await chrome.storage.local.set({ reasonkit_config: DEFAULT_CONFIG });
    console.log('[ReasonKit] Default configuration set');
  }
});

/**
 * Handle messages from content scripts and popup.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[ReasonKit] Message received:', message.type, 'from:', sender.tab?.url || 'popup');

  switch (message.type) {
    case 'GET_AUTH_TOKEN':
      handleGetAuthToken(sendResponse);
      return true; // Keep channel open for async response

    case 'SET_AUTH_TOKEN':
      handleSetAuthToken(message.token, sendResponse);
      return true;

    case 'CLEAR_AUTH_TOKEN':
      handleClearAuthToken(sendResponse);
      return true;

    case 'GET_CONFIG':
      handleGetConfig(sendResponse);
      return true;

    case 'SET_CONFIG':
      handleSetConfig(message.config, sendResponse);
      return true;

    case 'GET_STATUS':
      handleGetStatus(sendResponse);
      return true;

    case 'CAPTURE_COMPLETE':
      handleCaptureComplete(message.result, sender.tab, sendResponse);
      return true;

    default:
      console.warn('[ReasonKit] Unknown message type:', message.type);
      sendResponse({ error: 'Unknown message type' });
      return false;
  }
});

/**
 * Get the authentication token from secure storage.
 */
async function handleGetAuthToken(sendResponse) {
  try {
    const result = await chrome.storage.local.get('reasonkit_auth_token');
    sendResponse({ token: result.reasonkit_auth_token || null });
  } catch (error) {
    console.error('[ReasonKit] Failed to get auth token:', error);
    sendResponse({ error: error.message });
  }
}

/**
 * Set the authentication token in secure storage.
 */
async function handleSetAuthToken(token, sendResponse) {
  try {
    if (!token || typeof token !== 'string') {
      throw new Error('Invalid token');
    }
    await chrome.storage.local.set({ reasonkit_auth_token: token });
    console.log('[ReasonKit] Auth token updated');
    sendResponse({ success: true });
  } catch (error) {
    console.error('[ReasonKit] Failed to set auth token:', error);
    sendResponse({ error: error.message });
  }
}

/**
 * Clear the authentication token from storage.
 */
async function handleClearAuthToken(sendResponse) {
  try {
    await chrome.storage.local.remove('reasonkit_auth_token');
    console.log('[ReasonKit] Auth token cleared');
    sendResponse({ success: true });
  } catch (error) {
    console.error('[ReasonKit] Failed to clear auth token:', error);
    sendResponse({ error: error.message });
  }
}

/**
 * Get the current configuration.
 */
async function handleGetConfig(sendResponse) {
  try {
    const result = await chrome.storage.local.get('reasonkit_config');
    sendResponse({ config: result.reasonkit_config || DEFAULT_CONFIG });
  } catch (error) {
    console.error('[ReasonKit] Failed to get config:', error);
    sendResponse({ error: error.message });
  }
}

/**
 * Update the configuration.
 */
async function handleSetConfig(config, sendResponse) {
  try {
    // Merge with defaults to ensure all required fields exist
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    await chrome.storage.local.set({ reasonkit_config: mergedConfig });
    console.log('[ReasonKit] Configuration updated');
    sendResponse({ success: true, config: mergedConfig });
  } catch (error) {
    console.error('[ReasonKit] Failed to set config:', error);
    sendResponse({ error: error.message });
  }
}

/**
 * Get the extension and server status.
 */
async function handleGetStatus(sendResponse) {
  try {
    const [configResult, tokenResult, statsResult] = await Promise.all([
      chrome.storage.local.get('reasonkit_config'),
      chrome.storage.local.get('reasonkit_auth_token'),
      chrome.storage.local.get('reasonkit_stats')
    ]);

    const config = configResult.reasonkit_config || DEFAULT_CONFIG;
    const hasToken = !!tokenResult.reasonkit_auth_token;
    const stats = statsResult.reasonkit_stats || {
      captures_total: 0,
      captures_success: 0,
      captures_failed: 0,
      last_capture_at: null
    };

    // Check server connectivity
    let serverConnected = false;
    let serverVersion = null;

    try {
      const response = await fetch(`${config.server_url}/api/v1/status`, {
        method: 'GET',
        headers: {
          'X-ReasonKit-Client': 'extension',
          'X-ReasonKit-Version': chrome.runtime.getManifest().version
        }
      });

      if (response.ok) {
        const data = await response.json();
        serverConnected = true;
        serverVersion = data.version;
      }
    } catch (e) {
      // Server not reachable
    }

    sendResponse({
      extension_version: chrome.runtime.getManifest().version,
      has_auth_token: hasToken,
      server_connected: serverConnected,
      server_version: serverVersion,
      server_url: config.server_url,
      stats: stats
    });
  } catch (error) {
    console.error('[ReasonKit] Failed to get status:', error);
    sendResponse({ error: error.message });
  }
}

/**
 * Handle capture completion notification from content script.
 */
async function handleCaptureComplete(result, tab, sendResponse) {
  try {
    // Update statistics
    const statsResult = await chrome.storage.local.get('reasonkit_stats');
    const stats = statsResult.reasonkit_stats || {
      captures_total: 0,
      captures_success: 0,
      captures_failed: 0,
      last_capture_at: null
    };

    stats.captures_total++;
    if (result.success) {
      stats.captures_success++;
      stats.last_capture_at = new Date().toISOString();
    } else {
      stats.captures_failed++;
    }

    await chrome.storage.local.set({ reasonkit_stats: stats });

    // Update badge to show activity
    if (tab?.id) {
      chrome.action.setBadgeText({
        text: result.success ? '' : '!',
        tabId: tab.id
      });

      if (!result.success) {
        chrome.action.setBadgeBackgroundColor({
          color: '#f97316', // Orange for error
          tabId: tab.id
        });
      }
    }

    console.log('[ReasonKit] Capture complete:', result.capture_id, 'success:', result.success);
    sendResponse({ success: true });
  } catch (error) {
    console.error('[ReasonKit] Failed to handle capture complete:', error);
    sendResponse({ error: error.message });
  }
}

console.log('[ReasonKit] Background service worker loaded');

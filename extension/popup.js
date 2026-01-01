/**
 * ReasonKit Web Connector - Popup Script
 *
 * Handles the extension popup UI interactions.
 *
 * @version 0.1.0
 * @license Apache-2.0
 */

'use strict';

document.addEventListener('DOMContentLoaded', async () => {
  // UI Elements
  const versionEl = document.getElementById('version');
  const serverStatusEl = document.getElementById('server-status');
  const serverUrlEl = document.getElementById('server-url');
  const authStatusEl = document.getElementById('auth-status');
  const statTotalEl = document.getElementById('stat-total');
  const statSuccessEl = document.getElementById('stat-success');
  const statFailedEl = document.getElementById('stat-failed');
  const authTokenInput = document.getElementById('auth-token');
  const saveTokenBtn = document.getElementById('save-token');
  const clearTokenBtn = document.getElementById('clear-token');
  const captureNowBtn = document.getElementById('capture-now');
  const refreshStatusBtn = document.getElementById('refresh-status');
  const messageEl = document.getElementById('message');

  /**
   * Show a message to the user.
   */
  function showMessage(text, type = 'success') {
    messageEl.textContent = text;
    messageEl.className = `message ${type} visible`;
    setTimeout(() => {
      messageEl.classList.remove('visible');
    }, 3000);
  }

  /**
   * Update the status display.
   */
  async function updateStatus() {
    try {
      const status = await chrome.runtime.sendMessage({ type: 'GET_STATUS' });

      // Version
      versionEl.textContent = `v${status.extension_version}`;

      // Server status
      if (status.server_connected) {
        serverStatusEl.innerHTML = `
          <span class="status-indicator connected"></span>
          Connected${status.server_version ? ` (${status.server_version})` : ''}
        `;
        serverStatusEl.className = 'status-value connected';
      } else {
        serverStatusEl.innerHTML = `
          <span class="status-indicator disconnected"></span>
          Disconnected
        `;
        serverStatusEl.className = 'status-value disconnected';
      }

      // Server URL
      serverUrlEl.textContent = status.server_url || '-';

      // Auth status
      authStatusEl.textContent = status.has_auth_token ? 'Configured' : 'Not Set';
      authStatusEl.style.color = status.has_auth_token ? '#10b981' : '#f97316';

      // Statistics
      if (status.stats) {
        statTotalEl.textContent = status.stats.captures_total || 0;
        statSuccessEl.textContent = status.stats.captures_success || 0;
        statFailedEl.textContent = status.stats.captures_failed || 0;
      }
    } catch (error) {
      console.error('[ReasonKit] Failed to get status:', error);
      showMessage('Failed to get status', 'error');
    }
  }

  /**
   * Save the authentication token.
   */
  async function saveToken() {
    const token = authTokenInput.value.trim();
    if (!token) {
      showMessage('Please enter a token', 'error');
      return;
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: 'SET_AUTH_TOKEN',
        token: token
      });

      if (response.success) {
        showMessage('Token saved successfully');
        authTokenInput.value = '';
        await updateStatus();
      } else {
        throw new Error(response.error || 'Failed to save token');
      }
    } catch (error) {
      console.error('[ReasonKit] Failed to save token:', error);
      showMessage('Failed to save token', 'error');
    }
  }

  /**
   * Clear the authentication token.
   */
  async function clearToken() {
    try {
      const response = await chrome.runtime.sendMessage({ type: 'CLEAR_AUTH_TOKEN' });

      if (response.success) {
        showMessage('Token cleared');
        authTokenInput.value = '';
        await updateStatus();
      } else {
        throw new Error(response.error || 'Failed to clear token');
      }
    } catch (error) {
      console.error('[ReasonKit] Failed to clear token:', error);
      showMessage('Failed to clear token', 'error');
    }
  }

  /**
   * Trigger a manual capture on the current tab.
   */
  async function captureNow() {
    try {
      // Get the current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab) {
        showMessage('No active tab', 'error');
        return;
      }

      // Execute capture in the content script
      await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_NOW' });
      showMessage('Capture triggered');

      // Refresh status after a short delay
      setTimeout(updateStatus, 1000);
    } catch (error) {
      console.error('[ReasonKit] Failed to trigger capture:', error);
      showMessage('Failed to trigger capture', 'error');
    }
  }

  // Event listeners
  saveTokenBtn.addEventListener('click', saveToken);
  clearTokenBtn.addEventListener('click', clearToken);
  captureNowBtn.addEventListener('click', captureNow);
  refreshStatusBtn.addEventListener('click', updateStatus);

  // Allow Enter key to save token
  authTokenInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      saveToken();
    }
  });

  // Initial status update
  await updateStatus();
});
